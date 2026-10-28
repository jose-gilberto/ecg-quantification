"""Adapter exposing ECGFounder (a foundation model for ECG signals) as a
`quack`-compatible classifier (Fase 4, item 15).

ECGFounder (Li et al., NEJM AI 2025; https://github.com/PKUDigitalHealth/ECGFounder)
is a Net1D convolutional network pretrained on >10M ECGs from the
Harvard-Emory ECG Database (150 diagnostic labels). This project uses the
**single-lead** checkpoint (`1_lead_ECGFounder.pth`, Hugging Face repo
`PKUDigitalHealth/ECGFounder`), since MIT-BIH beat segments are
single-lead. The architecture is vendored in `_net1d.py` (MIT license);
the pretrained weights are downloaded/loaded at fit time.

Like `LITETimeClassifier`, the model is wrapped in
`quack.utils.SklearnClassifierWrapper`, which stores a *recipe*
(`functools.partial`, picklable) rather than a live model, so `clone()`
works throughout `quack` (per-CV-fold refits, per-class OvR members).

Two training modes (`freeze_backbone`)
--------------------------------------
- `freeze_backbone=True` (default) -- **linear probe**: the pretrained
  backbone is frozen and used as a feature extractor (its 1024-d global
  average-pooled features, the input of the original 150-class head);
  a multinomial logistic regression (after standardization) is fit on
  top. Cheap enough for the ~66 refits `build_all_quantifiers` needs, and
  the standard way to evaluate a foundation model's representations.
- `freeze_backbone=False` -- **full fine-tuning**, following the
  official fine-tuning notebook (Adam, lr 1e-4, weight decay 1e-5,
  pretrained backbone + fresh linear head), with cross-entropy for our
  single-label multiclass task. Optionally keeps the epoch with the
  lowest loss on a stratified validation split (`validation_fraction`).

Input preprocessing
-------------------
The official README insists on reproducing its preprocessing (500 Hz;
50 Hz notch, 0.67-40 Hz Butterworth band-pass and 0.4 s median-filter
baseline removal, all via `filtfilt`; z-score). Every input is therefore
(1) resampled from `input_sampling_rate` (MIT-BIH: 360 Hz) to 500 Hz with
a polyphase anti-aliasing filter, so the convolution kernels see the
same physical time scales as in pretraining; (2) filtered with the exact
same filter chain (`apply_ecgfounder_filter=True`); (3) z-scored per
segment. Two deliberate deviations, forced by this project's data:

- Segments are single beats (`ECGPreprocessor`'s default 1 s windows),
  not 10 s recordings. They are *not* zero-padded to 5000 samples: Net1D
  is fully convolutional with global average pooling, so a 500-sample
  input is valid, and padding would mostly feed it silence.
- The filters run on each 1 s window rather than on the whole record
  (the splits store windows, not records). The band-pass is unaffected
  in the window's centre, where the beat is; the median baseline filter
  zero-pads at the window edges (as `scipy.signal.medfilt` does in the
  original code), slightly distorting the outer ~0.2 s. Set
  `apply_ecgfounder_filter=False` to ablate this step.

MIT-BIH's lead is (modified) lead II, while the single-lead checkpoint was
pretrained on lead I: a domain shift that fine-tuning can absorb but a
linear probe cannot -- worth stating in the paper.

Resumable training (`training_checkpoint_root`, fine-tuning only)
-----------------------------------------------------------------
Mirrors the role of item 12's per-epoch checkpointing, in PyTorch. Each
training subset gets its own directory, named by a content hash of
`(X, y)` plus every hyperparameter that affects training -- the same
deterministic key `CachedFitClassifier` uses, so it is stable across
process restarts and distinct per CV fold / OvR member:

- after every epoch, `resume.pt` (model, optimizer, epoch, best
  validation loss and weights, per-epoch shuffling is seeded by epoch)
  -- an interrupted training continues from the last finished epoch;
- when training finishes, `final.pt` (the selected weights only, ~123 MB)
  replaces it -- a restarted run then *loads* that training instead of
  repeating it, even when the in-memory fit cache starts empty.

Fitted fine-tuned models also pickle as a reference to their `final.pt`
(not a copy of the weights) whenever it exists, so `ExperimentCheckpoint`
does not store one 123 MB copy per quantifier. Linear-probe models never
pickle the backbone (it is reloaded from the pretrained checkpoint), only
the small logistic-regression head.

Disk/memory guide (full fine-tuning, cv=10, 5 classes, ~66 trainings):
~8 GB of `final.pt` files, and up to the same in RAM while the fitted
models are alive in one process. The linear probe needs neither.
"""
import functools
import hashlib
import json
import logging
import math
import os
import pickle
import sys
import time

import numpy as np
from quack.utils.wrappers import SklearnClassifierWrapper


logger = logging.getLogger("ecg_quantification.ecg_founder")

HF_REPO_ID = "PKUDigitalHealth/ECGFounder"
HF_SINGLE_LEAD_FILENAME = "1_lead_ECGFounder.pth"
ECGFOUNDER_SAMPLING_RATE = 500

# Net1D hyperparameters of the released checkpoints (finetune_model.py,
# ft_1lead_ECGFounder); n_classes is set per task.
_NET1D_ARCHITECTURE = dict(
  in_channels=1,
  base_filters=64,
  ratio=1,
  filter_list=[64, 160, 160, 400, 400, 1024, 1024],
  m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
  kernel_size=16,
  stride=2,
  groups_width=16,
  verbose=False,
  use_bn=False,
  use_do=False,
)

# per-process memo: pretrained backbone state_dicts (by absolute path)
# and frozen feature extractors already moved to a device
_PRETRAINED_STATES: dict = {}
_FROZEN_BACKBONES: dict = {}


# --------------------------------------------------------------------------
# devices and checkpoints
# --------------------------------------------------------------------------

def _import_torch():
  """Imports PyTorch so it can share a process with TensorFlow (e.g.
  `--classifiers LITETime ECGFounder` in one run).

  Loading Triton (PyTorch's compiler backend, pulled in lazily by
  `torch._dynamo` -- e.g. on the first `torch.optim` optimizer) *after*
  TensorFlow segfaults the interpreter: both ship their own LLVM, and the
  symbols clash. This adapter only runs PyTorch eagerly (no
  `torch.compile`), so when TensorFlow is already loaded and Triton is
  not, Triton is marked unavailable: `import triton` then raises
  ImportError, which PyTorch handles as "no Triton". The reverse order
  (Triton first, then TensorFlow) is safe and left alone.
  """
  if "tensorflow" in sys.modules and "triton" not in sys.modules:
    sys.modules["triton"] = None
  import torch
  return torch


def configure_torch_device(device: str = "cuda") -> list[str]:
  """Checks that PyTorch can use `device`, failing fast with a diagnosis
  instead of silently training a 30M-parameter network on the CPU.

  Parameters
  ----------
  device : {"cuda", "cpu"}, default = "cuda"

  Returns
  -------
  devices : list of str
    Human-readable names of the device(s) that will be used.

  Raises
  ------
  ValueError
    If `device` is not "cuda" or "cpu".
  RuntimeError
    If `device="cuda"` and PyTorch sees no CUDA GPU.
  """
  if device not in ("cuda", "cpu"):
    raise ValueError(f"device must be 'cuda' or 'cpu', got {device!r}.")
  torch = _import_torch()

  if device == "cpu":
    return ["CPU"]
  if not torch.cuda.is_available():
    if torch.version.cuda is None:
      reason = (
        "the installed PyTorch build is CPU-only. Install a CUDA build, e.g. "
        "`pip install torch --index-url https://download.pytorch.org/whl/cu126` "
        "(pick the CUDA version your NVIDIA driver supports)."
      )
    else:
      reason = (
        f"PyTorch was built for CUDA {torch.version.cuda} but sees no GPU: the "
        f"NVIDIA driver may be too old for that CUDA version (check `nvidia-smi`; "
        f"if so, install a torch build for an older CUDA from "
        f"https://download.pytorch.org/whl/), or CUDA_VISIBLE_DEVICES hides it."
      )
    raise RuntimeError(
      f"device='cuda' was requested but no GPU is visible to PyTorch: {reason}\n"
      f"To run ECGFounder on the CPU on purpose, pass device='cpu' "
      f"(`--ecg-founder-device cpu` in the experiment scripts)."
    )
  return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


@functools.lru_cache(maxsize=None)
def resolve_ecg_founder_checkpoint(checkpoint_path: str = None) -> str:
  """Returns a local path to the single-lead ECGFounder weights.

  Resolution order: `checkpoint_path` if given; else the
  `ECGFOUNDER_CHECKPOINT` environment variable; else a download of
  `1_lead_ECGFounder.pth` (370 MB, cached by `huggingface_hub` under
  `~/.cache/huggingface`) from the `PKUDigitalHealth/ECGFounder` repo.
  """
  checkpoint_path = checkpoint_path or os.environ.get("ECGFOUNDER_CHECKPOINT")
  if checkpoint_path:
    if not os.path.isfile(checkpoint_path):
      raise FileNotFoundError(f"ECGFounder checkpoint not found: {checkpoint_path}")
    return os.path.abspath(checkpoint_path)
  try:
    from huggingface_hub import hf_hub_download
  except ImportError as error:
    raise ImportError(
      "No ECGFounder checkpoint given and huggingface_hub is not installed. Either "
      "pass checkpoint_path (--ecg-founder-checkpoint) / set ECGFOUNDER_CHECKPOINT to "
      f"a downloaded {HF_SINGLE_LEAD_FILENAME}, or install the extra: "
      "pip install -e \".[ecgfounder]\""
    ) from error
  return hf_hub_download(repo_id=HF_REPO_ID, filename=HF_SINGLE_LEAD_FILENAME)


def _numpy_safe_globals(with_legacy_names: bool = True) -> list:
  """numpy objects that may appear in a checkpoint's metadata or optimizer
  state (e.g. a numpy-float `val_auroc`, or a small array), allowlisted
  for `torch.load(weights_only=True)` -- they only rebuild plain data
  (scalars, dtypes, arrays), they cannot execute arbitrary code.

  `torch.load` matches allowlisted globals by the dotted name stored in the
  file. A checkpoint pickled under numpy 1.x -- the official ECGFounder
  release is one -- stores `numpy.core.multiarray.scalar`, while under
  numpy 2.x the very same function reports `numpy._core.multiarray.scalar`.
  So each one is also registered under its numpy 1.x name, using torch's
  `(object, "name in the file")` form (`with_legacy_names=False` drops
  those, for PyTorch versions that don't accept that form).
  """
  try:
    from numpy._core import multiarray
  except ImportError:  # numpy 1.x
    from numpy.core import multiarray

  safe = [np.dtype, np.ndarray, multiarray.scalar, multiarray._reconstruct]
  for name in ("Float64DType", "Float32DType", "Int64DType", "Int32DType", "BoolDType"):
    dtype_class = getattr(getattr(np, "dtypes", None), name, None)
    if dtype_class is not None:
      safe.append(dtype_class)
  if with_legacy_names:
    for function in (multiarray.scalar, multiarray._reconstruct):
      legacy = f"numpy.core.multiarray.{function.__name__}"
      if f"{function.__module__}.{function.__name__}" != legacy:
        safe.append((function, legacy))
  return safe


def _torch_load(path: str, trust_checkpoint: bool):
  """`torch.load` restricted to tensors/plain data (`weights_only=True`)
  unless `trust_checkpoint` -- unpickling an arbitrary file can execute
  code."""
  torch = _import_torch()

  try:
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:  # PyTorch < 2.5: no allowlist API
      return torch.load(path, map_location="cpu", weights_only=True)
    try:
      allowlist = _numpy_safe_globals()
      with safe_globals(allowlist):
        return torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, ValueError, AttributeError):
      # this PyTorch doesn't take (object, "name") entries
      with safe_globals(_numpy_safe_globals(with_legacy_names=False)):
        return torch.load(path, map_location="cpu", weights_only=True)
  except pickle.UnpicklingError as error:
    if not trust_checkpoint:
      raise RuntimeError(
        f"{path} contains objects beyond tensors/plain data, so it could not be "
        f"loaded safely (torch.load(weights_only=True)): {error}\n"
        f"If you trust this file (e.g. it is the official ECGFounder release), pass "
        f"trust_checkpoint=True (`--ecg-founder-trust-checkpoint`)."
      ) from error
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_pretrained_backbone_state(checkpoint_path: str, trust_checkpoint: bool = False) -> dict:
  """Loads and validates the pretrained backbone weights (everything but
  the 150-class `dense` head), memoized per process.

  Unlike the official `ft_1lead_ECGFounder` (`load_state_dict(...,
  strict=False)`, which silently ignores *any* mismatch -- a wrong file
  would quietly leave the network randomly initialized), every backbone
  parameter must be present with the right shape, and nothing unexpected
  may be left over.
  """
  path = os.path.abspath(checkpoint_path)
  if path in _PRETRAINED_STATES:
    return _PRETRAINED_STATES[path]

  checkpoint = _torch_load(path, trust_checkpoint)
  state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
  if not isinstance(state, dict):
    raise ValueError(f"{path} does not contain a state_dict.")
  state = {
    (key[len("module."):] if key.startswith("module.") else key): value
    for key, value in state.items()
  }
  backbone = {key: value for key, value in state.items() if not key.startswith("dense.")}

  first_conv = backbone.get("first_conv.conv.weight")
  if first_conv is not None and first_conv.shape[1] != 1:
    raise ValueError(
      f"{path} is a {first_conv.shape[1]}-lead ECGFounder checkpoint; this project's "
      f"beats are single-lead -- use {HF_SINGLE_LEAD_FILENAME}."
    )

  from ecg_quantification.quantifiers._net1d import Net1D
  reference = Net1D(n_classes=1, **_NET1D_ARCHITECTURE).state_dict()
  expected = {key: value.shape for key, value in reference.items() if not key.startswith("dense.")}
  missing = sorted(set(expected) - set(backbone))
  unexpected = sorted(set(backbone) - set(expected))
  wrong_shape = sorted(
    key for key in set(expected) & set(backbone) if tuple(backbone[key].shape) != tuple(expected[key])
  )
  if missing or unexpected or wrong_shape:
    raise ValueError(
      f"{path} does not match the ECGFounder single-lead Net1D architecture: "
      f"{len(missing)} missing (e.g. {missing[:3]}), {len(unexpected)} unexpected "
      f"(e.g. {unexpected[:3]}), {len(wrong_shape)} with a different shape "
      f"(e.g. {wrong_shape[:3]})."
    )

  backbone = {key: value.detach().float().contiguous() for key, value in backbone.items()}
  _PRETRAINED_STATES[path] = backbone
  return backbone


def _build_network(n_classes: int, backbone_state: dict):
  """Net1D with the pretrained backbone and a freshly initialized
  `n_classes`-way linear head (as `ft_1lead_ECGFounder` does)."""
  from ecg_quantification.quantifiers._net1d import Net1D
  network = Net1D(n_classes=n_classes, **_NET1D_ARCHITECTURE)
  result = network.load_state_dict(backbone_state, strict=False)
  # only the new head may be left at its random initialization
  assert sorted(result.missing_keys) == ["dense.bias", "dense.weight"], result.missing_keys
  assert not result.unexpected_keys, result.unexpected_keys
  return network


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

def _filter_like_ecgfounder(X: np.ndarray, fs: int) -> np.ndarray:
  """ECGFounder's `util.filter_bandpass`, vectorized over rows: 50 Hz
  notch (Q=30), 4th-order 0.67-40 Hz Butterworth band-pass (both
  zero-phase, `filtfilt`), then subtraction of a 0.4 s median-filter
  baseline (zero-padded at the edges, like `scipy.signal.medfilt`)."""
  from scipy.ndimage import median_filter
  from scipy.signal import butter, filtfilt, iirnotch

  b, a = iirnotch(50, 30, fs)
  filtered = filtfilt(b, a, X, axis=-1)
  b, a = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=fs)
  padlen = 3 * max(len(a), len(b))
  if X.shape[-1] <= padlen:
    raise ValueError(
      f"Segments of {X.shape[-1]} samples at {fs} Hz are too short for ECGFounder's "
      f"band-pass filter (needs > {padlen}); pass apply_ecgfounder_filter=False."
    )
  filtered = filtfilt(b, a, filtered, axis=-1)
  kernel_size = int(0.4 * fs) + 1
  if kernel_size % 2 == 0:
    kernel_size += 1
  baseline = median_filter(filtered, size=(1, kernel_size), mode="constant", cval=0.0)
  return filtered - baseline


def prepare_ecg_founder_input(X, input_sampling_rate: int = 360, apply_ecgfounder_filter: bool = True) -> np.ndarray:
  """Turns beat segments into ECGFounder input: shape `(n, 1, length)`,
  float32, at 500 Hz, filtered like ECGFounder's pretraining data and
  z-scored per segment (see the module docstring).

  Parameters
  ----------
  X : array-like of shape (n_samples, window_size) or (n_samples, 1, window_size)
  input_sampling_rate : int, default = 360
    Sampling rate of `X` (MIT-BIH: 360 Hz).
  apply_ecgfounder_filter : bool, default = True
  """
  from scipy.signal import resample_poly

  X = np.asarray(X, dtype=np.float64)
  if X.ndim == 3:
    if X.shape[1] != 1:
      raise ValueError(f"Expected single-lead input, got {X.shape[1]} channels.")
    X = X[:, 0, :]
  if X.ndim != 2:
    raise ValueError(f"Expected X of shape (n_samples, window_size), got {X.shape}.")

  if input_sampling_rate != ECGFOUNDER_SAMPLING_RATE:
    divisor = math.gcd(int(input_sampling_rate), ECGFOUNDER_SAMPLING_RATE)
    X = resample_poly(
      X, ECGFOUNDER_SAMPLING_RATE // divisor, int(input_sampling_rate) // divisor, axis=-1,
    )
  if apply_ecgfounder_filter:
    X = _filter_like_ecgfounder(X, ECGFOUNDER_SAMPLING_RATE)

  X = (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)
  return X[:, None, :].astype(np.float32)


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
  if seconds < 60:
    return f"{seconds:.1f}s"
  hours, rest = divmod(int(round(seconds)), 3600)
  minutes, secs = divmod(rest, 60)
  return f"{hours}:{minutes:02d}:{secs:02d}"


class _ECGFounderModel:
  """One ECGFounder classifier (built fresh by `SklearnClassifierWrapper`
  on every `.fit()`). See `ECGFounderClassifier` for the parameters."""

  def __init__(self,
               checkpoint_path: str = None,
               freeze_backbone: bool = True,
               input_sampling_rate: int = 360,
               apply_ecgfounder_filter: bool = True,
               n_epochs: int = 5,
               batch_size: int = 256,
               learning_rate: float = 1e-4,
               weight_decay: float = 1e-5,
               validation_fraction: float = 0.1,
               device: str = "cuda",
               training_checkpoint_root: str = None,
               trust_checkpoint: bool = False,
               random_state: int = None):
    self.checkpoint_path = checkpoint_path
    self.freeze_backbone = freeze_backbone
    self.input_sampling_rate = input_sampling_rate
    self.apply_ecgfounder_filter = apply_ecgfounder_filter
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    self.validation_fraction = validation_fraction
    self.device = device
    self.training_checkpoint_root = training_checkpoint_root
    self.trust_checkpoint = trust_checkpoint
    self.random_state = random_state

  # -- pickling: never the frozen backbone; fine-tuned weights only when
  # they are not already persisted in a final.pt --------------------------

  def __getstate__(self):
    state = self.__dict__.copy()
    network = state.pop("_network", None)
    if not self.freeze_backbone and hasattr(self, "classes_"):
      if self.__dict__.get("weights_path_") and os.path.isfile(self.weights_path_):
        state.pop("_weights", None)
      elif network is not None:
        state["_weights"] = {k: v.detach().cpu() for k, v in network.state_dict().items()}
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)

  # -- helpers ------------------------------------------------------------

  def _torch_device(self):
    torch = _import_torch()
    configure_torch_device(self.device)
    return torch.device(self.device)

  def _pretrained_path(self) -> str:
    return resolve_ecg_founder_checkpoint(self.checkpoint_path)

  def _seed(self) -> int:
    return 0 if self.random_state is None else int(self.random_state)

  def _frozen_backbone(self):
    """The shared, frozen pretrained network on this model's device."""
    torch = _import_torch()
    path = self._pretrained_path()
    key = (path, self.device)
    if key not in _FROZEN_BACKBONES:
      network = _build_network(1, _load_pretrained_backbone_state(path, self.trust_checkpoint))
      network.return_features = True
      network.eval()
      for parameter in network.parameters():
        parameter.requires_grad_(False)
      _FROZEN_BACKBONES[key] = network.to(self._torch_device())
    return _FROZEN_BACKBONES[key], torch.device(self.device)

  def _batched_forward(self, network, X_input: np.ndarray, device, features: bool) -> np.ndarray:
    torch = _import_torch()
    outputs = []
    network.eval()
    with torch.no_grad():
      for start in range(0, len(X_input), self.batch_size):
        batch = torch.from_numpy(X_input[start:start + self.batch_size]).to(device)
        out = network(batch)
        if features:
          out = out[1]
        outputs.append(out.float().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty((0, 0), dtype=np.float32)

  def _embed(self, X) -> np.ndarray:
    network, device = self._frozen_backbone()
    return self._batched_forward(
      network,
      prepare_ecg_founder_input(X, self.input_sampling_rate, self.apply_ecgfounder_filter),
      device, features=True,
    )

  def _finetuned_network(self):
    if self.__dict__.get("_network") is None:
      torch = _import_torch()
      if self.__dict__.get("_weights") is not None:
        weights = self._weights
      elif self.__dict__.get("weights_path_"):
        if not os.path.isfile(self.weights_path_):
          raise FileNotFoundError(
            f"Fine-tuned ECGFounder weights not found at {self.weights_path_} (was the "
            f"training checkpoint directory moved or deleted?)."
          )
        weights = torch.load(self.weights_path_, map_location="cpu", weights_only=True)["model"]
      else:
        raise RuntimeError("This ECGFounder model has no fitted weights.")
      from ecg_quantification.quantifiers._net1d import Net1D
      network = Net1D(n_classes=len(self.classes_), **_NET1D_ARCHITECTURE)
      network.load_state_dict(weights)
      self._network = network
      self.__dict__.pop("_weights", None)
    return self._network

  def _logits(self, X) -> np.ndarray:
    device = self._torch_device()
    network = self._finetuned_network()
    # kept on the CPU between calls: ~66 fine-tuned networks may be alive
    # at once (fit cache), far more than fits in GPU memory
    network.to(device)
    try:
      return self._batched_forward(
        network,
        prepare_ecg_founder_input(X, self.input_sampling_rate, self.apply_ecgfounder_filter),
        device, features=False,
      )
    finally:
      network.to("cpu")

  # -- sklearn-style API --------------------------------------------------

  def fit(self, X, y):
    y = np.asarray(y)
    self.classes_, y_index = np.unique(y, return_inverse=True)
    if len(self.classes_) < 2:
      raise ValueError("ECGFounder needs at least 2 classes in y.")
    if self.freeze_backbone:
      self._fit_linear_probe(X, y_index)
    else:
      self._fit_finetune(X, y, y_index)
    return self

  def predict_proba(self, X) -> np.ndarray:
    if self.freeze_backbone:
      return self.head_.predict_proba(self._embed(X))
    from scipy.special import softmax
    return softmax(self._logits(X).astype(np.float64), axis=1)

  def predict(self, X) -> np.ndarray:
    return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

  # -- linear probe -------------------------------------------------------

  def _fit_linear_probe(self, X, y_index):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    start = time.time()
    features = self._embed(X)
    embedded = time.time() - start
    self.head_ = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    self.head_.fit(features, y_index)
    logger.info(
      "linear probe: %d beats embedded in %s, head fit in %s",
      len(y_index), _format_duration(embedded), _format_duration(time.time() - start - embedded),
    )

  # -- full fine-tuning ---------------------------------------------------

  def _training_dir(self, X, y) -> str:
    """Stable per-(training subset, hyperparameters) directory."""
    from ecg_quantification.quantifiers._cached_classifier import _fingerprint
    pretrained = self._pretrained_path()
    config = dict(
      pretrained=os.path.basename(pretrained),
      pretrained_bytes=os.path.getsize(pretrained),
      input_sampling_rate=self.input_sampling_rate,
      apply_ecgfounder_filter=self.apply_ecgfounder_filter,
      n_epochs=self.n_epochs, batch_size=self.batch_size,
      learning_rate=self.learning_rate, weight_decay=self.weight_decay,
      validation_fraction=self.validation_fraction, random_state=self._seed(),
    )
    digest = hashlib.sha1(
      (_fingerprint(X, y) + json.dumps(config, sort_keys=True)).encode()
    ).hexdigest()[:16]
    return os.path.join(self.training_checkpoint_root, f"ecgfounder_{digest}")

  def _validation_split(self, y_index):
    n = len(y_index)
    indices = np.arange(n)
    if not self.validation_fraction:
      return indices, None
    counts = np.bincount(y_index)
    n_val = int(round(self.validation_fraction * n))
    if counts.min() < 2 or n_val < len(counts) or n - n_val < len(counts):
      logger.info("no validation split (too few samples per class); keeping the last epoch")
      return indices, None
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
      indices, test_size=n_val, stratify=y_index, random_state=self._seed(),
    )
    return train_idx, val_idx

  def _fit_finetune(self, X, y, y_index):
    torch = _import_torch()
    import torch.nn.functional as F

    device = self._torch_device()
    directory = self._training_dir(X, y) if self.training_checkpoint_root else None
    final_path = os.path.join(directory, "final.pt") if directory else None
    resume_path = os.path.join(directory, "resume.pt") if directory else None

    if final_path and os.path.isfile(final_path):
      # finished in an earlier run: reuse it instead of training again
      saved = torch.load(final_path, map_location="cpu", weights_only=True)
      if list(saved["classes"]) != [str(c) for c in self.classes_]:
        raise RuntimeError(f"{final_path} was trained on different classes.")
      self.weights_path_ = final_path
      logger.info("fine-tuning: loaded finished training from %s", directory)
      return

    torch.manual_seed(self._seed())
    X_input = prepare_ecg_founder_input(X, self.input_sampling_rate, self.apply_ecgfounder_filter)
    train_idx, val_idx = self._validation_split(y_index)
    network = _build_network(
      len(self.classes_), _load_pretrained_backbone_state(self._pretrained_path(), self.trust_checkpoint),
    ).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    first_epoch, best_loss, best_weights = 0, math.inf, None
    if resume_path and os.path.isfile(resume_path):
      saved = torch.load(resume_path, map_location="cpu", weights_only=True)
      network.load_state_dict(saved["model"])
      optimizer.load_state_dict(saved["optimizer"])
      first_epoch, best_loss, best_weights = saved["epoch"] + 1, saved["best_loss"], saved["best_model"]
      logger.info("fine-tuning: resuming after epoch %d/%d from %s", first_epoch, self.n_epochs, directory)

    X_train = torch.from_numpy(X_input[train_idx])
    y_train = torch.from_numpy(y_index[train_idx].astype(np.int64))
    start = time.time()
    for epoch in range(first_epoch, self.n_epochs):
      epoch_start = time.time()
      network.train()
      generator = torch.Generator().manual_seed(self._seed() * 100003 + epoch)
      order = torch.randperm(len(train_idx), generator=generator)
      total = 0.0
      for batch_start in range(0, len(order), self.batch_size):
        batch = order[batch_start:batch_start + self.batch_size]
        logits = network(X_train[batch].to(device))
        loss = F.cross_entropy(logits, y_train[batch].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch)
      train_loss = total / max(len(order), 1)

      if val_idx is not None:
        val_logits = self._batched_forward(network, X_input[val_idx], device, features=False)
        val_loss = float(F.cross_entropy(
          torch.from_numpy(val_logits), torch.from_numpy(y_index[val_idx].astype(np.int64)),
        ))
      else:
        val_loss = train_loss  # no validation: the last epoch wins
      improved = val_loss <= best_loss or val_idx is None
      if improved:
        best_loss = val_loss
        best_weights = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}

      logger.info(
        "fine-tuning epoch %d/%d: train_loss=%.4f%s in %s%s",
        epoch + 1, self.n_epochs, train_loss,
        f" val_loss={val_loss:.4f}" if val_idx is not None else "",
        _format_duration(time.time() - epoch_start), " (best)" if improved and val_idx is not None else "",
      )
      if resume_path:
        os.makedirs(directory, exist_ok=True)
        tmp = resume_path + ".tmp"
        torch.save({
          "epoch": epoch,
          "model": {k: v.detach().cpu() for k, v in network.state_dict().items()},
          "optimizer": optimizer.state_dict(),
          "best_loss": best_loss,
          "best_model": best_weights,
        }, tmp)
        os.replace(tmp, resume_path)  # atomic: never a half-written resume.pt

    if best_weights is None:  # n_epochs == 0
      best_weights = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
    network.load_state_dict(best_weights)
    self._network = network.to("cpu")
    if final_path:
      tmp = final_path + ".tmp"
      torch.save({"model": best_weights, "classes": [str(c) for c in self.classes_]}, tmp)
      os.replace(tmp, final_path)
      if os.path.isfile(resume_path):
        os.remove(resume_path)
      self.weights_path_ = final_path
    else:
      self.weights_path_ = None
    logger.info("fine-tuning done in %s", _format_duration(time.time() - start))


# --------------------------------------------------------------------------
# quack adapter functions (Fase 4, item 15)
# --------------------------------------------------------------------------

def _load_ecg_founder(**kwargs) -> _ECGFounderModel:
  """Builds a fresh, unfitted ECGFounder model (weights load at fit)."""
  return _ECGFounderModel(**kwargs)


def _ecg_founder_fit(model: _ECGFounderModel, X: np.ndarray, y: np.ndarray, **fit_params) -> None:
  """Fits `model` on `(X, y)` (linear probe or fine-tuning, per its
  `freeze_backbone`). No extra fit_params are accepted."""
  if fit_params:
    raise TypeError(
      f"ECGFounderClassifier.fit() does not accept extra fit_params, got: {sorted(fit_params)}. "
      f"Pass hyperparameters to ECGFounderClassifier(...) instead."
    )
  model.fit(X, y)


def _ecg_founder_predict(model: _ECGFounderModel, X: np.ndarray) -> np.ndarray:
  """Hard-label predictions (original labels, e.g. 'N'/'V' or 0/1)."""
  return model.predict(X)


def _ecg_founder_predict_proba(model: _ECGFounderModel, X: np.ndarray) -> np.ndarray:
  """Class probabilities, columns ordered as `model.classes_`."""
  return model.predict_proba(X)


def ECGFounderClassifier(checkpoint_path: str = None,
                         freeze_backbone: bool = True,
                         input_sampling_rate: int = 360,
                         apply_ecgfounder_filter: bool = True,
                         n_epochs: int = 5,
                         batch_size: int = 256,
                         learning_rate: float = 1e-4,
                         weight_decay: float = 1e-5,
                         validation_fraction: float = 0.1,
                         device: str = "cuda",
                         training_checkpoint_root: str = None,
                         trust_checkpoint: bool = False,
                         random_state: int = None,
                         supports_predict_proba: bool = True) -> SklearnClassifierWrapper:
  """Builds a `quack`-compatible classifier wrapping ECGFounder (single-lead).

  Parameters
  ----------
  checkpoint_path : str, default = None
    Local `1_lead_ECGFounder.pth`. `None`: the `ECGFOUNDER_CHECKPOINT`
    environment variable, else downloaded from Hugging Face (requires
    `huggingface_hub`; see `resolve_ecg_founder_checkpoint`).
  freeze_backbone : bool, default = True
    `True`: linear probe (frozen backbone + logistic regression on its
    1024-d features). `False`: full fine-tuning. See the module docstring.
  input_sampling_rate : int, default = 360
    Sampling rate of the beat segments (MIT-BIH: 360 Hz); resampled to
    ECGFounder's 500 Hz.
  apply_ecgfounder_filter : bool, default = True
    Apply ECGFounder's pretraining filter chain to every segment.
  n_epochs, batch_size, learning_rate, weight_decay : default = 5, 256, 1e-4, 1e-5
    Fine-tuning hyperparameters (the official notebook's values).
    `batch_size` is also the inference batch size in both modes.
  validation_fraction : float, default = 0.1
    Fine-tuning only: stratified share of the training data held out to
    keep the epoch with the lowest validation loss. `0` keeps the last
    epoch and trains on everything.
  device : {"cuda", "cpu"}, default = "cuda"
    Fails fast if "cuda" is requested but unavailable
    (`configure_torch_device`).
  training_checkpoint_root : str, default = None
    Fine-tuning only: directory for per-epoch resumable training state
    and finished weights (see the module docstring). `None`: nothing is
    written; an interrupted training restarts from scratch.
  trust_checkpoint : bool, default = False
    Allow `torch.load(weights_only=False)` when the pretrained file holds
    more than tensors/plain data. Only for files you trust.
  random_state : int, default = None
    Seeds weight init of the new head, shuffling and the validation split.
  supports_predict_proba : bool, default = True
    Whether to expose `predict_proba` on the wrapper.

  Returns
  -------
  classifier : SklearnClassifierWrapper
    A fresh, unfitted wrapper; every `.fit()` builds a new model.

  Examples
  --------
  >>> from quack.quantifiers import CC
  >>> from ecg_quantification.quantifiers import ECGFounderClassifier
  >>> classifier = ECGFounderClassifier(device="cpu")
  >>> quantifier = CC(classifier=classifier)
  >>> quantifier.fit(X_train, y_train)  # doctest: +SKIP
  """
  # functools.partial (not a lambda): picklable by reference, so fitted
  # quantifiers built on this classifier can be checkpointed (item 14)
  model_factory = functools.partial(
    _load_ecg_founder,
    checkpoint_path=checkpoint_path,
    freeze_backbone=freeze_backbone,
    input_sampling_rate=input_sampling_rate,
    apply_ecgfounder_filter=apply_ecgfounder_filter,
    n_epochs=n_epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    validation_fraction=validation_fraction,
    device=device,
    training_checkpoint_root=training_checkpoint_root,
    trust_checkpoint=trust_checkpoint,
    random_state=random_state,
  )
  return SklearnClassifierWrapper(
    model_factory=model_factory,
    fit_fn=_ecg_founder_fit,
    predict_fn=_ecg_founder_predict,
    predict_proba_fn=_ecg_founder_predict_proba if supports_predict_proba else None,
    supports_predict_proba=supports_predict_proba,
  )
