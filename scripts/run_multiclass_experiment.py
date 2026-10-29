#!/usr/bin/env python
"""Runs the multiclass (N, V, A, L, R) quantification experiment under
the Clinically-Constrained Artificial Prevalence Protocol (C-APP).

Loads the patient-wise splits produced by `prepare_splits.py`, trains
every registered quantifier (native multiclass + One-vs-Rest wrapped
binary methods, across every requested base classifier) on the training
split, generates clinically-constrained test bags from the held-out
patients via `ClinicalPrevalenceBagGenerator`, and reports both the
standard Mean Absolute Error and the clinically-weighted
Zone-Crossing Error for every quantifier/classifier/bag-size
combination — mirroring the structure of the CBMS'26 Table I, extended
to the full multiclass, clinically-anchored setting.

Checkpointed and resumable: every fitted quantifier and every
(quantifier, bag_size) evaluation is persisted as soon as it completes
(see `ecg_quantification.ExperimentCheckpoint`). Re-running this exact
command after an interruption (power loss, OOM kill, job timeout) picks
up right where it left off instead of refitting/re-evaluating anything
already done. Use `--fresh` to discard an existing checkpoint and start
over instead.

Usage
-----
python scripts/run_multiclass_experiment.py \
    --splits-path data/processed/splits_multiclass.npz \
    --output-dir results/multiclass_capp \
    --n-bags 200 \
    --bag-sizes 100 200 500 \
    --classifiers LR SVM GB RF \
    --random-state 0

# after an interruption, just run the same command again:
python scripts/run_multiclass_experiment.py \
    --splits-path data/processed/splits_multiclass.npz \
    --output-dir results/multiclass_capp \
    --n-bags 200 --bag-sizes 100 200 500 --classifiers LR SVM GB RF --random-state 0

# including LITETime (Fase 3, item 14) -- requires the [deep] extra
# (aeon + GPU-enabled tensorflow); always runs with n_jobs=None (see
# --classifiers' help) and through a fit cache (item 13) regardless of
# --n-jobs. Trains on the GPU by default and stops right away if none is
# visible (pass --lite-time-device cpu to train on the CPU on purpose):
python scripts/run_multiclass_experiment.py \
    --splits-path data/processed/splits_multiclass.npz \
    --output-dir results/multiclass_capp \
    --n-bags 200 --bag-sizes 100 200 500 \
    --classifiers LR SVM GB RF LITETime \
    --lite-time-n-classifiers 5 --lite-time-n-epochs 1500 \
    --random-state 0 2>&1 | tee run.log

# including ECGFounder (Fase 4) -- requires the [ecgfounder] extra
# (PyTorch + huggingface_hub). Linear probe on the frozen pretrained
# backbone by default; weights downloaded once from Hugging Face unless
# --ecg-founder-checkpoint points to a local 1_lead_ECGFounder.pth. Use a
# separate --output-dir from a run with a different --classifiers list
# (the checkpoint's run config must match to resume):
python scripts/run_multiclass_experiment.py \
    --splits-path data/processed/splits_multiclass.npz \
    --output-dir results/multiclass_capp_ecgfounder \
    --n-bags 200 --bag-sizes 100 200 500 \
    --classifiers ECGFounder --no-include-feature-space \
    --ecg-founder-mode linear-probe \
    --random-state 0 2>&1 | tee run_ecgfounder.log

Progress output
---------------
Every line is timestamped and flushed immediately (so `| tee run.log` or
a tmux pane shows it live). Each quantifier gets a `[i/N]` counter, a
"fitting..." line *before* its fit starts and a "fit in ..." / "loaded
from checkpoint" line after. For LITETime, every underlying network-
ensemble training is also logged ("training #k started/done") along with
how many trainings each quantifier cost vs. reused from the fit cache
(ECGFounder likewise, plus one line per fine-tuning epoch) --
a from-scratch run needs about (cv + 1) x (1 + n_classes) trainings in
total (66 with cv=10 and 5 classes), most of them during the first
native (CC) and the first OvR (ACC) quantifier; the rest mostly reuse
cached fits and finish in seconds.
"""
import argparse
import datetime
import logging
import os
import shutil
import time

# before anything can import TensorFlow (some dependencies do so at import
# time): hide its C++ INFO/WARNING start-up chatter (oneDNN, CUDA factory
# registration, ...) so the progress lines below stay readable. Errors
# still show; an explicit TF_CPP_MIN_LOG_LEVEL in the environment wins.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# XLA (which Keras uses to compile LITETime's training step on the GPU)
# autotunes small matrix multiplications with Triton kernels compiled by
# the CUDA `ptxas` it finds on the machine. With an older ptxas (CUDA
# <= 12.6.2, e.g. a system/conda CUDA 12.4 found before the pip one) that
# fails with "Autotuner could not compile any configs for HLO ...
# __triton_gemm" on the very first training. Disabling Triton GEMM makes
# XLA use cuBLAS for those multiplications instead: same results,
# negligible cost here (LITETime's time is in its convolutions, and its
# only dense layer is n_classes wide). Must be set before XLA's first
# compilation; an explicit --xla_gpu_enable_triton_gemm in XLA_FLAGS wins.
_XLA_TRITON_GEMM_FLAG = '--xla_gpu_enable_triton_gemm'
if _XLA_TRITON_GEMM_FLAG not in os.environ.get('XLA_FLAGS', ''):
  os.environ['XLA_FLAGS'] = (
    os.environ.get('XLA_FLAGS', '') + f' {_XLA_TRITON_GEMM_FLAG}=false'
  ).strip()

# The Triton GEMM flag alone only sidesteps the *first* kernel an old
# ptxas cannot compile (the next one, e.g. an ArgMax reduce fusion, then
# fails the same way). The real fix is the ptxas itself: point XLA at the
# one pip installed with tensorflow[and-cuda] (nvidia-cuda-nvcc-cu12)
# instead of whatever older CUDA is found first. main() then checks its
# version before any training (see check_xla_ptxas). Importing
# ecg_quantification here does not load TensorFlow.
from ecg_quantification.quantifiers._lite_time import (  # noqa: E402
  MIN_SAFE_PTXAS_VERSION, configure_xla_ptxas, disable_keras_xla, ptxas_version,
)
XLA_PTXAS = configure_xla_ptxas()

import numpy as np

from ecg_quantification._clinical_bag_generator import capp_bag_generator
from ecg_quantification import (
  CLINICAL_RANGES,
  ZoneCrossingError,
  ExperimentCheckpoint,
  default_base_classifiers,
  build_all_quantifiers,
  build_feature_space_quantifiers,
  LITETimeClassifier,
  ECGFounderClassifier,
  cached_classifier_factory,
  configure_tensorflow_device,
  configure_torch_device,
  resolve_ecg_founder_checkpoint,
  fit_cache_stats,
)

# deep classifiers: fit through a per-classifier fit cache, always with
# n_jobs=None (see build_registry), and reported training by training
DEEP_CLASSIFIERS = ('LITETime', 'ECGFounder')
from quack.metrics import ae


def log(message: str = "") -> None:
  """Prints `message` prefixed with a timestamp (a bare call prints an
  empty separator line), flushing immediately so progress shows up live
  even when stdout is piped (`| tee run.log`), where Python would
  otherwise buffer it in large blocks."""
  if message:
    message = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
  print(message, flush=True)


def format_duration(seconds: float) -> str:
  if seconds < 60:
    return f"{seconds:.1f}s"
  return str(datetime.timedelta(seconds=int(round(seconds))))


class _LogHandler(logging.Handler):
  """Routes the package's own progress logging (e.g. LITETime's
  per-training lines from `ecg_quantification.fit_cache`) through `log()`,
  so it is timestamped/flushed like everything else. `print` resolves
  `sys.stdout` at emit time, so this keeps working when stdout is
  swapped (e.g. pytest's capsys)."""

  is_run_multiclass_experiment_handler = True

  def emit(self, record: logging.LogRecord) -> None:
    log(f"      {self.format(record)}")


def _attach_package_logging() -> None:
  package_logger = logging.getLogger('ecg_quantification')
  package_logger.setLevel(logging.INFO)
  # printed here only: TensorFlow/cvxpy may configure the root logger
  # mid-run, which would otherwise print every line a second time
  package_logger.propagate = False
  # marker attribute rather than isinstance: the script may be loaded
  # more than once per process (tests load it by path), and each load
  # defines its own _LogHandler class -- never stack duplicate handlers
  if not any(getattr(h, 'is_run_multiclass_experiment_handler', False)
             for h in package_logger.handlers):
    package_logger.addHandler(_LogHandler())


def check_xla_ptxas(ptxas_path: str | None, xla_enabled: bool) -> tuple[bool, str]:
  """Decides, before any LITETime GPU training, whether XLA can be used:
  `(ok, message)`. Not ok when XLA is enabled and its `ptxas` is older
  than MIN_SAFE_PTXAS_VERSION -- such a ptxas fails to compile some of
  XLA's kernels (and XLA warns it miscompiles others), so the run would
  crash minutes or hours in, or produce suspect numbers."""
  if not xla_enabled:
    return True, "XLA disabled for LITETime (--no-lite-time-xla): regular TensorFlow GPU kernels."
  version = ptxas_version(ptxas_path)
  if version is None:
    return True, (f"WARNING: could not determine the ptxas XLA will use ({ptxas_path or 'none found'}); "
                  f"if training fails with 'Autotuner could not compile any configs', see the "
                  f"README's `deep` extra section or pass --no-lite-time-xla.")
  shown = '.'.join(map(str, version))
  if version < MIN_SAFE_PTXAS_VERSION:
    minimum = '.'.join(map(str, MIN_SAFE_PTXAS_VERSION))
    return False, (
      f"XLA would compile LITETime's GPU kernels with ptxas {shown} ({ptxas_path}), older than "
      f"{minimum}: it fails to compile some of them ('Autotuner could not compile any configs "
      f"for HLO ...') and miscompiles others. Fix it with\n"
      f"    pip install -U \"nvidia-cuda-nvcc-cu12>={minimum},<13\"\n"
      f"(this script then points XLA at it automatically), or train without XLA: "
      f"--no-lite-time-xla."
    )
  return True, f"XLA ptxas: {ptxas_path} ({shown})"


def _quiet_tensorflow() -> None:
  """Keeps the terminal/log readable during LITETime runs: Keras' own
  per-call progress bars (aeon's `predict_proba` calls `model.predict`
  without `verbose=0`, printing a carriage-return bar per out-of-fold
  prediction -- unreadable once piped to a log file) and TensorFlow's
  harmless "tf.function retracing" warnings (every training builds fresh
  models, so retracing is expected). Training progress is reported by
  the fit cache's "training #k" lines instead."""
  import keras
  import tensorflow as tf
  keras.config.disable_interactive_logging()
  tf.get_logger().setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('--splits-path', type=str, required=True,
                       help="Path to the .npz produced by prepare_splits.py (label_mode=multiclass).")
  parser.add_argument('--output-dir', type=str, default='results/multiclass_capp',
                       help="Directory to write the results table(s) to.")
  parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help="Directory for fit/evaluation checkpoints. Defaults to "
                            "'<output-dir>/_checkpoint'.")
  parser.add_argument('--fresh', action='store_true',
                       help="Discard any existing checkpoint under --checkpoint-dir and start over, "
                            "instead of resuming from it.")

  parser.add_argument('--n-bags', type=int, default=200,
                       help="Number of C-APP test bags to generate per bag size.")
  parser.add_argument('--bag-sizes', type=int, nargs='+', default=[100, 200, 500],
                       help="Bag sizes to evaluate, mirroring CBMS'26 Table I.")

  parser.add_argument('--classifiers', type=str, nargs='+', default=['LR', 'SVM', 'GB', 'RF'],
                       choices=['LR', 'SVM', 'GB', 'RF', 'LITETime', 'ECGFounder'],
                       help="Base classifiers to evaluate every quantifier with. The deep "
                            "ones -- 'LITETime' (aeon's deep-learning ensemble, [deep] extra) "
                            "and 'ECGFounder' (pretrained ECG foundation model, [ecgfounder] "
                            "extra) -- are automatically run with n_jobs=None and through a "
                            "fit cache (see the --lite-time-*/--ecg-founder-* flags and item "
                            "13's docs) regardless of --n-jobs, since a per-fold process-"
                            "parallel refit of a deep model is both uncachable across "
                            "processes and resource-wasteful (see "
                            "docs/quantifier_registry_audit.md).")
  parser.add_argument('--include-feature-space', action=argparse.BooleanOptionalAction, default=True,
                       help="Also evaluate classifier-free quantifiers (HDx, ReadMe, ED).")
  parser.add_argument('--hdx-bins', type=int, default=8,
                       help="Quantile bins per feature for HDx, which needs discrete features "
                            "(registered as 'HDx_<n>bins').")
  parser.add_argument('--readme-bins', type=int, default=2,
                       help="Quantile bins per feature for ReadMe, which matches whole "
                            "patterns of several features (registered as 'ReadMe_<n>bins').")

  parser.add_argument('--cv', type=int, default=10,
                       help="Cross-validation folds for calibrated quantifiers.")
  parser.add_argument('--n-jobs', type=int, default=None,
                       help="Parallel jobs forwarded to every quantifier's own n_jobs, for "
                            "every classifier except LITETime (always n_jobs=None -- see "
                            "--classifiers' help).")
  parser.add_argument('--parallel-backend', type=str, default='loky', choices=['loky', 'threading'])

  parser.add_argument('--random-state', type=int, default=0)

  parser.add_argument('--lite-time-n-classifiers', type=int, default=5,
                       help="Ensemble size for LITETimeClassifier (only used if 'LITETime' "
                            "is in --classifiers).")
  parser.add_argument('--lite-time-n-epochs', type=int, default=1500,
                       help="Training epochs per LITETime ensemble member (only used if "
                            "'LITETime' is in --classifiers). Aeon's own default (1500) is "
                            "expensive -- consider lowering for a quick sanity check.")
  parser.add_argument('--lite-time-batch-size', type=int, default=64,
                       help="Batch size for LITETimeClassifier (only used if 'LITETime' is "
                            "in --classifiers).")
  parser.add_argument('--lite-time-device', type=str, default='gpu', choices=['gpu', 'cpu'],
                       help="Device LITETime trains on (only used if 'LITETime' is in "
                            "--classifiers). 'gpu' (default) stops with a diagnosis before "
                            "fitting anything if TensorFlow sees no GPU, instead of silently "
                            "falling back to a much slower CPU run; 'cpu' hides any GPU. Not "
                            "part of the checkpoint's run config, so a run can be resumed on "
                            "the other device.")
  parser.add_argument('--lite-time-xla', action=argparse.BooleanOptionalAction, default=True,
                       help="Compile LITETime's GPU training with XLA (Keras' default, faster). "
                            "Requires a CUDA ptxas >= 12.6.3, checked before training. "
                            "--no-lite-time-xla uses TensorFlow's regular GPU kernels instead "
                            "(no ptxas needed, usually somewhat slower). Not part of the run "
                            "config.")

  parser.add_argument('--ecg-founder-checkpoint', type=str, default=None,
                       help="Local 1_lead_ECGFounder.pth (only used if 'ECGFounder' is in "
                            "--classifiers). Default: $ECGFOUNDER_CHECKPOINT, else downloaded "
                            "once from Hugging Face (PKUDigitalHealth/ECGFounder, 370 MB) "
                            "before anything is fitted.")
  parser.add_argument('--ecg-founder-mode', type=str, default='linear-probe',
                       choices=['linear-probe', 'finetune'],
                       help="'linear-probe' (default): frozen pretrained backbone + logistic "
                            "regression on its features -- cheap. 'finetune': full fine-"
                            "tuning of the whole network per training subset (~66 trainings "
                            "with cv=10 and 5 classes) -- expensive; see the --ecg-founder-* "
                            "flags below.")
  parser.add_argument('--ecg-founder-n-epochs', type=int, default=5,
                       help="Fine-tuning epochs (finetune mode; official notebook: 5).")
  parser.add_argument('--ecg-founder-batch-size', type=int, default=256,
                       help="Training/inference batch size for ECGFounder.")
  parser.add_argument('--ecg-founder-lr', type=float, default=1e-4,
                       help="Fine-tuning learning rate (Adam; official notebook: 1e-4).")
  parser.add_argument('--ecg-founder-validation-fraction', type=float, default=0.1,
                       help="Fine-tuning: stratified share of each training subset held out "
                            "to keep the epoch with the lowest validation loss (0 = last epoch).")
  parser.add_argument('--ecg-founder-input-fs', type=int, default=360,
                       help="Sampling rate of the beat segments in --splits-path (MIT-BIH: "
                            "360 Hz); resampled to ECGFounder's 500 Hz.")
  parser.add_argument('--ecg-founder-filter', action=argparse.BooleanOptionalAction, default=True,
                       help="Apply ECGFounder's pretraining filter chain (50 Hz notch, "
                            "0.67-40 Hz band-pass, median baseline removal) to every segment.")
  parser.add_argument('--ecg-founder-device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help="Device for ECGFounder. 'cuda' (default) stops before fitting "
                            "anything if PyTorch sees no GPU. Not part of the run config.")
  parser.add_argument('--ecg-founder-trust-checkpoint', action='store_true',
                       help="Allow loading the pretrained file with full unpickling if it "
                            "holds more than tensors/plain data (only for a trusted file).")

  parser.add_argument('--include-threshold-family', action=argparse.BooleanOptionalAction, default=False,
                      help="Also evaluate native-multiclass X/Max/T50/MedianSweep alongside their "
                          "OvR-wrapped counterparts (ablation: OvR decomposition vs. native generalization).")

  return parser.parse_args()


def fit_cache_id(base: str, classifier_name: str) -> str:
  """Fit-cache scope of one deep classifier within one run."""
  return f"{base}::{classifier_name}"


def build_registry(classifier_names: list[str],
                   include_feature_space: bool,
                   cv: int,
                   n_jobs: int,
                   parallel_backend: str,
                   include_threshold_family: bool = False,
                   lite_time_kwargs: dict = None,
                   cache_id: str = None,
                   ecg_founder_kwargs: dict = None,
                   feature_space_kwargs: dict = None) -> dict[str, object]:
  """Assembles the full quantifier registry to evaluate: every
  classifier x quantifier combination, plus classifier-free feature-space
  quantifiers if requested.

  Parameters
  ----------
  lite_time_kwargs : dict, default = None
    Forwarded to `LITETimeClassifier(...)` when `'LITETime'` is in
    `classifier_names` (e.g. `n_classifiers`, `n_epochs`, `batch_size`,
    `random_state`). Ignored otherwise.
  cache_id : str, default = None
    Base scope for the deep classifiers' fit caches (item 13) -- give
    every independent run a distinct id (e.g. derived from
    `--checkpoint-dir`) so unrelated runs sharing this process never
    share cached fits. Each deep classifier gets its own scope,
    `fit_cache_id(cache_id, name)`: the cache keys on the training data
    alone, so LITETime and ECGFounder fitting the very same subset must
    never be served each other's model. Required when a deep classifier
    is requested.
  ecg_founder_kwargs : dict, default = None
    Forwarded to `ECGFounderClassifier(...)` when `'ECGFounder'` is in
    `classifier_names`. Ignored otherwise.
  feature_space_kwargs : dict, default = None
    Forwarded to `build_feature_space_quantifiers(...)` (`hdx_bins`,
    `readme_bins`, `random_state`) when `include_feature_space`.

  Notes
  -----
  The deep classifiers (LITETime, ECGFounder) are always built with
  `n_jobs=None` regardless of the `n_jobs` argument, and through
  `cached_classifier_factory` so the ~16
  quantifier recipes built on it share one fit cache instead of each
  independently refitting the (expensive) deep model per CV fold/OvR
  member -- see `docs/quantifier_registry_audit.md`'s item 13 note and
  `ecg_quantification.quantifiers._cached_classifier`'s module
  docstring. Every other classifier is built exactly as before, with the
  caller's own `n_jobs`.
  """
  classifiers = default_base_classifiers()
  registry = {}

  deep_factories = {
    'LITETime': lambda: LITETimeClassifier(**(lite_time_kwargs or {})),
    'ECGFounder': lambda: ECGFounderClassifier(**(ecg_founder_kwargs or {})),
  }
  for clf_name in classifier_names:
    if clf_name in DEEP_CLASSIFIERS:
      classifier_factory = cached_classifier_factory(
        deep_factories[clf_name], cache_id=fit_cache_id(cache_id, clf_name),
      )
      registry.update(build_all_quantifiers(
        clf_name, classifier_factory, cv=cv, n_jobs=None, parallel_backend=parallel_backend,
        include_threshold_family=include_threshold_family,
      ))
    else:
      registry.update(build_all_quantifiers(
        clf_name, classifiers[clf_name], cv=cv, n_jobs=n_jobs, parallel_backend=parallel_backend,
        include_threshold_family=include_threshold_family,
      ))

  if include_feature_space:
    registry.update(build_feature_space_quantifiers(
      n_jobs=n_jobs, parallel_backend=parallel_backend, **(feature_space_kwargs or {}),
    ))

  return registry


def evaluate_quantifier(name: str,
                        quantifier,
                        bags: list[tuple[np.ndarray, np.ndarray]],
                        true_prevalences: np.ndarray,
                        classes: np.ndarray,
                        zce_metric: ZoneCrossingError) -> list[dict]:
  """Scores every test bag against a single fitted quantifier, returning
  one result row (dict) per bag with both MAE and Zone-Crossing Error."""
  rows = []
  for i, (X_bag, _) in enumerate(bags):
    predicted = quantifier.predict(X_bag)

    # align predicted vector to the full class set: a quantifier fit on
    # the training data already saw every class (patient-wise split
    # preserves all 5 labels in train), so this is a direct comparison
    mae = ae(true_prevalences[i], predicted)
    zce = zce_metric(true_prevalences[i], predicted)

    rows.append({
      'quantifier': name,
      'bag_index': i,
      'mae': mae,
      'zce': zce,
      **{f'true_{c}': true_prevalences[i][j] for j, c in enumerate(classes)},
      **{f'pred_{c}': predicted[j] for j, c in enumerate(classes)},
    })
  return rows


def main() -> None:
  args = parse_args()
  run_start = time.time()
  _attach_package_logging()

  output_dir = args.output_dir
  checkpoint_dir = args.checkpoint_dir or os.path.join(output_dir, '_checkpoint')
  use_lite_time = 'LITETime' in args.classifiers
  use_ecg_founder = 'ECGFounder' in args.classifiers
  deep_requested = [name for name in DEEP_CLASSIFIERS if name in args.classifiers]

  if args.fresh and os.path.exists(checkpoint_dir):
    log(f"=== --fresh given: removing existing checkpoint at {checkpoint_dir} ===")
    shutil.rmtree(checkpoint_dir)

  os.makedirs(output_dir, exist_ok=True)
  checkpoint = ExperimentCheckpoint(checkpoint_dir)

  # only settings that change *what* gets fitted/evaluated go into the
  # run config -- n_jobs/parallel_backend/device only affect speed, not
  # what is being fitted, and forcing an exact match on those would make
  # resuming with a different setting needlessly fail
  run_config = {
    'splits_path': args.splits_path,
    'classifiers': sorted(args.classifiers),
    'include_feature_space': args.include_feature_space,
    'cv': args.cv,
    'random_state': args.random_state,
    'n_bags': args.n_bags,
    'bag_sizes': sorted(args.bag_sizes),
  }
  if use_lite_time:
    # part of the run config (not just speed/parallelism) -- unlike
    # n_jobs, these change what actually gets fitted
    run_config.update({
      'lite_time_n_classifiers': args.lite_time_n_classifiers,
      'lite_time_n_epochs': args.lite_time_n_epochs,
      'lite_time_batch_size': args.lite_time_batch_size,
    })
  if use_ecg_founder:
    run_config.update({
      # identifies the weights without tying the run to one machine's path
      'ecg_founder_checkpoint': (os.path.basename(args.ecg_founder_checkpoint)
                                 if args.ecg_founder_checkpoint else 'huggingface'),
      'ecg_founder_mode': args.ecg_founder_mode,
      'ecg_founder_input_fs': args.ecg_founder_input_fs,
      'ecg_founder_filter': args.ecg_founder_filter,
    })
    if args.ecg_founder_mode == 'finetune':
      run_config.update({
        'ecg_founder_n_epochs': args.ecg_founder_n_epochs,
        'ecg_founder_batch_size': args.ecg_founder_batch_size,
        'ecg_founder_lr': args.ecg_founder_lr,
        'ecg_founder_validation_fraction': args.ecg_founder_validation_fraction,
      })
  checkpoint.record_run_config(run_config)

  if use_lite_time:
    # before loading data or building anything: a missing GPU should stop
    # the run within seconds, not after hours of CPU training
    try:
      devices = configure_tensorflow_device(args.lite_time_device)
    except RuntimeError as error:
      log(f"ERROR: {error}")
      raise SystemExit(1)
    log(f"LITETime device: {args.lite_time_device.upper()} ({', '.join(devices)})")
    if args.lite_time_device == 'gpu':
      ok, message = check_xla_ptxas(XLA_PTXAS, args.lite_time_xla)
      if not ok:
        log(f"ERROR: {message}")
        raise SystemExit(1)
      log(message)
      if args.lite_time_xla:
        log(f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}")
      else:
        disable_keras_xla()
    _quiet_tensorflow()

  ecg_founder_checkpoint = None
  if use_ecg_founder:
    # same fail-fast idea: GPU and pretrained weights (downloading them
    # if needed) are checked before any data is loaded or anything fitted
    try:
      devices = configure_torch_device(args.ecg_founder_device)
      ecg_founder_checkpoint = resolve_ecg_founder_checkpoint(args.ecg_founder_checkpoint)
    except (RuntimeError, ImportError, FileNotFoundError) as error:
      log(f"ERROR: {error}")
      raise SystemExit(1)
    log(f"ECGFounder device: {args.ecg_founder_device.upper()} ({', '.join(devices)})")
    log(f"ECGFounder weights: {ecg_founder_checkpoint} ({args.ecg_founder_mode})")

  log(f"=== [1/5] Loading splits from {args.splits_path} ===")
  data = np.load(args.splits_path, allow_pickle=True)
  X_train, y_train = data['X_train'], data['y_train']
  X_test, y_test, ids_test = data['X_test'], data['y_test'], data['ids_test']
  log(f"Train: {X_train.shape[0]} segments. Test pool: {X_test.shape[0]} segments "
      f"from {len(np.unique(ids_test))} patients.")

  log()
  log("=== [2/5] Building quantifier registry ===")
  if deep_requested and args.n_jobs not in (None, 1):
    log(f"  NOTE: --n-jobs={args.n_jobs} is ignored for {'/'.join(deep_requested)} "
        f"(always n_jobs=None) -- see --classifiers' help.")
  # one fit-cache scope per checkpoint_dir: reruns resuming the same
  # checkpoint reasonably reuse the same scope name (harmless either
  # way, since the cache is in-memory/per-process only -- cross-run
  # reuse is what ExperimentCheckpoint's fitted-quantifier persistence
  # already handles).
  cache_id = checkpoint_dir
  registry = build_registry(
    args.classifiers, args.include_feature_space,
    cv=args.cv, n_jobs=args.n_jobs, parallel_backend=args.parallel_backend,
    include_threshold_family=args.include_threshold_family,
    lite_time_kwargs=dict(
      n_classifiers=args.lite_time_n_classifiers,
      n_epochs=args.lite_time_n_epochs,
      batch_size=args.lite_time_batch_size,
      random_state=args.random_state,
    ),
    cache_id=cache_id,
    # the bin counts are part of HDx/ReadMe's registry names, so they need
    # no run-config entry: a different value is simply a new quantifier
    feature_space_kwargs=dict(
      hdx_bins=args.hdx_bins, readme_bins=args.readme_bins, random_state=args.random_state,
    ),
    ecg_founder_kwargs=dict(
      checkpoint_path=ecg_founder_checkpoint,
      freeze_backbone=args.ecg_founder_mode == 'linear-probe',
      input_sampling_rate=args.ecg_founder_input_fs,
      apply_ecgfounder_filter=args.ecg_founder_filter,
      n_epochs=args.ecg_founder_n_epochs,
      batch_size=args.ecg_founder_batch_size,
      learning_rate=args.ecg_founder_lr,
      validation_fraction=args.ecg_founder_validation_fraction,
      device=args.ecg_founder_device,
      # fine-tuning: per-epoch resumable state + finished weights live
      # next to the experiment checkpoint, so a restart resumes trainings
      # instead of repeating them (see _ecg_founder's module docstring)
      training_checkpoint_root=(os.path.join(checkpoint_dir, 'ecg_founder_trainings')
                                if args.ecg_founder_mode == 'finetune' else None),
      trust_checkpoint=args.ecg_founder_trust_checkpoint,
      random_state=args.random_state,
    ),
  )
  log(f"{len(registry)} quantifiers registered: {sorted(registry.keys())}")
  if checkpoint.fitted_names:
    log(f"Checkpoint found: {len(checkpoint.fitted_names)}/{len(registry)} already fitted, will be skipped.")

  # Every quantifier in the registry is fit directly on the original
  # string labels ('N', 'V', 'A', 'L', 'R') -- no manual encoding needed
  # here. Every native multiclass quantifier (CC, PCC, GAC, GPAC, FM,
  # EMQ) already comes out of build_registry() -> build_all_quantifiers()
  # -> build_multiclass_quantifiers() wrapped in LabelEncodedQuantifier,
  # which encodes y to integers before fitting the base classifier and
  # exposes .classes_ as the original string labels either way -- GAC is
  # the only one of the six that would otherwise actually break on string
  # labels (quack's BaseCalibratedQuantifier.fit() allocates its
  # hard-label out-of-fold buffer as a plain float64 array regardless of
  # y's dtype), the other five are wrapped too for consistency and so
  # they share one fit-cache group when combined with
  # ecg_quantification.quantifiers.CachedFitClassifier (item 13). See
  # ecg_quantification.quantifiers._label_encoded_quantifier's module
  # docstring for the full explanation.
  all_classes = np.unique(np.concatenate([y_train, y_test]))

  log()
  log("=== [3/5] Fitting all quantifiers on training data ===")
  expected = (args.cv + 1) * (1 + len(np.unique(y_train)))
  for deep_name in deep_requested:
    log(f"{deep_name}: a from-scratch run needs ~{expected} real trainings in total "
        f"((cv + 1) x (1 + n_classes)); every other {deep_name} fit reuses one of them.")
  if use_lite_time or (use_ecg_founder and args.ecg_founder_mode == 'linear-probe'):
    log("When resuming, quantifiers already in the checkpoint are loaded, but the "
        "in-memory fit cache starts empty: the first not-yet-fitted native and OvR "
        "quantifier retrain their group's trainings once"
        + (" (ECGFounder fine-tuning resumes from its own training checkpoints instead)."
           if use_ecg_founder and args.ecg_founder_mode == 'finetune' else "."))
  fitted = {}
  n_total = len(registry)
  for i, (name, quantifier) in enumerate(registry.items(), start=1):
    tag = f"[{i}/{n_total}] {name}"
    if checkpoint.is_fitted(name):
      fitted[name] = checkpoint.load_fitted(name)
      log(f"  {tag}: loaded from checkpoint")
      continue

    deep_name = next((d for d in deep_requested if name.startswith(f'{d}-')), None)
    deep_cache_id = fit_cache_id(cache_id, deep_name) if deep_name else None
    before = fit_cache_stats(deep_cache_id) if deep_name else None
    log(f"  {tag}: fitting...")
    start = time.time()
    quantifier.fit(X_train, y_train)
    elapsed = time.time() - start
    checkpoint.save_fitted(name, quantifier)
    fitted[name] = quantifier

    detail = ""
    if deep_name:
      after = fit_cache_stats(deep_cache_id)
      detail = (f" ({after['trainings'] - before['trainings']} new {deep_name} trainings, "
                f"{after['hits'] - before['hits']} reused from cache; "
                f"{after['trainings']} trained so far)")
    log(f"  {tag}: fit in {format_duration(elapsed)}{detail}")

  classes = all_classes
  zce_metric = ZoneCrossingError(clinical_ranges=CLINICAL_RANGES, classes=classes)

  log()
  log("=== [4/5] Evaluating on C-APP bags ===")
  for b, bag_size in enumerate(args.bag_sizes, start=1):
    log(f"--- bag size {b}/{len(args.bag_sizes)}: bag_size={bag_size}, n_bags={args.n_bags} ---")
    # regenerated every run (cheap) rather than checkpointed themselves:
    # with a fixed random_state, ClinicalPrevalenceBagGenerator produces
    # byte-identical bags every time, so this stays consistent with
    # whatever was already evaluated and checkpointed for this bag_size
    generator = capp_bag_generator(args.n_bags, bag_size, args.random_state)
    bags = generator.to_list(X_test, y_test)
    true_prevalences = generator.sampled_prevalences_

    for i, (name, quantifier) in enumerate(fitted.items(), start=1):
      tag = f"[{i}/{len(fitted)}] {name}"
      if checkpoint.is_evaluated(name, bag_size):
        log(f"  {tag}: already evaluated for bag_size={bag_size}, skipping")
        continue

      start = time.time()
      rows = evaluate_quantifier(name, quantifier, bags, true_prevalences, classes, zce_metric)
      for row in rows:
        row['bag_size'] = bag_size
      checkpoint.save_evaluation(name, bag_size, rows)

      maes = [r['mae'] for r in rows]
      zces = [r['zce'] for r in rows]
      log(f"  {tag}: MAE={np.mean(maes):.4f} (±{np.std(maes):.4f})  "
          f"ZCE={np.mean(zces):.4f} (±{np.std(zces):.4f})  "
          f"[{format_duration(time.time() - start)}]")

  log()
  log(f"=== [5/5] Saving results to {output_dir} ===")
  results_df = checkpoint.load_results()
  detailed_path = os.path.join(output_dir, 'detailed_results.csv')
  results_df.to_csv(detailed_path, index=False)
  log(f"Saved per-bag detail: {detailed_path}")

  summary = (
    results_df.groupby(['quantifier', 'bag_size'])
    .agg(mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
         zce_mean=('zce', 'mean'), zce_std=('zce', 'std'))
    .reset_index()
  )
  summary_path = os.path.join(output_dir, 'summary_table.csv')
  summary.to_csv(summary_path, index=False)
  log(f"Saved summary table: {summary_path}")

  log()
  log("=== Best quantifier per bag size (by MAE) ===")
  for bag_size in args.bag_sizes:
    subset = summary[summary['bag_size'] == bag_size].sort_values('mae_mean')
    best = subset.iloc[0]
    log(f"  bag_size={bag_size}: {best['quantifier']} "
        f"(MAE={best['mae_mean']:.4f}, ZCE={best['zce_mean']:.4f})")

  log()
  log("=== Best quantifier per bag size (by ZCE) ===")
  for bag_size in args.bag_sizes:
    subset = summary[summary['bag_size'] == bag_size].sort_values('zce_mean')
    best = subset.iloc[0]
    log(f"  bag_size={bag_size}: {best['quantifier']} "
        f"(ZCE={best['zce_mean']:.4f}, MAE={best['mae_mean']:.4f})")

  log()
  log(f"Done in {format_duration(time.time() - run_start)}.")


if __name__ == '__main__':
  main()