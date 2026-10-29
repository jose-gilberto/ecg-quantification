"""Tests for `ecg_quantification.quantifiers.ECGFounderClassifier` (Fase 4).

The real pretrained weights (370 MB, Hugging Face) are not needed: a
*fake* checkpoint with the exact structure of the official release
(`{'state_dict': <Net1D with a 150-class head>, 'optimizer': ...,
'val_auroc': <numpy float>, ...}`) and random weights exercises the same
loading, validation, preprocessing, training and pickling code paths.

Requires PyTorch (the `ecgfounder` extra); skipped otherwise.
"""
import argparse
import logging
import os

import joblib
import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from ecg_quantification.quantifiers import (  # noqa: E402
  ECGFounderClassifier, configure_torch_device, resolve_ecg_founder_checkpoint,
)
from ecg_quantification.quantifiers import _ecg_founder  # noqa: E402
from ecg_quantification.quantifiers._ecg_founder import (  # noqa: E402
  _NET1D_ARCHITECTURE, _load_pretrained_backbone_state, prepare_ecg_founder_input,
)
from ecg_quantification.quantifiers._net1d import Net1D  # noqa: E402


@pytest.fixture(scope="module")
def fake_checkpoint(tmp_path_factory):
  """A checkpoint shaped like the official `1_lead_ECGFounder.pth`."""
  _ecg_founder._import_torch()  # TensorFlow may be loaded by earlier tests
  torch.manual_seed(0)
  network = Net1D(n_classes=150, **_NET1D_ARCHITECTURE)
  optimizer = torch.optim.Adam(network.parameters())
  path = str(tmp_path_factory.mktemp("ecgfounder") / "1_lead_ECGFounder.pth")
  torch.save({
    "epoch": 3, "step": 1000,
    "state_dict": network.state_dict(),
    "optimizer": optimizer.state_dict(),
    "val_auroc": np.float64(0.91),  # numpy metadata, like the release's
  }, path)
  return path


@pytest.fixture(autouse=True)
def _fresh_memos():
  """Pretrained states/backbones are memoized per process by path; tests
  that write different files to reused paths need a clean slate."""
  _ecg_founder._PRETRAINED_STATES.clear()
  _ecg_founder._FROZEN_BACKBONES.clear()
  yield


def _make_data(n_per_class=20, length=360, seed=0, labels=("N", "V", "A")):
  rng = np.random.default_rng(seed)
  X, y = [], []
  for i, label in enumerate(labels):
    segment = rng.normal(size=(n_per_class, length))
    segment[:, 60 + 60 * i: 90 + 60 * i] += 5.0  # class-specific bump
    X.append(segment)
    y.append(np.full(n_per_class, label))
  X, y = np.vstack(X), np.concatenate(y)
  order = rng.permutation(len(y))
  return X[order], y[order]


class TestPreprocessing:

  def test_resamples_360hz_beats_to_500hz_and_zscores(self):
    X, _ = _make_data(length=360)
    out = prepare_ecg_founder_input(X, input_sampling_rate=360)
    assert out.shape == (len(X), 1, 500)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out.mean(axis=-1), 0, atol=1e-4)
    np.testing.assert_allclose(out.std(axis=-1), 1, atol=1e-3)

  def test_accepts_channel_axis_and_skips_resampling_at_500hz(self):
    X = np.random.default_rng(1).normal(size=(4, 1, 500))
    assert prepare_ecg_founder_input(X, input_sampling_rate=500).shape == (4, 1, 500)

  def test_filter_matches_the_official_per_row_implementation(self):
    """Vectorized filter == ECGFounder's util.filter_bandpass applied row
    by row (scipy.signal.medfilt zero-pads, like our median_filter)."""
    from scipy.signal import butter, filtfilt, iirnotch, medfilt
    X = np.random.default_rng(2).normal(size=(3, 500))
    ours = _ecg_founder._filter_like_ecgfounder(X, 500)
    for row, expected_row in zip(X, ours):
      b, a = iirnotch(50, 30, 500)
      ref = filtfilt(b, a, row)
      b, a = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=500)
      ref = filtfilt(b, a, ref)
      ref = ref - medfilt(ref, kernel_size=201)
      np.testing.assert_allclose(expected_row, ref, atol=1e-10)

  def test_too_short_for_filter_raises_but_works_without_it(self):
    X = np.random.default_rng(3).normal(size=(2, 16))
    with pytest.raises(ValueError, match="too short"):
      prepare_ecg_founder_input(X, input_sampling_rate=500)
    assert prepare_ecg_founder_input(X, input_sampling_rate=500, apply_ecgfounder_filter=False).shape == (2, 1, 16)

  def test_rejects_multilead_input(self):
    with pytest.raises(ValueError, match="single-lead"):
      prepare_ecg_founder_input(np.zeros((2, 12, 500)), input_sampling_rate=500)


class TestCheckpointLoading:

  def test_loads_backbone_without_the_150_class_head(self, fake_checkpoint):
    state = _load_pretrained_backbone_state(fake_checkpoint)
    assert not any(key.startswith("dense.") for key in state)
    assert "first_conv.conv.weight" in state

  def test_strips_dataparallel_prefix(self, tmp_path):
    network = Net1D(n_classes=150, **_NET1D_ARCHITECTURE)
    path = str(tmp_path / "dp.pth")
    torch.save({"state_dict": {f"module.{k}": v for k, v in network.state_dict().items()}}, path)
    assert "first_conv.conv.weight" in _load_pretrained_backbone_state(path)

  def test_wrong_architecture_is_an_error_not_a_silent_random_init(self, tmp_path):
    """The official loader uses strict=False and would silently keep
    random weights for a mismatched file; this one must refuse."""
    small = Net1D(n_classes=150, **{**_NET1D_ARCHITECTURE, "filter_list": [16, 32, 32, 80, 80, 256, 256]})
    path = str(tmp_path / "wrong.pth")
    torch.save({"state_dict": small.state_dict()}, path)
    with pytest.raises(ValueError, match="does not match"):
      _load_pretrained_backbone_state(path)

  def test_twelve_lead_checkpoint_is_rejected(self, tmp_path):
    network = Net1D(n_classes=150, **{**_NET1D_ARCHITECTURE, "in_channels": 12})
    path = str(tmp_path / "12_lead.pth")
    torch.save({"state_dict": network.state_dict()}, path)
    with pytest.raises(ValueError, match="12-lead"):
      _load_pretrained_backbone_state(path)

  def test_numpy1_pickled_metadata_loads_without_trust(self, tmp_path):
    """Regression: the official release was pickled under numpy 1.x, so its
    numpy metadata is stored as `numpy.core.multiarray.scalar`, while under
    numpy 2.x that function reports `numpy._core.multiarray.scalar` --
    allowlisting it under the new name alone made the real file fail
    `weights_only=True`. Rebuilds such a file by renaming the globals."""
    import zipfile
    network = Net1D(n_classes=150, **_NET1D_ARCHITECTURE)
    modern = str(tmp_path / "modern.pth")
    torch.save({"state_dict": network.state_dict(), "val_auroc": np.float64(0.91),
                "thresholds": np.array([0.5, 0.4])}, modern)
    legacy = str(tmp_path / "legacy.pth")
    with zipfile.ZipFile(modern) as src, zipfile.ZipFile(legacy, "w") as dst:
      for info in src.infolist():
        data = src.read(info.filename)
        if info.filename.endswith("data.pkl"):
          data = data.replace(b"numpy._core.multiarray", b"numpy.core.multiarray")
        dst.writestr(info, data)
    with zipfile.ZipFile(legacy) as archive:
      pickled = archive.read(next(n for n in archive.namelist() if n.endswith("data.pkl")))
    assert b"numpy.core.multiarray" in pickled and b"numpy._core" not in pickled
    assert "first_conv.conv.weight" in _load_pretrained_backbone_state(legacy)

  def test_arbitrary_pickled_objects_need_trust_checkpoint(self, tmp_path):
    network = Net1D(n_classes=150, **_NET1D_ARCHITECTURE)
    path = str(tmp_path / "pickled.pth")
    torch.save({"state_dict": network.state_dict(), "args": argparse.Namespace(lr=1e-4)}, path)
    with pytest.raises(RuntimeError, match="trust_checkpoint"):
      _load_pretrained_backbone_state(path)
    assert "first_conv.conv.weight" in _load_pretrained_backbone_state(path, trust_checkpoint=True)

  def test_resolve_checkpoint(self, fake_checkpoint, tmp_path, monkeypatch):
    resolve_ecg_founder_checkpoint.cache_clear()
    assert resolve_ecg_founder_checkpoint(fake_checkpoint) == os.path.abspath(fake_checkpoint)
    with pytest.raises(FileNotFoundError):
      resolve_ecg_founder_checkpoint(str(tmp_path / "missing.pth"))
    monkeypatch.setenv("ECGFOUNDER_CHECKPOINT", fake_checkpoint)
    resolve_ecg_founder_checkpoint.cache_clear()
    assert resolve_ecg_founder_checkpoint(None) == os.path.abspath(fake_checkpoint)
    resolve_ecg_founder_checkpoint.cache_clear()


class TestDevice:

  def test_rejects_unknown_device(self):
    with pytest.raises(ValueError, match="'cuda' or 'cpu'"):
      configure_torch_device("tpu")

  def test_cpu(self):
    assert configure_torch_device("cpu") == ["CPU"]

  def test_cuda_without_gpu_fails_fast(self, fake_checkpoint):
    if torch.cuda.is_available():
      pytest.skip("a CUDA GPU is visible here")
    with pytest.raises(RuntimeError, match="no GPU is visible to PyTorch"):
      configure_torch_device("cuda")
    X, y = _make_data()
    with pytest.raises(RuntimeError, match="no GPU is visible to PyTorch"):
      ECGFounderClassifier(checkpoint_path=fake_checkpoint).fit(X, y)  # default device='cuda'


class TestLinearProbe:

  def _classifier(self, checkpoint, **kwargs):
    return ECGFounderClassifier(checkpoint_path=checkpoint, device="cpu", batch_size=32, **kwargs)

  def test_fit_predict_predict_proba_with_string_labels(self, fake_checkpoint):
    X, y = _make_data()
    classifier = self._classifier(fake_checkpoint).fit(X, y)
    assert list(classifier.classes_) == ["A", "N", "V"]
    proba = classifier.predict_proba(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-6)
    predictions = classifier.predict(X)
    assert set(predictions) <= {"A", "N", "V"}
    # the frozen random-weight backbone still separates these bumps
    assert (predictions == y).mean() > 0.9

  def test_is_sklearn_cloneable(self, fake_checkpoint):
    X, y = _make_data()
    fitted = self._classifier(fake_checkpoint).fit(X, y)
    fresh = clone(fitted)
    assert not hasattr(fresh, "model_")
    assert hasattr(fresh, "predict_proba")

  def test_pickles_without_the_backbone(self, fake_checkpoint, tmp_path):
    X, y = _make_data()
    classifier = self._classifier(fake_checkpoint).fit(X, y)
    path = str(tmp_path / "probe.joblib")
    joblib.dump(classifier, path)
    assert os.path.getsize(path) < 1_000_000  # head only, not 123 MB of backbone
    _ecg_founder._FROZEN_BACKBONES.clear()  # as in a fresh process
    reloaded = joblib.load(path)
    np.testing.assert_allclose(reloaded.predict_proba(X), classifier.predict_proba(X), atol=1e-5)

  def test_logs_progress(self, fake_checkpoint, caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("ecg_quantification"), "propagate", True)
    X, y = _make_data()
    with caplog.at_level(logging.INFO, logger="ecg_quantification.ecg_founder"):
      self._classifier(fake_checkpoint).fit(X, y)
    assert any("linear probe: 60 beats embedded" in r.getMessage() for r in caplog.records)

  def test_quack_quantifiers_end_to_end(self, fake_checkpoint):
    from quack.quantifiers import CC, PACC
    from ecg_quantification.quantifiers import OneVsRestQuantifier
    X, y = _make_data()
    for quantifier in (CC(classifier=self._classifier(fake_checkpoint)),
                       OneVsRestQuantifier(PACC(classifier=self._classifier(fake_checkpoint), cv=2))):
      quantifier.fit(X, y)
      prevalences = quantifier.predict(X[:30])
      assert prevalences.shape == (3,)
      np.testing.assert_allclose(prevalences.sum(), 1, atol=1e-6)


class TestFinetune:

  def _classifier(self, checkpoint, **kwargs):
    defaults = dict(checkpoint_path=checkpoint, freeze_backbone=False, device="cpu",
                    n_epochs=2, batch_size=16, random_state=0)
    return ECGFounderClassifier(**{**defaults, **kwargs})

  def test_fit_predict_and_pickle_without_training_dir(self, fake_checkpoint, tmp_path):
    X, y = _make_data()
    classifier = self._classifier(fake_checkpoint).fit(X, y)
    proba = classifier.predict_proba(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-6)
    # no training dir: the fine-tuned weights travel inside the pickle
    path = str(tmp_path / "ft.joblib")
    joblib.dump(classifier, path)
    assert os.path.getsize(path) > 100_000_000
    np.testing.assert_allclose(joblib.load(path).predict_proba(X), proba, atol=1e-5)

  def test_training_dir_keeps_final_weights_and_lean_pickles(self, fake_checkpoint, tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("ecg_quantification"), "propagate", True)
    X, y = _make_data()
    root = str(tmp_path / "trainings")
    classifier = self._classifier(fake_checkpoint, training_checkpoint_root=root).fit(X, y)
    (directory,) = os.listdir(root)
    assert os.listdir(os.path.join(root, directory)) == ["final.pt"]  # resume.pt cleaned up

    path = str(tmp_path / "ft.joblib")
    joblib.dump(classifier, path)
    assert os.path.getsize(path) < 1_000_000  # a reference to final.pt, not the weights
    np.testing.assert_allclose(joblib.load(path).predict_proba(X), classifier.predict_proba(X), atol=1e-5)

    # a restarted run (fresh model, same data/config) loads instead of training
    # (clear first: if a script attached the package's INFO logging earlier
    # in this session, the first fit's records were captured too)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="ecg_quantification.ecg_founder"):
      again = self._classifier(fake_checkpoint, training_checkpoint_root=root).fit(X, y)
    messages = [r.getMessage() for r in caplog.records]
    assert any("loaded finished training" in m for m in messages)
    assert not any("fine-tuning epoch" in m for m in messages)
    np.testing.assert_allclose(again.predict_proba(X), classifier.predict_proba(X), atol=1e-5)

  def test_interrupted_training_resumes_from_last_epoch(self, fake_checkpoint, tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("ecg_quantification"), "propagate", True)
    X, y = _make_data()
    root = str(tmp_path / "trainings")

    # 60 samples, 10% validation -> 54 training samples -> 4 steps/epoch
    # with batch_size=16: crash on the first step of epoch 2
    original_step = torch.optim.Adam.step
    calls = {"n": 0}

    def crashing_step(self, *args, **kwargs):
      calls["n"] += 1
      if calls["n"] == 5:
        raise KeyboardInterrupt("simulated interruption")
      return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", crashing_step)
    with pytest.raises(KeyboardInterrupt):
      self._classifier(fake_checkpoint, training_checkpoint_root=root).fit(X, y)
    (directory,) = os.listdir(root)
    assert os.listdir(os.path.join(root, directory)) == ["resume.pt"]
    monkeypatch.setattr(torch.optim.Adam, "step", original_step)

    caplog.clear()  # only the resumed run's records (see the test above)
    with caplog.at_level(logging.INFO, logger="ecg_quantification.ecg_founder"):
      self._classifier(fake_checkpoint, training_checkpoint_root=root).fit(X, y)
    messages = [r.getMessage() for r in caplog.records]
    assert any("resuming after epoch 1/2" in m for m in messages)
    assert not any("fine-tuning epoch 1/2" in m for m in messages)
    assert any("fine-tuning epoch 2/2" in m for m in messages)
    assert os.listdir(os.path.join(root, directory)) == ["final.pt"]

  def test_different_hyperparameters_never_share_a_training_dir(self, fake_checkpoint, tmp_path):
    X, y = _make_data()
    root = str(tmp_path / "trainings")
    self._classifier(fake_checkpoint, training_checkpoint_root=root, n_epochs=1).fit(X, y)
    self._classifier(fake_checkpoint, training_checkpoint_root=root, n_epochs=1, learning_rate=1e-3).fit(X, y)
    assert len(os.listdir(root)) == 2

  def test_shares_a_process_with_tensorflow(self, fake_checkpoint):
    """Regression: TensorFlow loaded first, then PyTorch's optimizer
    (which lazily imports Triton) used to segfault the interpreter --
    the `--classifiers LITETime ECGFounder` case. Run in a subprocess so
    a crash fails this test instead of killing the whole session."""
    pytest.importorskip("tensorflow")
    import subprocess
    import sys
    code = (
      "import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\n"
      "import tensorflow as tf, numpy as np\n"
      "tf.constant(1.0)\n"
      "from ecg_quantification.quantifiers import ECGFounderClassifier\n"
      "rng = np.random.default_rng(0)\n"
      "X = rng.normal(size=(40, 360)); y = np.array(['N', 'V'] * 20)\n"
      f"ECGFounderClassifier(checkpoint_path={fake_checkpoint!r}, freeze_backbone=False,\n"
      "  device='cpu', n_epochs=1, batch_size=16, validation_fraction=0).fit(X, y)\n"
      "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")

  def test_rejects_fit_params(self, fake_checkpoint):
    X, y = _make_data()
    classifier = self._classifier(fake_checkpoint)
    classifier.fit_params = {"epochs": 3}
    with pytest.raises(TypeError, match="fit_params"):
      classifier.fit(X, y)
