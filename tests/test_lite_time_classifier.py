"""Tests for `ecg_quantification.quantifiers.LITETimeClassifier`.

Three layers, mirroring `test_one_vs_rest.py`'s approach:

- Unit tests on `_load_lite_time`/`_lite_time_fit`/`_lite_time_predict*`
  in isolation (fast, no real training beyond a couple of epochs).
- A `sklearn.base.clone()` compatibility check, since every `quack`
  quantifier that fits an internal classifier depends on it.
- A real integration test through `quack.quantifiers.ACC`
  (a `BaseCalibratedQuantifier`, exercising the CV/`joblib.Parallel` path)
  with `n_jobs > 1`, specifically to catch checkpoint-file collisions
  across concurrently-fitted folds when `persist=True` -- the scenario
  this module's docstring identifies as the reason `SklearnClassifierWrapper`
  is used at all here, rather than passing `LITETimeClassifier` directly.

Requires `aeon` + `tensorflow`; the whole module is skipped when either is
unavailable (both are optional `deep` extras, not core dependencies).
"""
import os
import numpy as np
import pytest
from sklearn.base import clone

pytest.importorskip("tensorflow")
pytest.importorskip("aeon")

from quack.quantifiers import CC, PACC
from ecg_quantification.quantifiers import LITETimeClassifier
from ecg_quantification.quantifiers._lite_time import _load_lite_time


def _make_data(n=40, length=48, n_classes=2, seed=0):
  rng = np.random.default_rng(seed)
  X = rng.normal(size=(n, length)).astype(np.float64)
  symbols = np.array(['N', 'V', 'A'])[:n_classes]
  y = symbols[rng.integers(0, n_classes, n)]
  return X, y


class TestLoadLiteTime:

  def test_persist_false_by_default_no_checkpoint_root_needed(self):
    model = _load_lite_time(n_classifiers=1, n_epochs=1, batch_size=8)
    assert model.get_params()['n_classifiers'] == 1

  def test_persist_true_requires_checkpoint_root(self):
    with pytest.raises(ValueError, match="checkpoint_root"):
      _load_lite_time(persist=True, checkpoint_root=None)

  def test_persist_true_creates_isolated_directory_per_build(self, tmp_path):
    checkpoint_root = str(tmp_path)
    model_a = _load_lite_time(n_classifiers=1, n_epochs=1, batch_size=8,
                              persist=True, checkpoint_root=checkpoint_root)
    model_b = _load_lite_time(n_classifiers=1, n_epochs=1, batch_size=8,
                              persist=True, checkpoint_root=checkpoint_root)

    assert model_a.file_path != model_b.file_path
    assert model_a.save_best_model is True
    assert model_a.save_last_model is True
    # both directories were actually created under checkpoint_root
    subdirs = os.listdir(checkpoint_root)
    assert len(subdirs) == 2


class TestLITETimeClassifierWrapper:

  def test_wrapper_is_sklearn_cloneable_unfitted(self):
    classifier = LITETimeClassifier(n_classifiers=1, n_epochs=1, batch_size=8)
    cloned = clone(classifier)
    assert cloned is not classifier
    assert not hasattr(cloned, 'model_')

  def test_predict_proba_exposed_by_default(self):
    classifier = LITETimeClassifier(n_classifiers=1, n_epochs=1, batch_size=8)
    assert hasattr(classifier, 'predict_proba')

  def test_predict_proba_absent_when_disabled(self):
    classifier = LITETimeClassifier(n_classifiers=1, n_epochs=1, batch_size=8,
                                    supports_predict_proba=False)
    assert not hasattr(classifier, 'predict_proba')

  def test_fit_predict_predict_proba_roundtrip(self):
    X, y = _make_data(n=30, length=40, n_classes=2, seed=1)
    classifier = LITETimeClassifier(n_classifiers=1, n_epochs=2, batch_size=8,
                                    random_state=0)
    classifier.fit(X, y)

    assert set(classifier.classes_) == {'N', 'V'}

    preds = classifier.predict(X)
    assert preds.shape == (30,)
    assert set(np.unique(preds)).issubset({'N', 'V'})

    proba = classifier.predict_proba(X)
    assert proba.shape == (30, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

  def test_rejects_unexpected_fit_params(self):
    classifier = LITETimeClassifier(n_classifiers=1, n_epochs=1, batch_size=8,
                                    fit_params={'some_unexpected_kwarg': True})
    X, y = _make_data(n=10, length=20, n_classes=2, seed=2)
    with pytest.raises(TypeError, match="fit_params"):
      classifier.fit(X, y)


class TestQuackIntegration:

  def test_cc_quantifier_end_to_end(self):
    """CC just calls classifier.fit/predict directly (no CV) -- the
    simplest possible wiring check."""
    X, y = _make_data(n=30, length=32, n_classes=2, seed=3)
    classifier = LITETimeClassifier(n_classifiers=1, n_epochs=2, batch_size=8,
                                    random_state=0)
    quantifier = CC(classifier=classifier)
    quantifier.fit(X, y)

    prevalences = quantifier.predict(X)
    assert prevalences.shape == (2,)
    np.testing.assert_allclose(prevalences.sum(), 1.0, atol=1e-6)

  def test_pacc_quantifier_parallel_folds_do_not_collide(self, tmp_path):
    """PACC is a BaseCalibratedQuantifier: fits one classifier per CV fold
    (plus one final full-data refit), dispatched via joblib.Parallel.
    With n_jobs=2 and persist=True, this exercises the exact scenario
    this module's docstring warns about -- concurrent folds writing
    checkpoint files under the same classifier config. If per-build
    directory isolation in `_load_lite_time` were broken, this would
    either raise (file lock/permission errors from Keras) or silently
    produce corrupted/overwritten checkpoints; neither shows up in the
    quantifier's numeric output alone, so we inspect the checkpoint
    directories directly rather than only checking `predict()` succeeds.

    Uses PACC rather than ACC/GAC here deliberately: those use hard-label
    ("predict") out-of-fold, and `BaseCalibratedQuantifier.fit()` (in
    `quack`, not this adapter) allocates that OOF buffer as a plain
    `np.zeros(n_samples)` float array regardless of `y.dtype` -- fitting
    them directly on string multiclass labels crashes with
    "could not convert string to float", independently of which
    classifier is used (reproduces identically with a plain
    LogisticRegression). See this module's docstring note on quack's
    string-label limitation. PACC's out-of-fold is `predict_proba`
    (always numeric), so it isn't affected and is the right choice for
    isolating the collision-avoidance behavior actually under test here.
    """
    X, y = _make_data(n=40, length=32, n_classes=2, seed=4)
    checkpoint_root = str(tmp_path)

    classifier = LITETimeClassifier(n_classifiers=1, n_epochs=2, batch_size=8,
                                    random_state=0,
                                    persist=True, checkpoint_root=checkpoint_root)
    quantifier = PACC(classifier=classifier, cv=3, n_jobs=2)
    quantifier.fit(X, y)

    prevalences = quantifier.predict(X)
    assert prevalences.shape == (2,)
    np.testing.assert_allclose(prevalences.sum(), 1.0, atol=1e-6)

    # 3 folds + 1 final full-data refit = 4 independent builds, each its
    # own subdirectory under checkpoint_root, each holding a real
    # checkpoint file (not empty, not silently skipped/overwritten).
    build_dirs = [d for d in os.listdir(checkpoint_root)
                  if os.path.isdir(os.path.join(checkpoint_root, d))]
    assert len(build_dirs) == 4
    for d in build_dirs:
      files = os.listdir(os.path.join(checkpoint_root, d))
      assert len(files) > 0, f"expected a checkpoint file in {d}, found none"
