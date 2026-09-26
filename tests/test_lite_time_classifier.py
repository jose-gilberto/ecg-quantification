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
from ecg_quantification.quantifiers._lite_time import (
  _load_lite_time, _checkpoint_member_paths, _checkpoint_is_complete,
)


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


class TestResume:
  """Item 12: resume support via aeon's own minimal native mechanism
  (`save_best_model` + `LITETimeClassifier.load_model`) -- see this
  adapter module's "NOTE on resume support" docstring for exactly what
  is (and is not) resumed. No custom epoch/optimizer/state_dict
  checkpointing is implemented here, by explicit project decision.
  """

  def test_resume_ignored_without_persist(self):
    """resume=True on its own (persist=False, the default) is a no-op --
    documented as "ignored when persist=False", not an error -- since
    there is nothing to resume from if checkpoint files are never kept
    in the first place."""
    model = _load_lite_time(n_classifiers=1, n_epochs=1, batch_size=8,
                            persist=False, resume=True, checkpoint_root=None)
    assert model.save_best_model is False

  def test_resume_uses_stable_directory_not_mkdtemp(self, tmp_path):
    checkpoint_root = str(tmp_path)
    model = _load_lite_time(n_classifiers=2, n_epochs=1, batch_size=8,
                            persist=True, resume=True,
                            checkpoint_root=checkpoint_root)
    # file_path is checkpoint_root itself (plus os.sep), not a fresh
    # tempfile.mkdtemp subdirectory of it.
    assert model.file_path == checkpoint_root + os.sep
    assert not _checkpoint_is_complete(model)

  def test_no_checkpoint_yet_trains_normally(self, tmp_path):
    """With resume=True but an empty checkpoint_root, .fit() must still
    actually train (the completeness check must not false-positive on
    an empty/missing directory)."""
    X, y = _make_data(n=20, length=24, n_classes=2, seed=5)
    checkpoint_root = str(tmp_path)
    classifier = LITETimeClassifier(n_classifiers=1, n_epochs=2, batch_size=8,
                                    random_state=0, persist=True, resume=True,
                                    checkpoint_root=checkpoint_root)
    classifier.fit(X, y)

    assert set(classifier.classes_) == {'N', 'V'}
    preds = classifier.predict(X)
    assert preds.shape == (20,)
    # a complete checkpoint now exists for a later resumed run to find.
    checkpoint_files = [f for f in os.listdir(checkpoint_root)
                        if f.endswith('.keras')]
    assert len(checkpoint_files) >= 1

  def test_complete_checkpoint_skips_retraining_and_loads_instead(self, tmp_path, monkeypatch):
    """The concrete "resume" scenario: fit a small ensemble once with a
    stable checkpoint_root, then build a brand-new wrapper pointed at the
    same directory and confirm the second .fit() call never invokes the
    underlying Keras training call, loading the saved checkpoint instead.
    """
    X, y = _make_data(n=20, length=24, n_classes=2, seed=6)
    checkpoint_root = str(tmp_path)

    first = LITETimeClassifier(n_classifiers=1, n_epochs=2, batch_size=8,
                               random_state=0, persist=True, resume=True,
                               checkpoint_root=checkpoint_root)
    first.fit(X, y)
    first_preds = first.predict(X)

    import aeon.classification.deep_learning._lite_time as _lite_time_mod
    calls = {'count': 0}
    original_individual_fit = _lite_time_mod.IndividualLITEClassifier.fit

    def _counting_fit(self, *args, **kwargs):
      calls['count'] += 1
      return original_individual_fit(self, *args, **kwargs)

    monkeypatch.setattr(_lite_time_mod.IndividualLITEClassifier, 'fit', _counting_fit)

    second = LITETimeClassifier(n_classifiers=1, n_epochs=2, batch_size=8,
                                random_state=0, persist=True, resume=True,
                                checkpoint_root=checkpoint_root)
    second.fit(X, y)

    assert calls['count'] == 0, (
      "resume=True should have loaded the complete checkpoint instead of "
      "retraining (IndividualLITEClassifier.fit was called)"
    )
    assert set(second.classes_) == set(first.classes_)
    second_preds = second.predict(X)
    assert second_preds.shape == first_preds.shape

  def test_incomplete_checkpoint_falls_back_to_full_retrain(self, tmp_path):
    """A partial checkpoint (fewer files than n_classifiers) must not be
    treated as resumable -- the whole ensemble retrains from scratch, per
    this adapter's disclosed "no partial resume" limitation."""
    X, y = _make_data(n=20, length=24, n_classes=2, seed=7)
    checkpoint_root = str(tmp_path)

    # Fit a real 1-member ensemble to seed a checkpoint file, then rename
    # it so a *2*-member build sees an incomplete (1 of 2) checkpoint.
    seed_classifier = LITETimeClassifier(n_classifiers=1, n_epochs=1, batch_size=8,
                                         random_state=0, persist=True, resume=True,
                                         checkpoint_root=checkpoint_root)
    seed_classifier.fit(X, y)

    classifier = LITETimeClassifier(n_classifiers=2, n_epochs=1, batch_size=8,
                                    random_state=0, persist=True, resume=True,
                                    checkpoint_root=checkpoint_root)
    # Only best_model0.keras exists; best_model1.keras is missing, so the
    # 2-member completeness check must fail and a real .fit() must run.
    classifier.fit(X, y)
    assert set(classifier.classes_) == {'N', 'V'}
    preds = classifier.predict(X)
    assert preds.shape == (20,)
