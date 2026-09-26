"""Item 13 (Fase 3), end-to-end: `build_all_quantifiers` with
`LITETimeClassifier` as the base classifier, through
`cached_classifier_factory`, counting real Keras `.fit()` calls to verify
the fit-count collapse `_cached_classifier`'s module docstring describes
actually happens for the real adapter/registry (not just the synthetic
counting-classifier scenario in `test_cached_classifier.py`).

Kept in its own module (rather than folded into
`test_cached_classifier.py`) so the generic, TensorFlow-free caching
mechanics tests there stay fast and always collected, while this module
alone is skipped when `aeon`/`tensorflow` are unavailable.

Deliberately small-scale (3 classes, `n_classifiers=1`, `n_epochs=1`,
`cv=2`) to keep real wall-clock time low while still exercising every
sharing path this module is about: native multiclass CV quantifiers
(GAC/GPAC/FM/EMQ), One-vs-Rest members across several quantifier types,
and CC/PCC's single full-data fit.
"""
import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("aeon")

from ecg_quantification.quantifiers import (
  LITETimeClassifier, build_all_quantifiers, cached_classifier_factory, clear_fit_cache,
)


def _make_data(n=45, length=24, n_classes=3, seed=0):
  rng = np.random.default_rng(seed)
  X = rng.normal(size=(n, length)).astype(np.float64)
  symbols = np.array(['N', 'V', 'A'])[:n_classes]
  y = symbols[rng.integers(0, n_classes, n)]
  return X, y


@pytest.fixture(autouse=True)
def _isolate_fit_cache():
  clear_fit_cache()
  yield
  clear_fit_cache()


class TestBuildAllQuantifiersWithLiteTime:

  def test_full_registry_fits_and_predicts_with_collapsed_fit_count(self, monkeypatch):
    X, y = _make_data(n=45, length=24, n_classes=3, seed=0)
    cv = 2  # cv folds + 1 final refit = 3 fits per independent training subset

    lite_time_factory = lambda: LITETimeClassifier(
      n_classifiers=1, n_epochs=1, batch_size=8, random_state=0,
    )
    classifier_factory = cached_classifier_factory(lite_time_factory, cache_id='build_all-lite_time')

    registry = build_all_quantifiers(
      'LITETime', classifier_factory, cv=cv, n_jobs=None, parallel_backend='loky',
    )
    # 6 native (CC, PCC, GAC, GPAC, FM, EMQ) + 10 OvR-wrapped binary
    assert len(registry) == 16

    import aeon.classification.deep_learning._lite_time as _lite_time_mod
    calls = {'count': 0}
    original_individual_fit = _lite_time_mod.IndividualLITEClassifier.fit

    def _counting_fit(self, *args, **kwargs):
      calls['count'] += 1
      return original_individual_fit(self, *args, **kwargs)

    monkeypatch.setattr(_lite_time_mod.IndividualLITEClassifier, 'fit', _counting_fit)

    for name, quantifier in registry.items():
      quantifier.fit(X, y)

    # GPAC/FM/EMQ (string y) share (cv + 1) fits; GAC gets its own
    # (cv + 1)-fit group since LabelEncodedQuantifier int-encodes y
    # before delegating (see _cached_classifier's module docstring, "On
    # not sharing across differently-labeled data"); OvR quantifiers
    # share (n_classes * (cv + 1)) fits across every one of the 10 OvR
    # quantifier types; CC/PCC reuse the GPAC/FM/EMQ group's full-data
    # refit. n_classifiers=1, so each LITETimeClassifier.fit() call
    # trains exactly 1 IndividualLITEClassifier.
    n_classes = 3
    expected_max_real_fits = 2 * (cv + 1) + n_classes * (cv + 1)
    assert calls['count'] <= expected_max_real_fits, (
      f"expected at most {expected_max_real_fits} real LITETime trainings "
      f"(collapsed via CachedFitClassifier), got {calls['count']}"
    )
    # sanity: caching must have actually engaged, not merely stayed under
    # the ceiling by accident -- naively (no caching) this would be
    # 4 native x (cv+1) + 10 OvR x n_classes x (cv+1) + 2 (CC, PCC)
    naive_fits = 4 * (cv + 1) + 10 * n_classes * (cv + 1) + 2
    assert calls['count'] < naive_fits
    print(f"real LITETime trainings: {calls['count']} (naive would have been {naive_fits})")

    for name, quantifier in registry.items():
      prevalences = quantifier.predict(X)
      assert prevalences.shape == (n_classes,)
      np.testing.assert_allclose(prevalences.sum(), 1.0, atol=1e-6, err_msg=name)
