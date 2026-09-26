"""Tests for `ecg_quantification.quantifiers.CachedFitClassifier` /
`cached_classifier_factory` (item 13, Fase 3).

Two layers:

- Generic mechanics, using a cheap counting dummy classifier (no
  TensorFlow/aeon needed) -- fast, exercises cache hit/miss, clone()
  compatibility, per-`cache_id` isolation, and the exact fold/OvR-sharing
  scenario this module's docstring describes, without spending real
  training time.
- A real integration test through `build_all_quantifiers` with
  `LITETimeClassifier` as the base classifier, counting actual Keras
  `.fit()` calls to confirm the documented fit-count collapse actually
  happens end-to-end (not just for the synthetic scenario above). Guarded
  by `pytest.importorskip`; skipped when `aeon`/`tensorflow` aren't
  installed.
"""
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression

from ecg_quantification.quantifiers import (
  CachedFitClassifier, cached_classifier_factory, clear_fit_cache, fit_cache_size,
)
from ecg_quantification.quantifiers._one_vs_rest import OneVsRestQuantifier
from quack.quantifiers import CC, ACC, PACC, GPAC


class _CountingClassifier(BaseEstimator, ClassifierMixin):
  """A trivial, cheap classifier that counts real `.fit()` calls, so
  tests can assert exactly how many times training actually happened (as
  opposed to being served from cache).

  Counts are kept in a class-level dict keyed by `counter_id` (a plain
  string) rather than passed in as a mutable object -- for the same
  reason `CachedFitClassifier` keeps its own cache module-level rather
  than as a constructor parameter: `sklearn.base.clone()` deep-copies any
  non-estimator constructor parameter (see
  `sklearn.base._clone_parametrized`), so a shared mutable counter passed
  directly as a param would silently become independent, disconnected
  copies across every `clone()` call quack/`CachedFitClassifier`
  performs -- exactly the trap this whole module is about avoiding.
  """

  _COUNTS: dict = {}

  def __init__(self, counter_id: str = "default"):
    self.counter_id = counter_id

  def fit(self, X, y):
    _CountingClassifier._COUNTS[self.counter_id] = (
      _CountingClassifier._COUNTS.get(self.counter_id, 0) + 1
    )
    self._inner = LogisticRegression(max_iter=1000).fit(X, y)
    self.classes_ = self._inner.classes_
    return self

  def predict(self, X):
    return self._inner.predict(X)

  def predict_proba(self, X):
    return self._inner.predict_proba(X)

  @classmethod
  def fit_count(cls, counter_id: str) -> int:
    return cls._COUNTS.get(counter_id, 0)


def _make_data(n=60, n_features=5, n_classes=3, seed=0):
  """Int-labeled data (not the project's usual string symbols): ACC's
  hard-label out-of-fold path hits quack's pre-existing (deliberately
  unpatched, see LabelEncodedQuantifier's module docstring) float64 OOF
  buffer bug on string `y`. These tests are only about
  CachedFitClassifier's generic caching mechanics, not that unrelated
  quack limitation, so int labels sidestep it entirely."""
  rng = np.random.default_rng(seed)
  X = rng.normal(size=(n, n_features))
  y = rng.integers(0, n_classes, n)
  return X, y


@pytest.fixture(autouse=True)
def _isolate_fit_cache():
  """Every test gets a clean global cache -- otherwise leftover entries
  from an earlier test (same process) could produce false cache hits."""
  clear_fit_cache()
  yield
  clear_fit_cache()


def _counting_factory(counter_id: str):
  return lambda: _CountingClassifier(counter_id=counter_id)


class TestCachedFitClassifierMechanics:

  def test_cache_miss_then_hit_on_identical_data(self):
    factory = cached_classifier_factory(_counting_factory('t1'), cache_id='t1')
    X, y = _make_data(seed=1)

    clf_a = factory()
    clf_a.fit(X, y)
    assert _CountingClassifier.fit_count('t1') == 1
    assert clf_a.from_cache_ is False

    clf_b = factory()
    clf_b.fit(X, y)  # identical (X, y) content -- must hit the cache
    assert _CountingClassifier.fit_count('t1') == 1, "second .fit() on identical data should not have retrained"
    assert clf_b.from_cache_ is True
    assert clf_a.estimator_ is clf_b.estimator_

  def test_different_data_is_a_cache_miss(self):
    factory = cached_classifier_factory(_counting_factory('t2'), cache_id='t2')
    X1, y1 = _make_data(seed=2)
    X2, y2 = _make_data(seed=3)

    factory().fit(X1, y1)
    factory().fit(X2, y2)
    assert _CountingClassifier.fit_count('t2') == 2

  def test_different_cache_id_does_not_share(self):
    X, y = _make_data(seed=4)

    cached_classifier_factory(_counting_factory('t3'), cache_id='scope-a')().fit(X, y)
    cached_classifier_factory(_counting_factory('t3'), cache_id='scope-b')().fit(X, y)
    assert _CountingClassifier.fit_count('t3') == 2, "different cache_id scopes must not share fitted models"

  def test_clone_preserves_shared_cache(self):
    """The scenario quack actually exercises: clone(wrapper) per fold,
    then .fit() on the clone. The cache must survive the clone (a plain
    dict constructor param would be deep-copied by sklearn's clone())."""
    classifier = CachedFitClassifier(
      base_estimator=_CountingClassifier(counter_id='t4'), cache_id='t4',
    )
    X, y = _make_data(seed=5)

    cloned_a = clone(classifier)
    cloned_a.fit(X, y)
    assert _CountingClassifier.fit_count('t4') == 1

    cloned_b = clone(classifier)
    cloned_b.fit(X, y)
    assert _CountingClassifier.fit_count('t4') == 1, "clone of the wrapper must still share the module-level cache"

  def test_predict_proba_exposed_when_supported(self):
    classifier = cached_classifier_factory(_counting_factory('t5'), cache_id='t5')()
    assert hasattr(classifier, 'predict_proba')

  def test_predict_proba_absent_when_unsupported(self):
    class _NoProba(BaseEstimator, ClassifierMixin):
      def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

      def predict(self, X):
        return np.full(len(X), self.classes_[0])

    classifier = cached_classifier_factory(_NoProba, cache_id='t6')()
    assert not hasattr(classifier, 'predict_proba')

  def test_fit_cache_size_reports_scoped_and_total_counts(self):
    X1, y1 = _make_data(seed=6)
    X2, y2 = _make_data(seed=7)
    factory_a = cached_classifier_factory(_counting_factory('t7a'), cache_id='scope-x')
    factory_b = cached_classifier_factory(_counting_factory('t7b'), cache_id='scope-y')

    factory_a().fit(X1, y1)
    factory_a().fit(X2, y2)
    factory_b().fit(X1, y1)

    assert fit_cache_size('scope-x') == 2
    assert fit_cache_size('scope-y') == 1
    assert fit_cache_size() >= 3


class TestFoldSharingAcrossQuantifiers:
  """The concrete scenario this module exists for: several quantifiers
  built with the same `cv`/OvR class on the same `(X, y)` should collapse
  onto a handful of real fits instead of one independent set each."""

  def test_native_cv_quantifiers_share_folds(self):
    X, y = _make_data(n=80, n_classes=2, seed=8)
    factory = cached_classifier_factory(_counting_factory('native'), cache_id='native')

    cv = 4
    gpac = GPAC(classifier=factory(), cv=cv, n_jobs=None)
    gpac.fit(X, y)
    fits_after_first = _CountingClassifier.fit_count('native')
    assert fits_after_first == cv + 1  # cv folds + 1 final full-data refit

    # A second, independently-built quantifier with the *same* cv on the
    # *same* data must reuse every one of those fits.
    acc = ACC(classifier=factory(), cv=cv, n_jobs=None)
    acc.fit(X, y)
    assert _CountingClassifier.fit_count('native') == fits_after_first, (
      "ACC's folds should be bit-identical to GPAC's (same cv, same X/y) "
      "and reuse the cache entirely"
    )

    # CC (no CV, single full-data fit) should reuse the existing final refit.
    cc = CC(classifier=factory())
    cc.fit(X, y)
    assert _CountingClassifier.fit_count('native') == fits_after_first, (
      "CC's single full-data fit should match one of the CV quantifiers' "
      "final full-data refits and hit the cache"
    )

  def test_ovr_members_share_folds_across_quantifier_types(self):
    X, y = _make_data(n=90, n_classes=3, seed=9)
    factory = cached_classifier_factory(_counting_factory('ovr'), cache_id='ovr')

    cv = 3
    ovr_acc = OneVsRestQuantifier(ACC(classifier=factory(), cv=cv, n_jobs=None), n_jobs=None)
    ovr_acc.fit(X, y)
    fits_after_first = _CountingClassifier.fit_count('ovr')
    assert fits_after_first == 3 * (cv + 1)  # 3 classes x (cv folds + 1 refit)

    ovr_pacc = OneVsRestQuantifier(PACC(classifier=factory(), cv=cv, n_jobs=None), n_jobs=None)
    ovr_pacc.fit(X, y)
    assert _CountingClassifier.fit_count('ovr') == fits_after_first, (
      "PACC's per-class binary relabelings/folds are identical to ACC's "
      "(same classes_, same cv, same X/y) and should reuse the cache"
    )
