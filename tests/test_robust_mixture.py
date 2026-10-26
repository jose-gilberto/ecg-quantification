"""Tests for `ecg_quantification.quantifiers._robust_mixture`: the fixes
for the first full multiclass run's crash while evaluating HDx
(`ValueError: shapes (18506640,5) and (2,) not aligned`) -- HDx/ReadMe on
continuous features, and quack's binary-only solver fallback on a
5-class problem.
"""
import warnings

import cvxpy as cvx
import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from quack.quantifiers import GAC as QuackGAC
from quack.quantifiers import HDx as QuackHDx
from ecg_quantification.quantifiers import _robust_mixture
from ecg_quantification.quantifiers._robust_mixture import (
  FM, GAC, GPAC, HDx, ReadMe, DiscretizedQuantifier, _SafeRawSubspaceMixture,
)

pytestmark = pytest.mark.filterwarnings(
  "ignore:.*DPP.*:UserWarning",
  "ignore:Solution may be inaccurate.*:UserWarning",
)

LABELS = np.array(['N', 'V', 'A', 'L', 'R'])


def _beats(n, seed, prevalence=None, n_features=60, shift=1.5):
  """Continuous, z-score-like 'beats' with a class-specific bump."""
  rng = np.random.default_rng(seed)
  y = LABELS[rng.choice(len(LABELS), n, p=prevalence)]
  X = rng.normal(size=(n, n_features))
  for i, label in enumerate(LABELS):
    X[y == label, 10 * i: 10 * i + 8] += shift
  return X, y


def _true(y_test, classes):
  return np.array([np.mean(y_test == c) for c in classes])


def _assert_simplex(prevalences, n_classes):
  assert prevalences.shape == (n_classes,)
  assert np.all(prevalences >= -1e-9)
  np.testing.assert_allclose(prevalences.sum(), 1.0, atol=1e-6)


class TestQuackBehaviorThisFixes:

  def test_quack_hdx_on_continuous_features_is_degenerate(self):
    """The real run's situation, at small scale: every training value is
    its own bin, so no test value ever matches one -- the test frequency
    vector is all zeros and there is nothing to fit prevalences to."""
    X, y = _beats(300, seed=1)
    X_test, _ = _beats(50, seed=2)
    quantifier = QuackHDx().fit(X, y)
    assert quantifier.conditional_matrix_.shape == (60 * 300, 5)
    assert quantifier._compute_score(X_test).sum() == 0

  def test_quack_solver_fallback_is_binary_only(self):
    """What crashed the run once CLARABEL failed on that problem."""
    X, y = _beats(300, seed=1)
    quantifier = QuackHDx().fit(X, y)
    test_frequencies = quantifier._compute_score(X[:50])
    with pytest.raises(ValueError, match="not aligned"):
      quantifier._golden_section_search_fallback(test_frequencies)


class TestMulticlassSafeFallback:

  def _fitted_gac(self, cls=GAC, seed=3):
    # integer labels: quack's GAC itself can't take string labels (what
    # LabelEncodedQuantifier handles in the registry)
    X, y = _beats(400, seed=seed)
    return cls(classifier=LogisticRegression(max_iter=1000), cv=3).fit(X, np.unique(y, return_inverse=True)[1])

  def test_raw_continuous_hdx_no_longer_crashes(self):
    X, y = _beats(300, seed=1)
    X_test, _ = _beats(50, seed=2)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      _assert_simplex(HDx().fit(X, y).predict(X_test), 5)

  def test_falls_back_to_slsqp_when_every_solver_fails(self, monkeypatch):
    quantifier = self._fitted_gac()
    X_test, _ = _beats(200, seed=4, prevalence=[0.6, 0.1, 0.1, 0.1, 0.1])
    reference = quantifier.predict(X_test)

    def failing(self, test_frequencies):
      raise cvx.SolverError("simulated CLARABEL failure")

    monkeypatch.setattr(GAC, "_solve_via_convex_programming", failing)
    monkeypatch.setattr(_robust_mixture, "_RETRY_SOLVERS", ())
    with pytest.warns(UserWarning, match="SLSQP"):
      fallback = quantifier.predict(X_test)
    _assert_simplex(fallback, 5)
    np.testing.assert_allclose(fallback, reference, atol=0.03)

  def test_retries_with_another_conic_solver_first(self, monkeypatch):
    quantifier = self._fitted_gac()
    X_test, _ = _beats(200, seed=5)
    reference = quantifier.predict(X_test)
    original = QuackGAC._solve_via_convex_programming

    def solve_then_fail(self, test_frequencies):
      original(self, test_frequencies)  # sets the problem's parameter values
      raise cvx.SolverError("simulated CLARABEL failure")

    monkeypatch.setattr(GAC, "_solve_via_convex_programming", solve_then_fail)
    with pytest.warns(UserWarning, match="solved with SCS"):
      retried = quantifier.predict(X_test)
    np.testing.assert_allclose(retried, reference, atol=0.02)

  def test_binary_problems_keep_quacks_behavior(self, monkeypatch):
    X, y = _beats(300, seed=6)
    y_binary = (y == 'N').astype(int)
    quantifier = GAC(classifier=LogisticRegression(max_iter=1000), cv=3).fit(X, y_binary)

    def failing(self, test_frequencies):
      raise cvx.SolverError("simulated")

    monkeypatch.setattr(GAC, "_solve_via_convex_programming", failing)
    with pytest.warns(UserWarning, match="Falling back to GSS"):
      _assert_simplex(quantifier.predict(X[:100]), 2)

  @pytest.mark.parametrize("cls", [GAC, GPAC, FM])
  def test_are_drop_in_quack_quantifiers(self, cls):
    quantifier = cls(classifier=LogisticRegression(max_iter=1000), cv=3)
    assert isinstance(quantifier, getattr(__import__("quack.quantifiers", fromlist=[cls.__name__]), cls.__name__))
    assert type(clone(quantifier)) is cls
    assert clone(quantifier).get_params()["cv"] == 3


class TestDiscretizedQuantifier:

  def test_discretized_hdx_estimates_prevalences(self):
    X, y = _beats(1500, seed=7)
    prevalence = [0.6, 0.15, 0.1, 0.1, 0.05]
    X_test, y_test = _beats(500, seed=8, prevalence=prevalence)
    quantifier = DiscretizedQuantifier(HDx(), n_bins=8).fit(X, y)
    assert quantifier.quantifier_.conditional_matrix_.shape == (60 * 8, 5)
    assert list(quantifier.classes_) == sorted(LABELS)
    estimate = quantifier.predict(X_test)
    _assert_simplex(estimate, 5)
    assert np.abs(estimate - _true(y_test, quantifier.classes_)).mean() < 0.05

  def test_readme_submodels_use_the_safe_fallback_and_pickle(self, tmp_path):
    X, y = _beats(600, seed=9)
    quantifier = DiscretizedQuantifier(ReadMe(n_subsets=10, random_state=0), n_bins=2).fit(X, y)
    assert all(type(sub) is _SafeRawSubspaceMixture for sub in quantifier.quantifier_.sub_quantifiers_)
    path = str(tmp_path / "readme.joblib")
    joblib.dump(quantifier, path)
    reloaded = joblib.load(path)
    np.testing.assert_allclose(reloaded.predict(X[:100]), quantifier.predict(X[:100]))

  def test_bins_are_fit_on_training_data_only(self):
    X, y = _beats(300, seed=10)
    quantifier = DiscretizedQuantifier(HDx(), n_bins=4).fit(X, y)
    edges = [e.copy() for e in quantifier.discretizer_.bin_edges_]
    quantifier.predict(X[:50] * 10)  # out-of-range test values: clipped into end bins
    assert all(np.array_equal(a, b) for a, b in zip(edges, quantifier.discretizer_.bin_edges_))

  def test_clone_and_params(self):
    quantifier = DiscretizedQuantifier(HDx(), n_bins=5, strategy="uniform")
    cloned = clone(quantifier)
    assert cloned.get_params()["n_bins"] == 5
    assert cloned.get_params()["strategy"] == "uniform"
    assert not hasattr(cloned, "quantifier_")


class TestRegistry:

  def test_feature_space_names_carry_the_bin_count(self):
    from ecg_quantification.quantifiers import build_feature_space_quantifiers
    registry = build_feature_space_quantifiers(hdx_bins=8, readme_bins=2, random_state=0)
    assert set(registry) == {'HDx_8bins', 'ReadMe_2bins', 'ED'}
    assert registry['HDx_8bins'].n_bins == 8
    assert registry['ReadMe_2bins'].quantifier.random_state == 0

  def test_native_mixture_quantifiers_are_the_safe_subclasses(self):
    from ecg_quantification.quantifiers import build_multiclass_quantifiers
    registry = build_multiclass_quantifiers(lambda: LogisticRegression(max_iter=1000), cv=3)
    for name, cls in (('GAC', GAC), ('GPAC', GPAC), ('FM', FM)):
      assert type(registry[name].quantifier) is cls
