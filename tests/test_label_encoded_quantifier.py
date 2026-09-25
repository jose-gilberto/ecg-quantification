"""Tests for `ecg_quantification.quantifiers.LabelEncodedQuantifier`.

Layers, mirroring `test_one_vs_rest.py`'s approach:

- Direct unit tests on `.fit()`/`.predict()`/`.classes_` with a real
  `quack.quantifiers.GAC` (the one quantifier in this project's registry
  actually affected by the hard-label OOF dtype limitation this wrapper
  works around -- see the module's docstring).
- A same-assignment equivalence check against fitting `GAC` directly on
  pre-encoded integer labels: the wrapper must not just avoid crashing,
  it must produce identical estimated prevalences.
- A `GPAC` (predict_proba-based, never affected by the bug) check,
  confirming the wrapper is a safe no-op there too -- every multiclass
  quantifier in the registry should be able to go through it uniformly.
- A real end-to-end integration test with `LITETimeClassifier`, the
  concrete case this wrapper is being built ahead of: fitting `GAC`
  natively multiclass (no `OneVsRestQuantifier`) directly on 5-class
  ECG-like string labels.
"""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from quack.quantifiers import GAC, GPAC
from ecg_quantification.quantifiers import LabelEncodedQuantifier


SYMBOLS = np.array(['N', 'V', 'A', 'L', 'R'])


def _string_multiclass_dataset(n_classes=5, n_samples=200, seed=0):
  X, y_int = make_classification(
    n_samples=n_samples, n_classes=n_classes, n_informative=6,
    n_clusters_per_class=1, random_state=seed,
  )
  y = SYMBOLS[:n_classes][y_int]
  return X, y, y_int


class TestBasicFitPredict:

  def test_gac_fits_and_predicts_on_string_labels_without_crashing(self):
    X, y, _ = _string_multiclass_dataset()
    quantifier = LabelEncodedQuantifier(GAC(classifier=LogisticRegression(max_iter=1000), cv=5))
    quantifier.fit(X, y)

    prevalences = quantifier.predict(X)
    assert prevalences.shape == (5,)
    assert prevalences.sum() == pytest.approx(1.0)

  def test_classes_exposed_as_original_sorted_string_labels(self):
    X, y, _ = _string_multiclass_dataset()
    quantifier = LabelEncodedQuantifier(GAC(classifier=LogisticRegression(max_iter=1000), cv=5))
    quantifier.fit(X, y)

    assert list(quantifier.classes_) == sorted(SYMBOLS.tolist())
    assert quantifier.n_classes_ == 5
    assert quantifier.train_prevalence_.sum() == pytest.approx(1.0)
    assert quantifier.train_prevalence_.shape == (5,)

  def test_wrapped_quantifier_template_is_not_mutated(self):
    """The `quantifier` passed to the constructor is a template, not the
    fitted object -- `.fit()` must clone it, matching how
    `OneVsRestQuantifier` treats `base_quantifier`."""
    X, y, _ = _string_multiclass_dataset()
    template = GAC(classifier=LogisticRegression(max_iter=1000), cv=5)
    quantifier = LabelEncodedQuantifier(template)
    quantifier.fit(X, y)

    assert not hasattr(template, 'classes_')
    assert quantifier.quantifier_ is not template
    assert hasattr(quantifier.quantifier_, 'classes_')

  def test_raises_for_fewer_than_two_classes(self):
    X = np.random.default_rng(0).normal(size=(10, 3))
    y = np.array(['N'] * 10)
    quantifier = LabelEncodedQuantifier(GAC(classifier=LogisticRegression(max_iter=1000), cv=5))
    with pytest.raises(ValueError, match="at least 2 distinct classes"):
      quantifier.fit(X, y)

  def test_sklearn_cloneable_unfitted(self):
    quantifier = LabelEncodedQuantifier(GAC(classifier=LogisticRegression(max_iter=1000), cv=5))
    cloned = clone(quantifier)
    assert cloned is not quantifier
    assert not hasattr(cloned, 'classes_')


def _reorder_to_int_class_positions(p_from_string_fit, string_classes_sorted, symbol_map):
  """Maps a wrapper prediction (ordered by `string_classes_sorted`, i.e.
  `quantifier.classes_`) back to the position order `make_classification`'s
  integer labels use (`symbol_map[i]` is the string standing in for
  integer class `i`) -- so it can be compared against a direct fit on
  the pre-`symbol_map`-relabeling integer `y`.

  Needed because `symbol_map` (`SYMBOLS`, generation order) and
  `np.unique(symbol_map)` (alphabetical, what `LabelEncodedQuantifier`
  actually sorts by) disagree: e.g. `SYMBOLS = ['N','V','A','L','R']` but
  `sorted(SYMBOLS) = ['A','L','N','R','V']`. Without this remap, a naive
  element-wise comparison looks like a correctness failure when it is
  really just two different, individually self-consistent orderings of
  the same five numbers -- confirmed by construction here, not asserted.
  """
  p_from_string_fit = np.asarray(p_from_string_fit)
  string_to_int_position = {c: i for i, c in enumerate(symbol_map)}
  reordered = np.zeros_like(p_from_string_fit)
  for wrapper_pos, c in enumerate(string_classes_sorted):
    reordered[string_to_int_position[c]] = p_from_string_fit[wrapper_pos]
  return reordered


class TestNumericEquivalence:

  def test_matches_gac_fit_directly_on_equivalent_int_labels(self):
    """Not just 'doesn't crash': the estimated prevalences from fitting
    through the wrapper on string labels must be identical (after
    accounting for the string/int class ordering difference explained in
    `_reorder_to_int_class_positions`) to fitting `GAC` directly on the
    same assignment, pre-encoded as integers -- confirming the
    encode/decode round-trip introduces no discrepancy of its own."""
    X, y, y_int = _string_multiclass_dataset(seed=1)
    symbol_map = SYMBOLS[:5]

    quantifier = LabelEncodedQuantifier(GAC(classifier=LogisticRegression(max_iter=1000), cv=5))
    quantifier.fit(X, y)
    p_wrapped = _reorder_to_int_class_positions(
      quantifier.predict(X), quantifier.classes_, symbol_map,
    )

    p_direct = (
      GAC(classifier=LogisticRegression(max_iter=1000), cv=5)
      .fit(X, y_int)
      .predict(X)
    )

    np.testing.assert_allclose(p_wrapped, p_direct, atol=1e-10)

  def test_gpac_predict_proba_based_also_works_as_a_safe_no_op(self):
    """GPAC's OOF is predict_proba (always numeric) -- it was never
    affected by the underlying bug, but routing it through this wrapper
    too (for a uniform registry-building rule) must not change its
    result."""
    X, y, y_int = _string_multiclass_dataset(seed=2)
    symbol_map = SYMBOLS[:5]

    quantifier = LabelEncodedQuantifier(GPAC(classifier=LogisticRegression(max_iter=1000), cv=5))
    quantifier.fit(X, y)
    p_wrapped = _reorder_to_int_class_positions(
      quantifier.predict(X), quantifier.classes_, symbol_map,
    )

    p_direct = (
      GPAC(classifier=LogisticRegression(max_iter=1000), cv=5)
      .fit(X, y_int)
      .predict(X)
    )

    np.testing.assert_allclose(p_wrapped, p_direct, atol=1e-10)


class TestLITETimeIntegration:
  """The concrete case this wrapper is being built ahead of: running
  LITETimeClassifier on ecg_quantification's actual multiclass problem,
  natively (no OneVsRestQuantifier), directly on string beat symbols."""

  def test_gac_with_lite_time_classifier_on_string_multiclass_labels(self):
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("aeon")
    from ecg_quantification.quantifiers import LITETimeClassifier

    X, y, _ = _string_multiclass_dataset(n_samples=80, seed=3)
    classifier = LITETimeClassifier(n_classifiers=1, n_epochs=2, batch_size=8, random_state=0)

    quantifier = LabelEncodedQuantifier(GAC(classifier=classifier, cv=3, n_jobs=1))
    quantifier.fit(X, y)

    assert list(quantifier.classes_) == sorted(SYMBOLS.tolist())

    prevalences = quantifier.predict(X)
    assert prevalences.shape == (5,)
    assert prevalences.sum() == pytest.approx(1.0)
