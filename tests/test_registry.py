"""Tests for `ecg_quantification.quantifiers._registry`'s
`build_multiclass_quantifiers`, specifically the "wrap all six native
multiclass quantifiers in `LabelEncodedQuantifier`" decision (a follow-up
to item 13/GAC's original special-case-only wrapping): standardizing
this way means every native multiclass quantifier fits its base
classifier on byte-identical integer-encoded labels, letting
`CachedFitClassifier` share one fit-cache group across all six instead of
splitting GAC off into its own (see that module's docstring).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from ecg_quantification.quantifiers import (
  LabelEncodedQuantifier, build_multiclass_quantifiers, build_all_quantifiers,
)


def _make_data(n=90, n_features=5, seed=0):
  rng = np.random.default_rng(seed)
  X = rng.normal(size=(n, n_features))
  symbols = np.array(['N', 'V', 'A'])
  y = symbols[rng.integers(0, 3, n)]
  return X, y


class TestNativeMulticlassQuantifiersAreAllLabelEncoded:

  def test_every_native_multiclass_recipe_is_wrapped(self):
    registry = build_multiclass_quantifiers(lambda: LogisticRegression(max_iter=1000), cv=3)
    assert set(registry.keys()) == {'CC', 'PCC', 'GAC', 'GPAC', 'FM', 'EMQ'}
    for name, quantifier in registry.items():
      assert isinstance(quantifier, LabelEncodedQuantifier), (
        f"{name} should be wrapped in LabelEncodedQuantifier"
      )

  def test_threshold_family_native_variants_are_also_wrapped(self):
    registry = build_multiclass_quantifiers(
      lambda: LogisticRegression(max_iter=1000), cv=3, include_threshold_family=True,
    )
    for name in ('X-native', 'Max-native', 'T50-native', 'MedianSweep-native'):
      assert isinstance(registry[name], LabelEncodedQuantifier), name

  def test_fit_predict_roundtrip_exposes_original_string_labels(self):
    X, y = _make_data(seed=1)
    registry = build_multiclass_quantifiers(lambda: LogisticRegression(max_iter=1000), cv=3)

    for name, quantifier in registry.items():
      quantifier.fit(X, y)
      assert set(quantifier.classes_) == {'N', 'V', 'A'}, name
      prevalences = quantifier.predict(X)
      assert prevalences.shape == (3,), name
      np.testing.assert_allclose(prevalences.sum(), 1.0, atol=1e-6, err_msg=name)

  def test_ovr_wrapped_binary_quantifiers_are_not_label_encoded(self):
    """OneVsRestQuantifier already binarizes labels to int internally
    (`(y == c).astype(int)`) -- it never needs (or gets) an extra
    LabelEncodedQuantifier layer."""
    registry = build_all_quantifiers('LR', lambda: LogisticRegression(max_iter=1000), cv=3, n_jobs=None)
    ovr_names = {'LR-ACC', 'LR-PACC', 'LR-X', 'LR-Max', 'LR-T50', 'LR-MedianSweep',
                 'LR-HDy', 'LR-DyS', 'LR-FormanMM', 'LR-CDE'}
    for name in ovr_names:
      assert not isinstance(registry[name], LabelEncodedQuantifier), name
