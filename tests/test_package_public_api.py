"""Guards against `ecg_quantification.__all__` drifting out of sync with
what the package actually imports at top level -- exactly the bug fixed
in this module (several quantifier-registry functions and
`OneVsRestQuantifier` were importable but silently missing from
`__all__`, so `from ecg_quantification import *` and star-import-aware
tooling couldn't see them).
"""
import ecg_quantification as pkg


EXPECTED_PUBLIC_SYMBOLS = {
  'ClinicalPrevalenceBagGenerator',
  'APPBagGenerator',
  'ExperimentCheckpoint',
  'CLINICAL_RANGES',
  'ZONE_WEIGHTS',
  'RISK_ZONE_LABELS',
  'BaseDatasetDownloader',
  'MITBIHDownloader',
  'ECGPreprocessor',
  'MULTICLASS_SYMBOLS',
  'find_variation_window',
  'plot_signal_window',
  'patient_train_test_split',
  'patient_kfold_split',
  'assert_no_patient_leakage',
  'guaranteed_patient_train_test_split',
  'ZoneCrossingError',
  'default_base_classifiers',
  'build_multiclass_quantifiers',
  'build_feature_space_quantifiers',
  'build_binary_quantifiers_for_ovr',
  'build_all_quantifiers',
  'OneVsRestQuantifier',
}


def test_all_declares_exactly_the_expected_public_symbols():
  assert set(pkg.__all__) == EXPECTED_PUBLIC_SYMBOLS


def test_every_symbol_in_all_is_actually_importable():
  # catches the inverse mistake: a name listed in __all__ that was
  # renamed/removed but never cleaned up there
  for name in pkg.__all__:
    assert hasattr(pkg, name), f"'{name}' is listed in __all__ but not actually bound on the package."


def test_no_public_looking_attribute_is_missing_from_all():
  # catches the original bug: something importable and clearly public
  # (no leading underscore, not a submodule/dunder) that never made it
  # into __all__
  import types

  public_attrs = {
    name for name in vars(pkg)
    if not name.startswith('_')
    and not isinstance(getattr(pkg, name), types.ModuleType)
  }
  undeclared = public_attrs - set(pkg.__all__)
  assert not undeclared, f"Public symbols missing from __all__: {sorted(undeclared)}"
