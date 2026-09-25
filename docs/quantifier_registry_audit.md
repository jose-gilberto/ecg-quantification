# Quantifier registry audit (Phase 1, journal extension)

Empirically verified (fit each quantifier directly on 5-class y,
bypassing OneVsRestQuantifier) which quack quantifiers accept a
multiclass `y` natively vs. raise `ValueError` for >2 classes.

| Natively multiclass | Binary-only (needs OneVsRestQuantifier) |
|---|---|
| CC, PCC, GAC, GPAC, FM, EMQ | ACC, PACC, HDy, DyS, FormanMM, CDE |
| X, Max, T50, MedianSweep    | |
| HDx, ReadMe, ED             | |

**Note**: quack's own module organization suggests the threshold family
(X, Max, T50, MedianSweep) is binary-only, matching
`OneVsRestQuantifier`'s own module docstring in this repo. Empirically,
however, all four accept multiclass `y` directly without error. Both
variants (native and OvR-wrapped) are now available via
`build_multiclass_quantifiers(..., include_threshold_family=True)` /
`build_all_quantifiers(..., include_threshold_family=True)`, enabling a
direct OvR-vs-native ablation for these four methods.

**Note (string labels)**: "accepts multiclass `y` natively" above means
"doesn't raise `ValueError` for >2 classes" -- it does not mean every
native multiclass quantifier tolerates *string* labels (this project's
actual `y` dtype: 'N', 'V', 'A', 'L', 'R'). Of the six natively
multiclass quantifiers, GAC alone breaks on string `y` with
`ValueError: could not convert string to float`, because
`quack.quantifiers.base.BaseCalibratedQuantifier.fit()` allocates its
hard-label out-of-fold buffer (`_get_oof_method() == "predict"`, GAC's
case) as a plain `float64` array regardless of `y`'s dtype. GPAC, FM, EM
all use `_get_oof_method() == "predict_proba"` (always numeric) and are
unaffected; every OvR-wrapped binary quantifier is fit on `int`-binarized
labels by `OneVsRestQuantifier` regardless. This was found (and is
deliberately *not* patched in `quack` itself, by project decision) while
integrating `LITETimeClassifier`; `build_multiclass_quantifiers` wraps
GAC in `ecg_quantification.quantifiers.LabelEncodedQuantifier` to work
around it -- see that class's module docstring for the full explanation.
Any future native multiclass quantifier added to this registry should be
checked the same way before assuming string labels "just work".