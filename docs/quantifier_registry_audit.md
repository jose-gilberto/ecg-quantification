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