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
multiclass quantifiers, GAC alone actually *breaks* on string `y` with
`ValueError: could not convert string to float`, because
`quack.quantifiers.base.BaseCalibratedQuantifier.fit()` allocates its
hard-label out-of-fold buffer (`_get_oof_method() == "predict"`, GAC's
case) as a plain `float64` array regardless of `y`'s dtype. GPAC, FM, EM
all use `_get_oof_method() == "predict_proba"` (always numeric) and
CC/PCC never build an OOF buffer at all, so none of the other five would
actually break; every OvR-wrapped binary quantifier is fit on
`int`-binarized labels by `OneVsRestQuantifier` regardless. This was
found (and is deliberately *not* patched in `quack` itself, by project
decision) while integrating `LITETimeClassifier`.

`build_multiclass_quantifiers` wraps *all six* native multiclass
quantifiers (not only GAC) in
`ecg_quantification.quantifiers.LabelEncodedQuantifier` -- see that
class's module docstring for the full explanation. Standardizing this
way (rather than a GAC-only special case) means every one of them fits
its base classifier on the exact same integer-encoded labels
internally, which is what lets `CachedFitClassifier` (item 13, below)
share one fit-cache group across all six instead of splitting GAC off
into its own. Any future native multiclass quantifier added to this
registry should be checked the same way (does it actually need the
wrapper to avoid crashing?) but wrapped regardless, for this reason.

**Note (deep classifiers + `build_all_quantifiers`, item 13)**: naively
running `build_all_quantifiers(..., cv=10)` with `LITETimeClassifier` as
the base classifier multiplies into ~596 real ensemble trainings (see
`ecg_quantification.quantifiers._cached_classifier`'s module docstring
for the full breakdown: 6 native CV quantifiers x 11 fits, 10 OvR
quantifiers x 5 classes x 11 fits). This is because `quack`'s CV
splitting (`StratifiedKFold(shuffle=False)`, deterministic given
`(X, y, cv)`) and `OneVsRestQuantifier`'s per-class relabeling (`y == c`,
a pure function of `(y, c)`) mean many quantifiers end up independently
refitting the base classifier on *exactly* the same training subset.
`CachedFitClassifier` / `cached_classifier_factory(base_factory,
cache_id)` memoize `.fit()` by a content hash of `(X, y)`, collapsing
this to ~66 real trainings (verified empirically end-to-end on a small
3-class synthetic problem: 12 real fits vs. 104 naive, matching the
predicted formula exactly -- see
`tests/test_cached_classifier_lite_time_integration.py`).

Practical recipe for running `build_all_quantifiers` with LITETime:

```python
from ecg_quantification.quantifiers import (
  LITETimeClassifier, build_all_quantifiers, cached_classifier_factory,
)

lite_time_factory = lambda: LITETimeClassifier(n_classifiers=5, n_epochs=1500, random_state=0)
classifier_factory = cached_classifier_factory(lite_time_factory, cache_id='some-unique-run-id')

registry = build_all_quantifiers(
  'LITETime', classifier_factory, cv=10,
  n_jobs=None,  # required -- see below
)
```

Two things matter for this to actually work as intended:

- `n_jobs=None` (sequential): the fit cache is a module-level, in-process
  dict. `joblib`'s default `"loky"` backend dispatches CV folds/OvR
  members to separate worker *processes*, each with its own independent
  copy of the cache -- with `n_jobs > 1` cache hits across processes
  simply don't happen (not a correctness bug, just lost savings). This
  also happens to be the right choice for a deep model regardless of
  caching: TensorFlow already parallelizes internally per `.fit()` call,
  and spawning several worker processes each loading their own
  TensorFlow runtime competes for the same CPU/GPU rather than helping.
- All six native multiclass quantifiers now join the same shared fit
  group (see the "string labels" note above): `LabelEncodedQuantifier`
  standardizes every one of them onto the same integer-encoded `y`
  internally, so their `(X, y)` content hashes are byte-identical to
  each other, not just their row partitions.

Not solved by this: `ExperimentCheckpoint` (Fase 0) still cannot persist
a *fitted* LITETime-based quantifier at all (aeon's `"cant_pickle": True`
tag on a fitted `LITETimeClassifier` makes `joblib.dump` unreliable) --
this is the same gap already disclosed in `_lite_time`'s own module
docstring, and remains open; `run_multiclass_experiment.py`'s
checkpointed pipeline is not yet wired to accept LITETime as a
`--classifiers` choice.