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

**Update (item 14)**: the `ExperimentCheckpoint`/pickling gap noted here
previously ("cannot persist a fitted LITETime-based quantifier at all")
turned out to be misdiagnosed. Empirically, a fitted Keras model's
weights round-trip through `joblib.dump`/`joblib.load` fine; the actual
(and only) blocker was `_lite_time.py`'s `LITETimeClassifier()` factory
using a `lambda` for `model_factory`, which `pickle`/`joblib` cannot
serialize by reference regardless of what it builds. Fixed by using
`functools.partial` instead -- see `_lite_time.py`'s "NOTE on
checkpointing". `run_multiclass_experiment.py` now accepts `'LITETime'`
as a `--classifiers` choice (always run with `n_jobs=None` and through
`cached_classifier_factory`, per item 13), with dedicated
`--lite-time-n-classifiers`/`--lite-time-n-epochs`/
`--lite-time-batch-size` flags -- see that script's module docstring for
a full example command. Verified end-to-end (fit, checkpoint, reload,
predict, resume-skips-refit) on a small synthetic 5-class problem in
`tests/test_run_multiclass_experiment_lite_time.py`; running it for real
requires the `[deep]` extra and the actual MITBIH splits (not available
in every environment -- e.g. PhysioNet is unreachable from this
sandbox's network egress, so the real Fase 3 item 14 comparison numbers
still need to be produced on a machine with both installed).

**Update (GPU + progress output)**: the `[deep]` extra now installs
`tensorflow[and-cuda]` on Linux (`[deep-cpu]` keeps the CPU-only build),
and `run_multiclass_experiment.py` trains LITETime on the GPU by default
(`--lite-time-device {gpu,cpu}`), failing fast before any fitting if no
GPU is visible. Its output is timestamped and flushed line by line (safe
to `| tee` or watch in tmux), with `[i/N]` counters before/after every
quantifier fit and evaluation. `CachedFitClassifier` logs every real
training ("training #k started/done") on the
`ecg_quantification.fit_cache` logger and keeps counters
(`fit_cache_stats`), so each LITETime quantifier's fit line reports how
many new trainings it cost vs. reused from the cache. In practice, with
`cv=10` and 5 classes, `LITETime-CC` pays 1 training, `LITETime-GAC` the
remaining 10 of the native group, and `LITETime-ACC` (the first OvR
quantifier) all 55 per-class ones -- every other LITETime quantifier
then fits in seconds from the cache.

**Update (XLA/ptxas on GPU)**: on the first real GPU run, Keras' XLA
compilation of LITETime's training step failed with `Autotuner could not
compile any configs for HLO ... __triton_gemm`, preceded by a warning
that XLA was using `ptxas` 12.4 (CUDA <= 12.6.2). The script now sets
`XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` before TensorFlow loads
(XLA uses cuBLAS for those small matrix multiplications instead; an
explicit setting in the environment is respected). Upgrading `ptxas` to
>= 12.6.3 is still recommended -- see the README's `deep` extra section.

**Update (Fase 4, ECGFounder)**: `ECGFounderClassifier` (single-lead
ECGFounder, Net1D vendored from the official repo under MIT) is wired into
`run_multiclass_experiment.py` as `--classifiers ECGFounder`, through the
same fit cache as LITETime but in its **own cache scope**
(`<checkpoint-dir>::<classifier>`): the cache keys on the training data
alone, so two deep classifiers fitting the same subset must never share
one. Two modes: `--ecg-founder-mode linear-probe` (default; frozen
backbone + logistic regression on its 1024-d features) and `finetune`
(full fine-tuning, official notebook hyperparameters, best epoch by
validation loss, per-epoch resumable training state + finished weights
under `<checkpoint-dir>/ecg_founder_trainings/` -- item 16). Inputs are
resampled 360 -> 500 Hz, filtered with ECGFounder's own filter chain and
z-scored per beat; see `_ecg_founder.py`'s module docstring for the two
deliberate deviations from the official preprocessing (1 s beat windows,
not padded to 10 s; filtering per window rather than per record) and the
lead mismatch (pretrained on lead I, MIT-BIH is MLII).

**Update (XLA/ptxas, follow-up)**: disabling Triton GEMM only moved the
failure to the next kernel the old `ptxas` 12.4 could not compile (an
ArgMax reduce fusion, a few trainings in). The script now points XLA at
the pip-installed `ptxas` (`--xla_gpu_cuda_data_dir`, from
`nvidia-cuda-nvcc-cu12`), checks its version (>= 12.6.3) before training
and stops with the fix if it is older; `--no-lite-time-xla` trains without
XLA as an escape hatch.
