"""Fit-cache wrapper collapsing redundant classifier retraining across
quantifiers/OvR members that share identical CV folds (item 13, Fase 3).

Motivation
----------
`build_all_quantifiers` builds ~16 quantifier recipes per base classifier:
6 native multiclass (CC, PCC, GAC, GPAC, FM, EMQ) plus 10 One-vs-Rest
wrapped binary ones (ACC, PACC, X, Max, T50, MedianSweep, HDy, DyS,
FormanMM, CDE). Every CV-based quantifier (`BaseCalibratedQuantifier`
subclasses: GAC, GPAC, FM, EMQ, and every OvR member) independently
clones and refits the base classifier once per CV fold *plus* one final
full-data refit (`cv + 1` fits), and every OvR-wrapped quantifier repeats
that whole CV procedure once per class (`n_classes` independent binary
sub-problems). For a cheap sklearn classifier this is negligible; for
`LITETimeClassifier` (minutes per `.fit()` call, training a whole
ensemble of neural networks) naively running
`build_all_quantifiers(..., cv=10)` multiplies into an intractable number
of real trainings on the 5-class ECG problem:

  4 native CV quantifiers (GAC, GPAC, FM, EMQ) x 11 fits    =  44
  10 OvR quantifiers x 5 classes x 11 fits                  = 550
  CC, PCC: 1 full-data fit each, no CV                      =   2
  ---------------------------------------------------------------
  naive total                                               = 596 real LITETime trainings

(counting each `.fit()` call on the base classifier once; with
`n_classifiers > 1` every one of those trains a whole ensemble, not just
one network.)

`CachedFitClassifier` collapses this using a fact about how `quack`
builds its folds: `sklearn.model_selection.check_cv(cv, y,
classifier=True)` (used internally by `BaseCalibratedQuantifier.fit`)
returns a `StratifiedKFold` with `shuffle=False` -- quack never passes a
`random_state`/`shuffle=True` for the int-`cv` case -- so **the fold
split is a deterministic, pure function of `(the per-sample class
partition, cv)`**, not of which quantifier is asking for it, and not of
the *labels themselves* (`StratifiedKFold` internally re-encodes `y` via
`np.unique(y, return_inverse=True)` before stratifying -- exactly the
convention `LabelEncodedQuantifier` already mirrors, see that module).
Concretely:

- GPAC, FM and EMQ (all fit directly on the original string `y`, no
  wrapper) built with the same `cv` on the same `(X, y)` produce
  bit-identical `(train_idx, test_idx)` pairs, so their per-fold *and*
  final-refit training subsets are identical across all three -- 11 fits
  shared, not 33.
- GAC does *not* join that group: it is fit through
  `LabelEncodedQuantifier` (see that module), which encodes `y` to
  integer codes before delegating, specifically to work around a
  separate `quack` limitation. `_fingerprint` hashes `y`'s actual bytes
  (not a partition-only encoding), by design -- see "On not sharing
  across differently-labeled data" below -- so GAC's int-coded training
  subsets are a genuine cache miss against GPAC/FM/EMQ's string-labeled
  ones, even though the row *partition* is identical. GAC gets its own
  11-fit group instead of joining the other three's.
- Every OvR quantifier's per-class binary relabeling (`y == c).astype(int)`)
  is a pure function of `(y, c)`, independent of which quantifier (ACC
  vs. PACC vs. HDy, ...) is doing the relabeling -- so the same 5 classes
  x 11 fits are shared across all 10 OvR quantifier types, not repeated
  per type.
- CC/PCC's single full-data fit (on the original string `y`) reuses the
  GPAC/FM/EMQ group's final full-data refit, if any.

  GPAC + FM + EMQ (string y)      -> 11 shared fits
  GAC (int-coded y, own group)    -> 11 shared fits
  10 OvR quantifiers x 5 classes  -> 55 shared fits (11 per class)
  CC, PCC -> reuse the GPAC/FM/EMQ group's full-data refit (0 extra, usually)
  ---------------------------------------------------------------------
  cached total                                            ~=  77 real LITETime trainings

On not sharing across differently-labeled data: `_fingerprint` hashes
`y`'s raw bytes/dtype, not a partition-only encoding (e.g. not
`np.unique(y, return_inverse=True)[1]`), even though that would let GAC's
int-coded subsets collide with GPAC/FM/EMQ's string-labeled ones (same
row partition either way). This is intentional, not a missed
optimization: a cache *hit* also hands back the cached model's
`classes_` as-is, and `LabelEncodedQuantifier` depends on the classifier
it wraps reporting `classes_` in its own `[0, ..., n_classes-1]` int
encoding to decode predictions back to the original labels correctly.
Unifying the two groups would hand GAC back a classifier whose
`classes_` are the original string labels instead -- silently breaking
that decoding. Treating differently-labeled data as a genuine cache miss
costs a modest 11 extra fits; getting this wrong would be a silent
correctness bug, not a performance one.

This is not an approximation: a cache *hit* returns the exact classifier
instance already fit on that exact training subset, so every quantifier
still gets a genuinely fitted model for its actual fold, not a
substitute. The only thing skipped is redundantly re-fitting the same
model on the same data purely because a different quantifier asked for
it. For a non-deterministic classifier (`LITETimeClassifier` carries
aeon's `"non_deterministic": True` tag -- different random weight
init/dropout per `.fit()` call unless `random_state` is fixed), this
also means every quantifier sharing a fold gets the exact same fitted
model instance/predictions for it, rather than each independently
re-rolling initialization -- which is the point: "freeze" one classifier
per training subset and reuse it, rather than pretending each
quantifier's copy is meaningfully different.

Process-based parallelism (`n_jobs` > 1, the default `parallel_backend=
"loky"` in every quack quantifier and `OneVsRestQuantifier`) is NOT
compatible with this cache: joblib's "loky" backend dispatches fold/
per-class jobs to separate worker *processes*, each with its own Python
interpreter and therefore its own independent copy of the module-level
cache below -- writes in one worker are invisible to another, silently
reducing (not breaking) cache effectiveness rather than causing
incorrect results (a cache miss just refits normally, correctly). Always
combine `CachedFitClassifier` with `n_jobs=None` (sequential,
single-process) whenever the wrapped classifier is a deep model; this is
the right choice anyway for `LITETimeClassifier`, since TensorFlow
already parallelizes internally per `.fit()` call, and spawning several
worker *processes* that each load their own TensorFlow runtime competes
for the same CPU/GPU resources rather than genuinely accelerating
anything.

Scope note (what this does *not* solve): fitting the full registry this
way still produces ~66 genuinely fitted deep ensembles, each potentially
minutes to train -- a real, still-substantial cost, just no longer a
~9x redundant one. It also does not address `ExperimentCheckpoint`
(Fase 0) being unable to persist a *fitted* LITETime-based quantifier at
all (aeon's `"cant_pickle": True` tag on a fitted `LITETimeClassifier`
makes `joblib.dump` unreliable) -- this is the same, already-disclosed
gap noted in `_lite_time`'s own module docstring, and remains a separate,
not-yet-scheduled task; wiring `run_multiclass_experiment.py`'s
checkpointed pipeline through LITETime is deliberately left out of this
module.
"""
import hashlib
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted


_FIT_CACHE: dict = {}


def clear_fit_cache(cache_id: str = None) -> None:
  """Drops cached fitted models.

  Parameters
  ----------
  cache_id : str, default = None
    Clears only entries under this scope when given; clears everything
    otherwise. Call this between independent experiments that reuse the
    same `cache_id` (e.g. re-running the same script from scratch),
    since a stale entry from an earlier run would otherwise be returned
    instead of retraining on genuinely different data.
  """
  if cache_id is None:
    _FIT_CACHE.clear()
    return
  for key in [k for k in _FIT_CACHE if k[0] == cache_id]:
    del _FIT_CACHE[key]


def fit_cache_size(cache_id: str = None) -> int:
  """Number of distinct fitted models currently cached, optionally
  scoped to one `cache_id` -- mainly for tests/diagnostics, to verify
  the collapse described in this module's docstring actually happened
  rather than just asserting on wall-clock time."""
  if cache_id is None:
    return len(_FIT_CACHE)
  return sum(1 for k in _FIT_CACHE if k[0] == cache_id)


def _fingerprint(X, y) -> str:
  """Content hash of a training subset, stable across independently
  constructed but numerically-identical `(X, y)` arrays -- e.g. the same
  rows sliced out by two different quantifiers' identical CV folds."""
  X = np.asarray(X)
  y = np.asarray(y)
  hasher = hashlib.sha1()
  hasher.update(repr(X.dtype).encode())
  hasher.update(str(X.shape).encode())
  hasher.update(np.ascontiguousarray(X).tobytes())
  hasher.update(repr(y.dtype).encode())
  hasher.update(str(y.shape).encode())
  hasher.update(np.ascontiguousarray(y).tobytes())
  return hasher.hexdigest()


class CachedFitClassifier(BaseEstimator):
  """Wraps `base_estimator`, memoizing `.fit()` by a content hash of
  `(X, y)` under a shared `cache_id` scope -- see this module's
  docstring for why identical CV folds/OvR relabelings across different
  quantifiers make this a safe, exact (not approximate) way to skip
  redundant retraining of an expensive classifier like
  `LITETimeClassifier`.

  Parameters
  ----------
  base_estimator : sklearn-compatible estimator, default = None
    A fresh, unfitted classifier instance to wrap (e.g. an
    `ecg_quantification.quantifiers.LITETimeClassifier(...)`). Not a
    factory -- `clone(base_estimator)` is called on a genuine cache
    miss, matching how quack itself clones the base classifier, so this
    stays a drop-in replacement for a plain classifier instance. Use
    `cached_classifier_factory` below to turn a factory into one of
    these per call, for use as `build_all_quantifiers`'s
    `classifier_factory` argument.
  cache_id : str, default = "default"
    Identifies this cache's scope so multiple *independent* experiments
    (different bag configurations, different hyperparameters, different
    scripts) sharing the same Python process don't collide. Two
    `CachedFitClassifier`s built with the same `cache_id` intentionally
    share fitted models for identical `(X, y)` subsets -- give every
    logically distinct experiment its own `cache_id`, and call
    `clear_fit_cache(cache_id)` before reusing one (e.g. rerunning a
    script from scratch in the same interpreter/notebook session).

  Attributes
  ----------
  estimator_ : object
    The fitted underlying estimator for this training subset -- either
    freshly fit (cache miss) or reused from an earlier identical fit
    (cache hit).
  classes_ : ndarray of shape (n_classes,)
    Forwarded from `estimator_` (or `np.unique(y)` if the estimator
    doesn't expose it).
  from_cache_ : bool
    Whether this particular `.fit()` call was served from the cache
    (`True`) or actually trained a new model (`False`) -- mainly for
    tests/diagnostics.

  Notes
  -----
  Stores the shared cache in a module-level dict (`_FIT_CACHE`) rather
  than as an instance attribute, precisely so it survives
  `sklearn.base.clone()`: `clone()` calls `copy.deepcopy` on any
  constructor parameter that isn't itself an estimator (see
  `sklearn.base._clone_parametrized`), so a plain dict passed in as a
  parameter would be deep-copied on every fold/member clone -- silently
  defeating the whole point of caching across quack's per-fold
  `clone(base_classifier)` calls. `cache_id` (a plain string) is the
  only thing that needs to survive `clone()`/`deepcopy` intact, and
  strings copy by value safely -- the actual cache storage lives outside
  sklearn's parameter/clone model entirely.

  `predict_proba` is bound as an instance attribute in `__init__` only
  when `base_estimator` itself exposes one (checked via plain
  `hasattr`, which also picks up an *instance*-bound `predict_proba`
  such as `SklearnClassifierWrapper`'s, not just a class-level one) --
  never defined as a class method that would always make
  `hasattr(wrapper, "predict_proba")` return `True` regardless of
  whether the wrapped estimator actually supports it. This matters:
  `BaseCalibratedQuantifier.fit`'s fail-fast check and several
  quantifiers (`PCC`, `PACC`, `GPAC`, `EM`, ...) branch on
  `hasattr(classifier, "predict_proba")` before running (or even
  starting) any cross-validation.
  """

  def __init__(self, base_estimator=None, cache_id: str = "default"):
    self.base_estimator = base_estimator
    self.cache_id = cache_id
    if base_estimator is not None and hasattr(base_estimator, "predict_proba"):
      self.predict_proba = self._predict_proba_impl

  def fit(self, X, y) -> "CachedFitClassifier":
    """Fits (or reuses an identical earlier fit of) `base_estimator` on
    `(X, y)`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
      Training data, in whatever form `base_estimator` expects.
    y : array-like of shape (n_samples,)
      Training labels.

    Returns
    -------
    self : object
      Returns the fitted wrapper instance itself.
    """
    key = (self.cache_id, _fingerprint(X, y))
    cached = _FIT_CACHE.get(key)

    if cached is not None:
      self.estimator_ = cached
      self.from_cache_ = True
    else:
      fitted = clone(self.base_estimator)
      fitted.fit(X, y)
      _FIT_CACHE[key] = fitted
      self.estimator_ = fitted
      self.from_cache_ = False

    estimator_classes = getattr(self.estimator_, "classes_", None)
    self.classes_ = np.asarray(estimator_classes) if estimator_classes is not None else np.unique(y)
    return self

  def predict(self, X) -> np.ndarray:
    """Delegates to the fitted (or cache-reused) estimator's `.predict()`."""
    check_is_fitted(self, "estimator_")
    return np.asarray(self.estimator_.predict(X))

  def _predict_proba_impl(self, X) -> np.ndarray:
    """Delegates to the fitted (or cache-reused) estimator's
    `.predict_proba()`; only bound as `self.predict_proba` when
    `base_estimator` supports it (see `__init__`)."""
    check_is_fitted(self, "estimator_")
    return np.asarray(self.estimator_.predict_proba(X))


def cached_classifier_factory(base_factory, cache_id: str):
  """Wraps a zero-argument classifier factory so every classifier it
  builds shares one `CachedFitClassifier` fit cache.

  This is the intended entry point for item 13: pass the result of this
  function directly as the `classifier_factory` argument of
  `build_all_quantifiers`/`build_multiclass_quantifiers`/
  `build_binary_quantifiers_for_ovr` in place of a plain factory, with
  every other call unchanged. No registry code needs to know caching is
  happening -- each quantifier still calls the factory expecting a fresh,
  unfitted, `clone()`-compatible classifier, and gets one; the caching
  only takes effect the moment `.fit()` is actually called on it, keyed
  by the training data it's asked to fit.

  Parameters
  ----------
  base_factory : Callable[[], BaseEstimator]
    Zero-argument callable building a fresh, unfitted classifier
    instance (e.g. `lambda: LITETimeClassifier(n_classifiers=5,
    n_epochs=1500, random_state=0)`).
  cache_id : str
    Forwarded to every `CachedFitClassifier` this factory builds -- see
    `CachedFitClassifier`'s docstring. Use one `cache_id` per logically
    independent experiment run.

  Returns
  -------
  factory : Callable[[], CachedFitClassifier]
    A factory suitable for `classifier_factory=...` in the registry
    builder functions.

  Examples
  --------
  >>> from ecg_quantification.quantifiers import (
  ...   LITETimeClassifier, build_all_quantifiers, cached_classifier_factory,
  ... )
  >>> lite_time_factory = lambda: LITETimeClassifier(
  ...   n_classifiers=5, n_epochs=1500, random_state=0,
  ... )
  >>> registry = build_all_quantifiers(
  ...   'LITETime',
  ...   cached_classifier_factory(lite_time_factory, cache_id='lite_time-run1'),
  ...   cv=10, n_jobs=None,  # n_jobs=None: see this module's docstring
  ... )  # doctest: +SKIP
  """
  def factory():
    return CachedFitClassifier(base_estimator=base_factory(), cache_id=cache_id)
  return factory
