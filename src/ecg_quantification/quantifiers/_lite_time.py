"""Adapter exposing aeon's `LITETimeClassifier` (LITE / LITEMVTime, a deep
learning ensemble for time series classification) as a `quack`-compatible
classifier.

Unlike `ECGFounderClassifier` (a foundation model with no scikit-learn
`get_params`/`set_params` support, so every call has to be routed through
custom adapter functions), aeon's `LITETimeClassifier` already subclasses
`sklearn.base.BaseEstimator` (via `aeon.base.BaseAeonEstimator`) and
supports `sklearn.base.clone()` out of the box on an *unfitted* instance —
so, in principle, it could be passed directly as `classifier=` to a `quack`
quantifier without any wrapper at all.

It is still wrapped via `SklearnClassifierWrapper` here, for two reasons
specific to how `quack` refits classifiers:

1. Per-fit file isolation. Every `LITETimeClassifier.fit()` call writes at
   least one `.keras` checkpoint file per ensemble member to `file_path`
   while training (via Keras' `ModelCheckpoint` callback) — even when
   `save_best_model=False` (the aeon default), in which case the file is
   given a nanosecond-timestamp name and deleted right after, so
   collisions are negligible. `BaseCalibratedQuantifier` fits one
   classifier per CV fold, dispatched through
   `joblib.Parallel(backend="loky")` — genuine OS-process parallelism.
   The moment `persist=True` is used below (needed for epoch-level
   checkpoint/resume — see the project's `ExperimentCheckpoint`), the
   checkpoint filename becomes deterministic
   (`best_file_name + <ensemble-member index>`), and concurrent folds
   sharing one `file_path` WOULD silently overwrite each other's
   checkpoints. `_load_lite_time` below gives every `.fit()` call (i.e.
   every `model_factory()` build) its own fresh subdirectory under
   `checkpoint_root`, so this stays safe regardless of `n_jobs`.
2. Uniform registry shape. Keeping every non-off-the-shelf-sklearn
   classifier (`ECGFounderClassifier`, `LITETimeClassifier` here) behind
   the same `SklearnClassifierWrapper` recipe lets
   `default_base_classifiers()`-style factory dicts treat them
   identically, and gives a stable `fit_fn`/`predict_fn`/`predict_proba_fn`
   seam to later wire epoch-level checkpoint/resume into, without
   touching `quack` internals or `LITETimeClassifier` itself.

NOTE on checkpointing (important, do not skip): a *fitted*
`LITETimeClassifier` carries aeon's `"cant_pickle": True` tag — its fitted
state holds live Keras/TensorFlow model objects, which `joblib.dump`
(used by `ExperimentCheckpoint.save_fitted`) cannot serialize reliably.
Persisting a fitted `LITETimeClassifier` must go through its own
`save_best_model`/`save_last_model`/`LITETimeClassifier.load_model`
file-based mechanism instead of `ExperimentCheckpoint`'s generic joblib
path. This adapter exposes `persist`/`checkpoint_root` so that
integration can be wired in later without changing this file again;
wiring it into `ExperimentCheckpoint` itself is a separate task.

NOTE on resume support (`resume=True`, item 12 of the Fase 3 plan): by
project decision, this does *not* reimplement epoch-level
state_dict/optimizer/epoch-counter checkpointing. aeon's own
`LITETimeClassifier._fit` trains its `n_classifiers` ensemble members in a
plain sequential Python loop with no `initial_epoch` parameter exposed
anywhere in its public API — there is no supported way to resume a single
member mid-training, only to reconstruct an already-fully-trained
ensemble from disk via the classmethod `LITETimeClassifier.load_model`
(itself documented as "load pre-trained keras models from disk instead of
fitting", i.e. an inference-time loader, not a training-resumption one).
Given that constraint, `resume=True` here implements the minimal, purely
aeon-native mechanism available: `persist=True` already makes every
`IndividualLITEClassifier` member write its best-loss checkpoint to
`{file_path}{best_file_name}{n}.keras` (`n` in `range(n_classifiers)`) via
Keras' own `ModelCheckpoint(save_best_only=True)` callback. `resume=True`
additionally (a) points `file_path` at a *stable* directory
(`checkpoint_root` itself, not a fresh `tempfile.mkdtemp` subdirectory of
it — see point 1 above) so those files survive a process restart, and (b)
has `_lite_time_fit` check, before calling `model.fit(...)`, whether every
expected member checkpoint file already exists in that directory; if so,
it loads the complete ensemble via `LITETimeClassifier.load_model(...)`
and mutates `model` in place instead of retraining, otherwise it fits
normally (which starts accumulating a fresh, complete checkpoint set for
next time, since `save_best_model=True` is already active whenever
`persist=True`).

This means the *only* thing "resumed" is skipping a full retrain once
every ensemble member already finished training in a previous run — there
is no partial resume. If a run is interrupted partway (e.g. after only 2
of 5 members finished), the checkpoint directory holds an incomplete set,
the completeness check fails, and the *whole* ensemble is retrained from
scratch on the next `.fit()` call (the 2 finished members' checkpoint
files are simply overwritten as training proceeds again from member 0).
This is a direct, disclosed consequence of using aeon's own minimal
native support rather than reimplementing its per-member training loop
to add true mid-ensemble/mid-epoch resumption.

`resume=True` requires `persist=True` and a `checkpoint_root` that is
*not* shared concurrently by more than one logical fit unit — the same
caveat as `persist=True` alone (point 1 above), except here the caller
(not this adapter) is responsible for supplying a distinct, stable
directory per independent fit (e.g. one directory per CV fold if fitting
under parallel cross-validation is combined with `resume=True`), since a
stable directory can no longer be auto-isolated per `model_factory()`
call the way `tempfile.mkdtemp` isolates the `persist=True`/`resume=False`
case.

NOTE on multiclass + hard-label calibrated quantifiers (`ACC`, `GAC`):
`LITETimeClassifier` handles multiclass string labels ('N', 'V', 'A', 'L',
'R') natively and directly -- unlike the sklearn base classifiers used
elsewhere in this project, it never needs `OneVsRestQuantifier` to become
multiclass-capable. Resist the temptation to skip OvR and fit `ACC` or
`GAC` directly on raw multiclass string labels with it, though:
`quack.quantifiers.base.BaseCalibratedQuantifier.fit()` allocates its
out-of-fold prediction buffer as a plain `np.zeros(n_samples)` (float64)
whenever `_get_oof_method() == "predict"` (`ACC`'s and `GAC`'s case),
regardless of `y.dtype` -- so assigning back string hard-label predictions
raises `ValueError: could not convert string to float`. This is a `quack`
limitation independent of this adapter (reproduces identically with a
plain `LogisticRegression` fit on string `y`), and it does not affect
`OneVsRestQuantifier` (which always binarizes labels to `int` via
`(y == c).astype(int)` before fitting each member), nor any
`predict_proba`-based calibrated quantifier (`PACC`, `EM`, `GPAC`, ...).

NOTE on input shape: `ECGPreprocessor.process_all_records()` (and
`process_record`) return `X` as a 2D array of shape
`(n_samples, window_size)`. aeon's collection estimators accept this
directly — an `(n_cases, n_timepoints)` 2D array is automatically treated
as a univariate collection and reshaped to `(n_cases, 1, n_timepoints)`
internally (`BaseCollectionEstimator._preprocess_collection`) — so no
manual reshape is needed here.

`tensorflow` (aeon's soft dependency for `LITETimeClassifier`) is
intentionally NOT imported at module load time: importing
`ecg_quantification.quantifiers` must not force a ~600MB TensorFlow
install on users who only need the sklearn-based quantifiers. The import
is deferred to `_load_lite_time`, the first point a `LITETimeClassifier`
is actually built.
"""
import os
import tempfile
import numpy as np
from quack.utils.wrappers import SklearnClassifierWrapper


def _load_lite_time(n_classifiers: int = 5,
                    use_litemv: bool = False,
                    n_epochs: int = 1500,
                    batch_size: int = 64,
                    persist: bool = False,
                    checkpoint_root: str = None,
                    resume: bool = False,
                    random_state: int = None,
                    verbose: bool = False,
                    **kwargs):
  """Builds a fresh, unfitted `LITETimeClassifier` instance.

  Parameters
  ----------
  n_classifiers : int, default = 5
    Number of LITE (or LITEMV) members in the ensemble. Forwarded as-is.
  use_litemv : bool, default = False
    `True` selects LITEMV (aeon's multivariate-oriented variant); ECG
    beat segments here are univariate, so this normally stays `False`.
  n_epochs : int, default = 1500
    Forwarded as-is. aeon's own default (1500) is inherited rather than
    silently overridden, but every quick experiment/test should pass a
    small value explicitly — training 1500 epochs per CV fold is
    expensive.
  batch_size : int, default = 64
    Forwarded as-is.
  persist : bool, default = False
    When `True`, this build's checkpoint files are kept
    (`save_best_model=True`, `save_last_model=True`) in a *fresh*
    subdirectory created under `checkpoint_root` (see the module
    docstring's point 1 on why every build needs its own directory).
    When `False` (default), `LITETimeClassifier` still writes transient
    per-epoch checkpoint files internally (that's how Keras'
    `ModelCheckpoint` + `save_best_only=True` picks the best epoch), but
    deletes them right after `.fit()` returns and nothing is left for
    `ExperimentCheckpoint`/callers to resume from.
  checkpoint_root : str, default = None
    Directory under which a fresh subdirectory is created for this
    build's checkpoint files. Required when `persist=True`; ignored
    otherwise.
  resume : bool, default = False
    When `True` (requires `persist=True`), checkpoint files are written
    directly to `checkpoint_root` itself instead of a fresh
    `tempfile.mkdtemp` subdirectory of it, so they survive a process
    restart and a later build pointed at the same `checkpoint_root` can
    find them (see the module docstring's "NOTE on resume support").
    The caller is then responsible for `checkpoint_root` not being
    shared concurrently by more than one logical fit unit (e.g. give
    each CV fold its own directory if combining this with parallel CV).
    Ignored when `persist=False`.
  random_state : int, default = None
    Forwarded as-is.
  verbose : bool, default = False
    Forwarded as-is.
  **kwargs
    Forwarded to `LITETimeClassifier` (e.g. `n_filters`, `kernel_size`,
    `strides`, `activation`, `optimizer`, `loss`, `metrics`, `callbacks`).

  Returns
  -------
  model : aeon.classification.deep_learning.LITETimeClassifier
    A fresh, unfitted classifier instance.

  Raises
  ------
  ValueError
    If `persist=True` but `checkpoint_root` is not given.
  """
  from aeon.classification.deep_learning import LITETimeClassifier as _AeonLITETimeClassifier

  build_kwargs = dict(
    n_classifiers=n_classifiers,
    use_litemv=use_litemv,
    n_epochs=n_epochs,
    batch_size=batch_size,
    random_state=random_state,
    verbose=verbose,
    **kwargs,
  )

  if persist:
    if checkpoint_root is None:
      raise ValueError(
        "checkpoint_root must be given when persist=True: LITETimeClassifier "
        "writes a checkpoint file per ensemble member on every .fit() call, "
        "and quack refits one classifier per CV fold (in parallel, via "
        "joblib). Every build needs its own subdirectory so concurrent "
        "folds don't overwrite each other's checkpoint files -- see this "
        "module's docstring."
      )
    os.makedirs(checkpoint_root, exist_ok=True)
    fit_dir = checkpoint_root if resume else tempfile.mkdtemp(dir=checkpoint_root, prefix="lite_time_")
    build_kwargs.update(
      file_path=fit_dir + os.sep,
      save_best_model=True,
      save_last_model=True,
    )

  return _AeonLITETimeClassifier(**build_kwargs)


def _checkpoint_member_paths(model) -> list:
  """Expected `.keras` checkpoint file path per ensemble member for an
  unfitted `model` built with `persist=True` (matches aeon's own naming:
  `{file_path}{best_file_name}{n}.keras`, `n` in `range(n_classifiers)` --
  see `IndividualLITEClassifier._fit`/`LITETimeClassifier._fit`).
  """
  return [
    model.file_path + model.best_file_name + str(n) + ".keras"
    for n in range(model.n_classifiers)
  ]


def _checkpoint_is_complete(model) -> bool:
  """Whether every ensemble member's checkpoint file already exists."""
  return all(os.path.isfile(path) for path in _checkpoint_member_paths(model))


def _lite_time_fit(model, X: np.ndarray, y: np.ndarray, **fit_params) -> None:
  """Fits `model` on `(X, y)`.

  Parameters
  ----------
  model : aeon.classification.deep_learning.LITETimeClassifier
    The model instance returned by `_load_lite_time`.
  X : array-like of shape (n_samples, window_size)
    Beat segments (see `ecg_quantification.preprocessing.ECGPreprocessor`).
    Passed through unchanged; aeon reshapes 2D input internally.
  y : array-like of shape (n_samples,)
    Beat labels. String symbols ('N', 'V', 'A', 'L', 'R') work directly —
    verified empirically, aeon label-encodes internally and restores the
    original labels on `classes_`/`predict()`.
  **fit_params
    Not accepted by `LITETimeClassifier.fit(X, y)`; any non-empty
    `fit_params` raises rather than being silently dropped.

  Notes
  -----
  If `model` was built with `persist=True` (so `model.save_best_model` is
  set and `model.file_path` holds a per-ensemble-member checkpoint file
  naming convention -- see the module docstring's "NOTE on resume
  support") and a *complete* checkpoint (one `.keras` file per ensemble
  member) already exists at `model.file_path`, training is skipped
  entirely: `model` is mutated in place via aeon's own
  `LITETimeClassifier.load_model(...)`, matching the `classes_`
  (`np.unique(y)`, computed the same way aeon's own `_fit_setup` would)
  that a fresh `.fit()` on this `y` would have produced. This is always
  safe to check, `persist=True and resume=False` (the parallel-CV-fold
  case) always builds a fresh, empty `tempfile.mkdtemp` directory per
  call, so the completeness check below can never spuriously succeed
  there -- only `resume=True`'s stable directory can hold a leftover
  checkpoint from an earlier run.

  Raises
  ------
  TypeError
    If `fit_params` is non-empty.
  """
  if fit_params:
    raise TypeError(
      f"LITETimeClassifier.fit() does not accept extra fit_params, got: "
      f"{sorted(fit_params)}. Pass model hyperparameters (n_epochs, "
      f"batch_size, ...) to LITETimeClassifier(...) instead."
    )

  if getattr(model, "save_best_model", False) and _checkpoint_is_complete(model):
    from aeon.classification.deep_learning import LITETimeClassifier as _AeonLITETimeClassifier
    classes = np.unique(np.asarray(y))
    loaded = _AeonLITETimeClassifier.load_model(
      model_path=_checkpoint_member_paths(model), classes=classes,
    )
    model.classifiers_ = loaded.classifiers_
    model.n_classifiers = loaded.n_classifiers
    model.classes_ = loaded.classes_
    model.n_classes_ = loaded.n_classes_
    model.is_fitted = True
    return

  model.fit(np.asarray(X, dtype=float), y)


def _lite_time_predict(model, X: np.ndarray) -> np.ndarray:
  """Hard-label predictions from a fitted `LITETimeClassifier`."""
  return model.predict(np.asarray(X, dtype=float))


def _lite_time_predict_proba(model, X: np.ndarray) -> np.ndarray:
  """Class-probability predictions from a fitted `LITETimeClassifier`."""
  return model.predict_proba(np.asarray(X, dtype=float))


def LITETimeClassifier(n_classifiers: int = 5,
                       use_litemv: bool = False,
                       n_epochs: int = 1500,
                       batch_size: int = 64,
                       persist: bool = False,
                       checkpoint_root: str = None,
                       resume: bool = False,
                       random_state: int = None,
                       verbose: bool = False,
                       supports_predict_proba: bool = True,
                       **model_kwargs) -> SklearnClassifierWrapper:
  """Builds a `quack`-compatible classifier wrapping aeon's
  `LITETimeClassifier`.

  Returns a `SklearnClassifierWrapper` configured with LITETime's fit/
  predict adapters, so it can be passed directly as the `classifier=`
  argument of any `quack` quantifier (`CC`, `ACC`, `GAC`, `EM`, ...) or
  `ecg_quantification.quantifiers.OneVsRestQuantifier` — exactly like
  `ECGFounderClassifier`.

  Parameters
  ----------
  n_classifiers, use_litemv, n_epochs, batch_size, random_state, verbose
    Forwarded to `_load_lite_time` (see its docstring).
  persist : bool, default = False
    Forwarded to `_load_lite_time`. `True` keeps each build's best-loss
    checkpoint file per ensemble member on disk instead of deleting it
    after `.fit()`; required for `resume=True` to have anything to load.
  checkpoint_root : str, default = None
    Forwarded to `_load_lite_time`. Required when `persist=True`.
  resume : bool, default = False
    Forwarded to `_load_lite_time` (requires `persist=True`). When a
    complete checkpoint (every ensemble member) already exists at
    `checkpoint_root` from an earlier run, `.fit()` skips retraining
    entirely and loads it instead -- see this module's docstring's "NOTE
    on resume support" for exactly what is (and is not) resumed.
  supports_predict_proba : bool, default = True
    Whether to expose `predict_proba` on the resulting wrapper. Set to
    `False` only if you need quantifiers requiring `predict_proba`
    (`PCC`, `PACC`, `GPAC`, `EM`, ...) to fail fast at construction time
    instead of deep inside cross-validation.
  **model_kwargs
    Forwarded to `_load_lite_time` (e.g. `n_filters`, `kernel_size`,
    `strides`, `activation`, `optimizer`, `loss`, `metrics`, `callbacks`).

  Returns
  -------
  classifier : SklearnClassifierWrapper
    A fresh, unfitted wrapper. Every `.fit()` call (including the ones
    triggered internally per CV fold by `BaseCalibratedQuantifier`, or
    per One-vs-Rest member by `OneVsRestQuantifier`) builds a brand-new
    `LITETimeClassifier` via `_load_lite_time`, matching how every other
    `quack` classifier is refit from scratch on each call.

  Examples
  --------
  >>> from quack.quantifiers import CC
  >>> from ecg_quantification.quantifiers import LITETimeClassifier
  >>> classifier = LITETimeClassifier(n_classifiers=1, n_epochs=5)
  >>> quantifier = CC(classifier=classifier)
  >>> quantifier.fit(X_train, y_train)  # doctest: +SKIP
  """
  return SklearnClassifierWrapper(
    model_factory=lambda: _load_lite_time(
      n_classifiers=n_classifiers,
      use_litemv=use_litemv,
      n_epochs=n_epochs,
      batch_size=batch_size,
      persist=persist,
      checkpoint_root=checkpoint_root,
      resume=resume,
      random_state=random_state,
      verbose=verbose,
      **model_kwargs,
    ),
    fit_fn=_lite_time_fit,
    predict_fn=_lite_time_predict,
    predict_proba_fn=_lite_time_predict_proba if supports_predict_proba else None,
    supports_predict_proba=supports_predict_proba,
  )
