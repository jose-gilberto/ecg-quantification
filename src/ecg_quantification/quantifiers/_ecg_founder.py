"""Adapter exposing ECGFounder (a foundation model for ECG signals) as a
`quack`-compatible classifier.

`ECGFounder` is not a scikit-learn estimator (no `get_params`/`set_params`/
`clone` support), so it cannot be passed directly as the `classifier=`
argument of any `quack` quantifier. This module wraps it via
`quack.utils.SklearnClassifierWrapper`, which stores only a *recipe* to
build/load the model rather than a live instance, satisfying `clone()`'s
requirements throughout `quack` (per-CV-fold refitting in
`BaseCalibratedQuantifier`, per-member refitting in ensembles, etc.).

NOTE: The actual ECGFounder loading/inference calls below are stubbed out
(`NotImplementedError`) pending integration with the real model API/
checkpoint. Fill in `_load_ecg_founder`, `_ecg_founder_predict`, and
`_ecg_founder_predict_proba` once that API is available; the surrounding
wrapper machinery (freezing, classes_ handling, predict_proba gating) is
already wired to `quack`'s conventions and should not need to change.
"""
import numpy as np
from quack.utils.wrappers import SklearnClassifierWrapper


def _load_ecg_founder(checkpoint_path: str = None, freeze_backbone: bool = True, **kwargs):
  """Loads (or lazily initializes) an ECGFounder model instance.

  Parameters
  ----------
  checkpoint_path : str, default = None
    Path or identifier for the pretrained ECGFounder weights. If None,
    the model's own default pretrained checkpoint is expected to load.
  freeze_backbone : bool, default = True
    Whether the foundation model's backbone should be frozen, exposing
    only a lightweight classification head to be fit per quantifier
    call. Matters for `BaseCalibratedQuantifier` subclasses, which clone
    and refit this classifier once per CV fold: freezing keeps that
    inexpensive.
  **kwargs
    Forwarded to the underlying ECGFounder constructor/loader.

  Returns
  -------
  model : object
    A loaded ECGFounder model instance, ready for `.fit`/`.predict`-style
    calls via the adapter functions below.

  Raises
  ------
  NotImplementedError
    Placeholder until the real ECGFounder loading call is integrated.
  """
  raise NotImplementedError(
    "ECGFounder loading is not yet integrated. Implement this function to "
    "load the pretrained checkpoint (and optionally freeze its backbone) "
    "once the model API is available."
  )


def _ecg_founder_fit(model, X: np.ndarray, y: np.ndarray, **fit_params) -> None:
  """Fits (or fine-tunes) ECGFounder's classification head on `(X, y)`.

  Parameters
  ----------
  model : object
    The model instance returned by `_load_ecg_founder`.
  X : array-like of shape (n_samples, window_size)
    Raw or preprocessed ECG beat segments (see
    `ecg_quantification.preprocessing.ECGPreprocessor`).
  y : array-like of shape (n_samples,)
    Beat labels.
  **fit_params
    Extra keyword arguments forwarded to the underlying training call.

  Raises
  ------
  NotImplementedError
    Placeholder until the real ECGFounder fit/fine-tune call is integrated.
  """
  raise NotImplementedError(
    "ECGFounder fitting is not yet integrated. Implement this function to "
    "fine-tune (or fit a lightweight head on top of) the loaded model."
  )


def _ecg_founder_predict(model, X: np.ndarray) -> np.ndarray:
  """Hard-label predictions from a fitted ECGFounder model.

  Raises
  ------
  NotImplementedError
    Placeholder until the real ECGFounder inference call is integrated.
  """
  raise NotImplementedError(
    "ECGFounder prediction is not yet integrated. Implement this function "
    "to return hard class labels for X."
  )


def _ecg_founder_predict_proba(model, X: np.ndarray) -> np.ndarray:
  """Class-probability predictions from a fitted ECGFounder model.

  Raises
  ------
  NotImplementedError
    Placeholder until the real ECGFounder inference call is integrated.
  """
  raise NotImplementedError(
    "ECGFounder predict_proba is not yet integrated. Implement this "
    "function to return an (n_samples, n_classes) probability matrix."
  )


def ECGFounderClassifier(checkpoint_path: str = None,
                         freeze_backbone: bool = True,
                         supports_predict_proba: bool = True,
                         **model_kwargs) -> SklearnClassifierWrapper:
  """Builds a `quack`-compatible classifier wrapping ECGFounder.

  Returns a `SklearnClassifierWrapper` configured with ECGFounder's
  loading/fit/predict adapters, so it can be passed directly as the
  `classifier=` argument of any `quack` quantifier (`CC`, `ACC`, `GAC`,
  `EM`, ...) or `ecg_quantification.quantifiers.OneVsRestQuantifier`.

  Parameters
  ----------
  checkpoint_path : str, default = None
    Forwarded to `_load_ecg_founder`.
  freeze_backbone : bool, default = True
    Forwarded to `_load_ecg_founder`.
  supports_predict_proba : bool, default = True
    Whether to expose `predict_proba` on the resulting wrapper. Set to
    False if the specific ECGFounder configuration in use only exposes
    hard labels, so quantifiers requiring `predict_proba` (`PCC`,
    `PACC`, `GPAC`, `EM`, ...) fail fast at construction time instead of
    deep inside cross-validation.
  **model_kwargs
    Forwarded to `_load_ecg_founder` (e.g. device, batch size).

  Returns
  -------
  classifier : SklearnClassifierWrapper
    A fresh, unfitted wrapper. Every `.fit()` call (including the ones
    triggered internally per CV fold by `BaseCalibratedQuantifier`, or
    per One-vs-Rest member by `OneVsRestQuantifier`) builds a brand-new
    ECGFounder instance via `_load_ecg_founder`, matching how every
    other `quack` classifier is refit from scratch on each call.

  Examples
  --------
  >>> from quack.quantifiers import CC
  >>> from ecg_quantification.quantifiers import ECGFounderClassifier
  >>> classifier = ECGFounderClassifier(checkpoint_path='path/to/checkpoint.pt')
  >>> quantifier = CC(classifier=classifier)
  >>> quantifier.fit(X_train, y_train)  # doctest: +SKIP
  """
  return SklearnClassifierWrapper(
    model_factory=lambda: _load_ecg_founder(checkpoint_path, freeze_backbone, **model_kwargs),
    fit_fn=_ecg_founder_fit,
    predict_fn=_ecg_founder_predict,
    predict_proba_fn=_ecg_founder_predict_proba if supports_predict_proba else None,
    supports_predict_proba=supports_predict_proba,
  )