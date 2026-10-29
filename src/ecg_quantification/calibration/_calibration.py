"""Calibrating classifiers' probabilities without refitting them (Fase 5).

Everything here works on *precomputed* classifier outputs, so the deep
models (LITETime, ECGFounder) are calibrated from outputs they already
produced -- no retraining -- with the very same sklearn calibrators
(`CalibratedClassifierCV`, methods sigmoid / isotonic / temperature) used
for the classical ones:

- `PrecomputedScores`: an identity "classifier" whose `decision_function`
  (or `predict_proba`) returns its input. Inside `CalibratedClassifierCV`
  it lets sklearn fit its calibrators on scores computed elsewhere.
- `response_values`: the output sklearn's calibration would use --
  `decision_function` when the estimator has one, else `predict_proba`
  (sklearn's own order) -- so precomputed calibration matches
  `CalibratedClassifierCV` on the live estimator exactly.
- `calibrate_precomputed`: fit a calibrator on a calibration set's
  scores and labels, apply it to other scores.
- `cross_fit_calibration`: post-hoc calibration when the only outputs
  available are on the evaluation pool itself -- k-fold cross-fitting
  (grouped by patient), so no beat is calibrated by a map fit on its own
  patient.
- `fold_scores` + `calibrate_from_folds`: `CalibratedClassifierCV(
  estimator, cv=splits, ensemble=True)` for estimators that *can* be
  refit, split in two so each fold's model is trained once and its
  outputs reused for every method (identical results to running
  `CalibratedClassifierCV` once per method -- see the tests).
- `outputs_from_quantifier`: the base classifier's outputs out of an
  already fitted quack quantifier (e.g. `LITETime-CC` from an experiment
  checkpoint), mapped back to the original labels.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_array, check_is_fitted


CALIBRATION_METHODS = ("sigmoid", "isotonic", "temperature")
RESPONSES = ("decision_function", "predict_proba")


class PrecomputedScores(ClassifierMixin, BaseEstimator):
  """Identity classifier over precomputed outputs: `X` *is* the output
  matrix of some classifier (columns ordered as `classes`).

  Parameters
  ----------
  classes : array-like of shape (n_classes,)
  response : {"predict_proba", "decision_function"}, default = "predict_proba"
    Which method `X` came from. Only that method is exposed, so sklearn's
    calibration treats the scores exactly as it would the original
    estimator's (it prefers `decision_function` when present).
  """

  def __init__(self, classes=None, response: str = "predict_proba"):
    self.classes = classes
    self.response = response

  def fit(self, X=None, y=None):
    if self.classes is None:
      raise ValueError("PrecomputedScores needs `classes`.")
    if self.response not in RESPONSES:
      raise ValueError(f"response must be one of {RESPONSES}, got {self.response!r}.")
    self.classes_ = np.asarray(self.classes)
    return self

  def _scores(self, X) -> np.ndarray:
    check_is_fitted(self, "classes_")
    X = check_array(X, dtype=np.float64, ensure_2d=False)
    if X.ndim == 2 and X.shape[1] == 1:
      X = X.ravel()
    expected_columns = 1 if (self.response == "decision_function" and len(self.classes_) == 2) else len(self.classes_)
    columns = 1 if X.ndim == 1 else X.shape[1]
    if columns != expected_columns:
      raise ValueError(f"Expected {expected_columns} score column(s), got {columns}.")
    return X

  def _is_proba(self):
    return self.response == "predict_proba"

  def _is_decision(self):
    return self.response == "decision_function"

  @available_if(_is_proba)
  def predict_proba(self, X) -> np.ndarray:
    return self._scores(X)

  @available_if(_is_decision)
  def decision_function(self, X) -> np.ndarray:
    return self._scores(X)

  def predict(self, X) -> np.ndarray:
    scores = self._scores(X)
    if scores.ndim == 1:
      return self.classes_[(scores > 0).astype(int)]
    return self.classes_[np.argmax(scores, axis=1)]


def _check_method(method: str) -> None:
  if method not in CALIBRATION_METHODS:
    raise ValueError(f"method must be one of {CALIBRATION_METHODS}, got {method!r}.")


def response_values(estimator, X) -> tuple[np.ndarray, str]:
  """The output `CalibratedClassifierCV` would calibrate: `decision_function`
  if `estimator` has one, else `predict_proba`. Returns `(scores, name)`."""
  for response in RESPONSES:
    if hasattr(estimator, response):
      return np.asarray(getattr(estimator, response)(X), dtype=np.float64), response
  raise TypeError(f"{type(estimator).__name__} has neither decision_function nor predict_proba.")


def fit_precomputed_calibrator(scores_cal, y_cal, classes, method: str,
                               response: str = "predict_proba") -> CalibratedClassifierCV:
  """Fits a sklearn calibrator on precomputed outputs `scores_cal` (from
  `response`, columns ordered as `classes`). Apply it with
  `.predict_proba(scores)`; its columns follow `sorted(classes)`.

  Runs `CalibratedClassifierCV`'s own cross-validation path with a single
  split whose "training" part is one placeholder row per class (the
  identity model ignores it) and whose held-out part is the calibration
  data. That makes the fitted calibrator exactly the one
  `CalibratedClassifierCV(estimator, cv=..., ensemble=True)` builds for a
  fold -- including when the calibration data lacks a class: that class's
  map is then fit without positives, as sklearn does natively. (The
  `FrozenEstimator` route instead fails outright on a missing class.)
  """
  _check_method(method)
  classes = np.asarray(classes)
  scores_cal = np.asarray(scores_cal, dtype=np.float64)
  y_cal = np.asarray(y_cal)
  n_placeholders = len(classes)
  placeholder = np.repeat(scores_cal[:1], n_placeholders, axis=0)
  X = np.concatenate([placeholder, scores_cal])
  y = np.concatenate([classes, y_cal])
  split = [(np.arange(n_placeholders), np.arange(n_placeholders, len(X)))]
  calibrator = CalibratedClassifierCV(
    PrecomputedScores(classes, response), method=method, cv=split, ensemble=True,
  )
  return calibrator.fit(X, y)


def calibrate_precomputed(scores_cal, y_cal, scores_apply, classes, method: str,
                          response: str = "predict_proba") -> np.ndarray:
  """Calibrated probabilities for `scores_apply`, from a calibrator fit on
  `(scores_cal, y_cal)`.

  Returns
  -------
  calibrated : ndarray of shape (n_apply, n_classes)
    Columns ordered as `classes`; rows sum to 1.
  """
  calibrator = fit_precomputed_calibrator(scores_cal, y_cal, classes, method, response)
  proba = calibrator.predict_proba(np.asarray(scores_apply, dtype=np.float64))
  return align_proba(proba, calibrator.classes_, classes)


def make_splits(y, n_splits: int, groups=None, random_state: int = 0) -> list:
  """Stratified k-fold splits, grouped (e.g. by patient) when `groups` is
  given -- no group on both sides of a split."""
  y = np.asarray(y)
  placeholder = np.zeros((len(y), 1))
  if groups is None:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(placeholder, y))
  splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
  return list(splitter.split(placeholder, y, groups=np.asarray(groups)))


def cross_fit_calibration(scores, y, classes, method: str, groups=None, n_splits: int = 5,
                          random_state: int = 0, response: str = "predict_proba") -> tuple[np.ndarray, list]:
  """Post-hoc calibration of `scores` using only `(scores, y)` themselves.

  Each fold is calibrated by a calibrator fit on the other folds, so
  every beat is calibrated out-of-sample -- and, with `groups` (patient
  ids), out-of-patient.

  Returns
  -------
  calibrated : ndarray of shape (n_samples, n_classes)
  missing : list of list
    Per fold, the classes absent from the data its calibrator was fit on.
    With few patients per class, a patient-grouped fold can hold all of
    a class's patients; that class's calibration map is then learned
    without positives (reported so it can be read into the results).
  """
  scores = np.asarray(scores, dtype=np.float64)
  y = np.asarray(y)
  calibrated = np.empty((len(y), len(classes)))
  missing = []
  for fit_idx, apply_idx in make_splits(y, n_splits, groups, random_state):
    missing.append(sorted(set(np.asarray(classes).tolist()) - set(np.unique(y[fit_idx]).tolist())))
    calibrated[apply_idx] = calibrate_precomputed(
      scores[fit_idx], y[fit_idx], scores[apply_idx], classes, method, response,
    )
  return calibrated, missing


def align_proba(proba, estimator_classes, classes) -> np.ndarray:
  """Reorders/embeds probability columns `estimator_classes` into
  `classes` order; classes the estimator never saw get probability 0."""
  proba = np.asarray(proba, dtype=np.float64)
  aligned = np.zeros((proba.shape[0], len(classes)))
  position = {label: j for j, label in enumerate(np.asarray(classes).tolist())}
  for column, label in enumerate(np.asarray(estimator_classes).tolist()):
    aligned[:, position[label]] = proba[:, column]
  return aligned


def fold_scores(estimator, X, y, splits, X_apply) -> list[dict]:
  """For each `(train_idx, cal_idx)` split, fits a clone of `estimator` on
  the train part and keeps its calibration-relevant outputs
  (`response_values`) on the held-out part and on `X_apply`.

  The expensive half of `CalibratedClassifierCV(cv=splits, ensemble=True)`,
  reusable across methods (`calibrate_from_folds`) and cacheable to disk.

  Returns
  -------
  folds : list of dict
    `cal_idx`, `classes` (the fold model's -- a class absent from its
    training part is absent here too), `response`, `cal_scores`,
    `apply_scores`.
  """
  y = np.asarray(y)
  folds = []
  for train_idx, cal_idx in splits:
    model = clone(estimator).fit(X[train_idx], y[train_idx])
    cal_scores, response = response_values(model, X[cal_idx])
    apply_scores, _ = response_values(model, X_apply)
    folds.append({
      "cal_idx": np.asarray(cal_idx),
      "classes": np.asarray(model.classes_),
      "response": response,
      "cal_scores": cal_scores,
      "apply_scores": apply_scores,
    })
  return folds


def calibrate_from_folds(folds: list[dict], y, classes, method: str) -> np.ndarray:
  """The cheap half: per fold, a calibrator fit on the fold model's
  held-out outputs, applied to its `apply_scores`; probabilities averaged
  over folds, as `CalibratedClassifierCV(..., ensemble=True)` does."""
  y = np.asarray(y)
  calibrated = []
  for fold in folds:
    proba = calibrate_precomputed(
      fold["cal_scores"], y[fold["cal_idx"]], fold["apply_scores"],
      fold["classes"], method, fold["response"],
    )
    calibrated.append(align_proba(proba, fold["classes"], classes))
  return np.mean(calibrated, axis=0)


def _unwrap(classifier):
  """The estimator actually producing outputs: a `CachedFitClassifier`
  (deep classifiers' fit cache, item 13) wraps it in `estimator_`. Only
  that wrapper is unwrapped -- other estimators use `estimator_` for
  something else (e.g. a random forest's template tree)."""
  from ecg_quantification.quantifiers._cached_classifier import CachedFitClassifier
  if isinstance(classifier, CachedFitClassifier):
    return classifier.estimator_
  return classifier


def outputs_from_quantifier(quantifier, X) -> dict:
  """Outputs of the base classifier inside a fitted quack quantifier (CC,
  PCC, ... -- anything with a fitted `classifier_`), optionally wrapped in
  `LabelEncodedQuantifier`, mapped back to the original labels.

  Returns
  -------
  outputs : dict
    `classes` (original labels, sorted), `proba` (`predict_proba`,
    columns = `classes`), `scores` and `response` (what sklearn's
    calibration would use, see `response_values`).
  """
  inner = getattr(quantifier, "quantifier_", quantifier)
  classifier = getattr(inner, "classifier_", None)
  if classifier is None or not hasattr(classifier, "predict_proba"):
    raise TypeError(
      f"{type(inner).__name__} has no fitted classifier_ with predict_proba; use a "
      f"classifier-based quantifier such as CC or PCC."
    )
  estimator = _unwrap(classifier)
  classes = np.asarray(quantifier.classes_)
  estimator_classes = np.asarray(estimator.classes_)
  if inner is not quantifier:  # LabelEncodedQuantifier: integer codes -> labels
    estimator_classes = classes[estimator_classes.astype(int)]
  if sorted(estimator_classes.tolist()) != sorted(classes.tolist()):
    raise ValueError(f"Classifier classes {estimator_classes.tolist()} differ from {classes.tolist()}.")
  order = [int(np.flatnonzero(estimator_classes == label)[0]) for label in classes]

  proba = np.asarray(estimator.predict_proba(X), dtype=np.float64)[:, order]
  scores, response = response_values(estimator, X)
  if scores.ndim == 2:
    scores = scores[:, order]
  return {"classes": classes, "proba": proba, "scores": scores, "response": response}
