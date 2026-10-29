"""Probability-calibration metrics for multiclass classifiers (Fase 5).

Probability-based quantifiers (PCC, PACC, EMQ, ...) average or reweight
the base classifier's `predict_proba`, so their bias depends on how well
those probabilities are calibrated -- Reviewer 2's point (6). These
metrics measure that directly:

- `brier_score`: the multiclass Brier score (Brier, 1950), mean over
  samples of `sum_k (p_k - 1[y = k])^2`, in [0, 2]. A proper scoring rule:
  it mixes calibration and discrimination.
- `expected_calibration_error`: top-label ECE (Guo et al., 2017) -- bins
  the predicted class's confidence and averages |accuracy - confidence|
  over bins, weighted by bin size.
- `classwise_calibration_error`: classwise ECE (Kull et al., 2019) --
  the ECE of every class's probability column against that class's
  indicator, averaged over classes. Closer to what quantification needs:
  PCC's estimate for class k is the mean of column k, which is unbiased
  exactly when column k is calibrated.
- `reliability_curve`: the per-bin statistics behind a reliability
  diagram, top-label or for one class.

All functions take `proba` with columns ordered as `classes` and labels
`y_true` from `classes`.
"""
import numpy as np


BIN_STRATEGIES = ("uniform", "quantile")


def _validate(y_true, proba, classes):
  classes = np.asarray(classes)
  proba = np.asarray(proba, dtype=np.float64)
  y_true = np.asarray(y_true)
  if proba.ndim != 2 or proba.shape[1] != len(classes):
    raise ValueError(
      f"proba must have shape (n_samples, {len(classes)}) to match classes, got {proba.shape}."
    )
  if len(y_true) != proba.shape[0]:
    raise ValueError(f"y_true has {len(y_true)} samples but proba has {proba.shape[0]}.")
  if len(y_true) == 0:
    raise ValueError("Calibration metrics need at least one sample.")
  unknown = np.setdiff1d(np.unique(y_true), classes)
  if len(unknown):
    raise ValueError(f"Labels {unknown.tolist()} in y_true are not in classes {classes.tolist()}.")
  if np.any(proba < -1e-9) or np.any(proba > 1 + 1e-9):
    raise ValueError("proba values must lie in [0, 1].")
  return y_true, proba, classes


def _one_hot(y_true, classes) -> np.ndarray:
  order = np.argsort(classes)
  index = order[np.searchsorted(classes[order], y_true)]
  onehot = np.zeros((len(y_true), len(classes)))
  onehot[np.arange(len(y_true)), index] = 1.0
  return onehot


def _bin_ids(values: np.ndarray, n_bins: int, strategy: str) -> tuple[np.ndarray, np.ndarray]:
  """Bin index of every value in [0, 1], and the bin edges."""
  if strategy not in BIN_STRATEGIES:
    raise ValueError(f"strategy must be one of {BIN_STRATEGIES}, got {strategy!r}.")
  if n_bins < 1:
    raise ValueError("n_bins must be >= 1.")
  if strategy == "uniform":
    edges = np.linspace(0.0, 1.0, n_bins + 1)
  else:
    # equal-mass bins spanning [0, 1]; duplicate edges (many identical
    # values, e.g. confidences saturated at 1.0) collapse into fewer bins
    inner = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    edges = np.unique(np.concatenate([[0.0], inner, [1.0]]))
  # right-closed on the last bin so a value of exactly 1.0 is counted
  ids = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
  return ids, edges


def _binned_calibration(confidence: np.ndarray, correct: np.ndarray, n_bins: int, strategy: str) -> dict:
  ids, edges = _bin_ids(confidence, n_bins, strategy)
  n_slots = len(edges) - 1
  counts = np.bincount(ids, minlength=n_slots).astype(float)
  sum_conf = np.bincount(ids, weights=confidence, minlength=n_slots)
  sum_correct = np.bincount(ids, weights=correct, minlength=n_slots)
  with np.errstate(invalid="ignore", divide="ignore"):
    mean_conf = np.where(counts > 0, sum_conf / counts, np.nan)
    accuracy = np.where(counts > 0, sum_correct / counts, np.nan)
  return {
    "bin_edges": edges,
    "bin_confidence": mean_conf,
    "bin_accuracy": accuracy,
    "bin_count": counts,
  }


def _ece_from_bins(bins: dict) -> float:
  counts = bins["bin_count"]
  filled = counts > 0
  gaps = np.abs(bins["bin_accuracy"][filled] - bins["bin_confidence"][filled])
  return float(np.sum(counts[filled] * gaps) / np.sum(counts))


def brier_score(y_true, proba, classes, scale_by_half: bool = False) -> float:
  """Multiclass Brier score: mean over samples of `sum_k (p_k - 1[y=k])^2`.

  Parameters
  ----------
  y_true : array-like of shape (n_samples,)
  proba : array-like of shape (n_samples, n_classes)
    Columns ordered as `classes`.
  classes : array-like of shape (n_classes,)
  scale_by_half : bool, default = False
    Divide by 2, mapping the score to [0, 1] (`sklearn.metrics.
    brier_score_loss(..., scale_by_half=True)` for multiclass). The
    default keeps Brier's original [0, 2] definition.

  Returns
  -------
  score : float
    Lower is better.
  """
  y_true, proba, classes = _validate(y_true, proba, classes)
  score = float(np.mean(np.sum((proba - _one_hot(y_true, classes)) ** 2, axis=1)))
  return score / 2 if scale_by_half else score


def expected_calibration_error(y_true, proba, classes, n_bins: int = 15, strategy: str = "uniform") -> float:
  """Top-label expected calibration error (Guo et al., 2017).

  Bins every sample by the probability of its *predicted* class
  (confidence) and returns the bin-size-weighted mean of
  |accuracy - mean confidence| over non-empty bins.

  Parameters
  ----------
  y_true, proba, classes
    See `brier_score`.
  n_bins : int, default = 15
  strategy : {"uniform", "quantile"}, default = "uniform"
    Equal-width bins on [0, 1], or equal-mass bins (quantiles of the
    confidences; fewer bins if quantiles coincide).

  Returns
  -------
  ece : float
    In [0, 1]; lower is better.
  """
  return _ece_from_bins(reliability_curve(y_true, proba, classes, n_bins=n_bins, strategy=strategy))


def classwise_calibration_error(y_true, proba, classes, n_bins: int = 15,
                                strategy: str = "uniform", average: bool = True):
  """Classwise expected calibration error (Kull et al., 2019).

  For every class k, the ECE of column k (binned) against the indicator
  1[y = k]; averaged over classes unless `average=False`.

  Returns
  -------
  ece : float, or dict {class: float} when `average=False`
  """
  y_true, proba, classes = _validate(y_true, proba, classes)
  per_class = {
    label: _ece_from_bins(
      reliability_curve(y_true, proba, classes, n_bins=n_bins, strategy=strategy, class_label=label)
    )
    for label in classes
  }
  if average:
    return float(np.mean(list(per_class.values())))
  return per_class


def reliability_curve(y_true, proba, classes, n_bins: int = 15, strategy: str = "uniform",
                      class_label=None) -> dict:
  """Per-bin statistics for a reliability diagram.

  Parameters
  ----------
  y_true, proba, classes, n_bins, strategy
    See `expected_calibration_error`.
  class_label : optional
    `None` (default): top-label -- confidence of the predicted class vs.
    whether the prediction is correct. A class label: that class's
    probability column vs. its indicator (one-vs-rest).

  Returns
  -------
  curve : dict
    `bin_edges` (n_bins + 1,), and per bin `bin_confidence` (mean
    predicted probability), `bin_accuracy` (observed frequency) -- NaN
    for empty bins -- and `bin_count`.
  """
  y_true, proba, classes = _validate(y_true, proba, classes)
  if class_label is None:
    predicted = np.argmax(proba, axis=1)
    confidence = proba[np.arange(len(proba)), predicted]
    correct = (classes[predicted] == y_true).astype(float)
  else:
    matches = np.flatnonzero(classes == class_label)
    if len(matches) != 1:
      raise ValueError(f"class_label {class_label!r} is not one of {classes.tolist()}.")
    confidence = proba[:, matches[0]]
    correct = (y_true == class_label).astype(float)
  return _binned_calibration(np.clip(confidence, 0.0, 1.0), correct, n_bins, strategy)
