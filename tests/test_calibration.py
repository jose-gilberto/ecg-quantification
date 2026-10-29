"""Tests for the calibration study (Fase 5, T34): calibration metrics
(`ecg_quantification.metrics`), precomputed-output calibration
(`ecg_quantification.calibration`) and reliability plots."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from quack.quantifiers import CC
from ecg_quantification.calibration import (
  PrecomputedScores, calibrate_from_folds, calibrate_precomputed, cross_fit_calibration,
  fold_scores, make_splits, outputs_from_quantifier,
)
from ecg_quantification.metrics import (
  brier_score, classwise_calibration_error, expected_calibration_error, reliability_curve,
)
from ecg_quantification.quantifiers import CachedFitClassifier, LabelEncodedQuantifier, clear_fit_cache
from ecg_quantification.visualization import (
  plot_calibration_summary, plot_classwise_reliability, plot_reliability_diagram,
)

CLASSES = np.array(["A", "N", "V"])


def _probabilities(n=4000, seed=0, sharpen=1.0):
  """Labels drawn from Dirichlet probabilities (perfectly calibrated);
  `sharpen > 1` makes the reported probabilities overconfident."""
  rng = np.random.default_rng(seed)
  true = rng.dirichlet([1.0, 1.0, 1.0], size=n)
  y = np.array([rng.choice(CLASSES, p=row) for row in true])
  reported = true ** sharpen
  return y, reported / reported.sum(axis=1, keepdims=True)


def _data(n=600, seed=0, n_patients=12):
  rng = np.random.default_rng(seed)
  y = CLASSES[rng.integers(0, 3, n)]
  X = rng.normal(size=(n, 6))
  for i, label in enumerate(CLASSES):
    X[y == label, i] += 1.5
  groups = np.arange(n) % n_patients
  return X, y, groups


class TestMetrics:

  def test_hand_computed_top_label_ece(self):
    # 5 bins of width 0.2: confidences 0.9, 0.9 (1 of 2 correct) fall in
    # [0.8, 1], 0.6, 0.6 (both correct) in [0.6, 0.8)
    # -> (2/4)|0.5 - 0.9| + (2/4)|1.0 - 0.6| = 0.4
    classes = np.array([0, 1])
    proba = np.array([[0.9, 0.1], [0.9, 0.1], [0.4, 0.6], [0.4, 0.6]])
    y = np.array([0, 1, 1, 1])
    assert expected_calibration_error(y, proba, classes, n_bins=5) == pytest.approx(0.4)

  def test_perfect_predictions(self):
    y = np.array(["A", "N", "V", "N"])
    proba = (y[:, None] == CLASSES).astype(float)
    assert brier_score(y, proba, CLASSES) == 0.0
    assert expected_calibration_error(y, proba, CLASSES) == 0.0
    assert classwise_calibration_error(y, proba, CLASSES) == 0.0

  def test_calibrated_vs_overconfident(self):
    y, calibrated = _probabilities(sharpen=1.0)
    _, overconfident = _probabilities(sharpen=3.0)
    assert expected_calibration_error(y, calibrated, CLASSES) < 0.03
    assert expected_calibration_error(y, overconfident, CLASSES) > 0.1
    assert classwise_calibration_error(y, overconfident, CLASSES) > classwise_calibration_error(y, calibrated, CLASSES)

  def test_brier_matches_sklearn_multiclass(self):
    y, proba = _probabilities(n=500, seed=3)
    expected = brier_score_loss(y, proba, labels=CLASSES, scale_by_half=True)
    assert brier_score(y, proba, CLASSES, scale_by_half=True) == pytest.approx(expected)
    assert brier_score(y, proba, CLASSES) == pytest.approx(2 * expected)

  def test_unsorted_classes_are_respected(self):
    y, proba = _probabilities(n=500, seed=4)
    order = [2, 0, 1]
    assert brier_score(y, proba[:, order], CLASSES[order]) == pytest.approx(brier_score(y, proba, CLASSES))

  def test_reliability_curve_and_strategies(self):
    y, proba = _probabilities(n=2000, seed=5, sharpen=2.0)
    curve = reliability_curve(y, proba, CLASSES, n_bins=10)
    assert len(curve["bin_edges"]) == 11 and curve["bin_count"].sum() == 2000
    per_class = reliability_curve(y, proba, CLASSES, n_bins=10, class_label="N")
    filled = per_class["bin_count"] > 0
    assert np.all((per_class["bin_accuracy"][filled] >= 0) & (per_class["bin_accuracy"][filled] <= 1))
    quantile = reliability_curve(y, proba, CLASSES, n_bins=10, strategy="quantile")
    assert quantile["bin_count"].max() - quantile["bin_count"].min() <= 2  # equal mass
    per_class_values = classwise_calibration_error(y, proba, CLASSES, average=False)
    assert set(per_class_values) == set(CLASSES)

  def test_equal_mass_bins_with_saturated_confidences(self):
    """Every confidence exactly 1.0 (a very sure model) must not break the
    equal-mass binning (regression: all quantile edges coincided)."""
    y = np.array(["A", "N", "V", "N"])
    proba = (y[:, None] == CLASSES).astype(float)
    assert expected_calibration_error(y, proba, CLASSES, strategy="quantile") == 0.0
    y_mixed, proba_mixed = _probabilities(n=300, seed=10)
    saturated = np.vstack([proba, proba_mixed])
    labels = np.concatenate([y, y_mixed])
    assert 0 <= expected_calibration_error(labels, saturated, CLASSES, strategy="quantile") <= 1

  def test_input_validation(self):
    y, proba = _probabilities(n=50)
    with pytest.raises(ValueError, match="shape"):
      brier_score(y, proba[:, :2], CLASSES)
    with pytest.raises(ValueError, match="not in classes"):
      brier_score(np.where(y == "A", "Z", y), proba, CLASSES)
    with pytest.raises(ValueError, match="strategy"):
      expected_calibration_error(y, proba, CLASSES, strategy="kmeans")
    with pytest.raises(ValueError, match="class_label"):
      reliability_curve(y, proba, CLASSES, class_label="Q")


class TestPrecomputedCalibration:

  def test_precomputed_scores_expose_only_their_response(self):
    proba_model = PrecomputedScores(CLASSES, "predict_proba").fit()
    decision_model = PrecomputedScores(CLASSES, "decision_function").fit()
    assert hasattr(proba_model, "predict_proba") and not hasattr(proba_model, "decision_function")
    assert hasattr(decision_model, "decision_function") and not hasattr(decision_model, "predict_proba")

  @pytest.mark.parametrize("method", ["sigmoid", "isotonic", "temperature"])
  def test_calibration_reduces_overconfidence(self, method):
    y_cal, proba_cal = _probabilities(seed=1, sharpen=3.0)
    y_eval, proba_eval = _probabilities(seed=2, sharpen=3.0)
    calibrated = calibrate_precomputed(proba_cal, y_cal, proba_eval, CLASSES, method)
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-9)
    assert expected_calibration_error(y_eval, calibrated, CLASSES) < 0.5 * expected_calibration_error(
      y_eval, proba_eval, CLASSES)

  @pytest.mark.parametrize("estimator", [LogisticRegression(max_iter=1000),
                                         RandomForestClassifier(n_estimators=30, random_state=0)])
  @pytest.mark.parametrize("method", ["sigmoid", "isotonic", "temperature"])
  def test_fold_reuse_equals_calibrated_classifier_cv(self, estimator, method):
    """One model per fold, reused for every method == sklearn's
    CalibratedClassifierCV(cv=same splits, ensemble=True) run per method,
    for decision_function (LR) and predict_proba-only (RF) estimators."""
    X, y, groups = _data()
    X_apply = np.random.default_rng(9).normal(size=(150, 6))
    splits = make_splits(y, 4, groups=groups, random_state=0)
    folds = fold_scores(estimator, X, y, splits, X_apply)
    ours = calibrate_from_folds(folds, y, CLASSES, method)
    reference = CalibratedClassifierCV(estimator, method=method, cv=splits, ensemble=True).fit(X, y)
    np.testing.assert_allclose(ours, reference.predict_proba(X_apply), atol=1e-10)

  def test_cross_fitting_never_calibrates_a_patient_with_its_own_beats(self):
    _, _, groups = _data()
    y, _ = _probabilities(n=len(groups), seed=6)
    for fit_idx, apply_idx in make_splits(y, 4, groups=groups, random_state=0):
      assert not set(groups[fit_idx]) & set(groups[apply_idx])

  def test_cross_fit_calibration_reports_missing_classes(self):
    y, proba = _probabilities(n=600, seed=7, sharpen=2.0)
    groups = np.arange(len(y)) % 6
    y = y.copy()
    y[groups == 0] = "V"   # 'A' only appears in patients 1-5 ...
    y[(groups != 0) & (y == "V")] = "N"  # ... and 'V' only in patient 0
    calibrated, missing = cross_fit_calibration(proba, y, CLASSES, "sigmoid", groups=groups, n_splits=3)
    assert calibrated.shape == proba.shape
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-9)
    assert ["V"] in missing  # the fold holding patient 0 calibrates V without positives

  @pytest.mark.parametrize("method", ["sigmoid", "isotonic", "temperature"])
  def test_missing_class_in_calibration_data_matches_sklearn(self, method):
    """A held-out part without some class: same result as sklearn's own
    CalibratedClassifierCV (that class's map fit without positives)."""
    X, y, _ = _data(seed=11)
    X_apply = np.random.default_rng(12).normal(size=(80, 6))
    train = np.flatnonzero(np.arange(len(y)) % 2 == 0)
    held_out = np.flatnonzero((np.arange(len(y)) % 2 == 1) & (y != "V"))
    splits = [(train, held_out)]
    estimator = LogisticRegression(max_iter=1000)
    ours = calibrate_from_folds(fold_scores(estimator, X, y, splits, X_apply), y, CLASSES, method)
    reference = CalibratedClassifierCV(estimator, method=method, cv=splits, ensemble=True).fit(X, y)
    np.testing.assert_allclose(ours, reference.predict_proba(X_apply), atol=1e-10)


class TestOutputsFromQuantifier:

  def test_label_encoded_cc_maps_codes_back_to_labels(self):
    X, y, _ = _data()
    quantifier = LabelEncodedQuantifier(CC(classifier=LogisticRegression(max_iter=1000))).fit(X, y)
    outputs = outputs_from_quantifier(quantifier, X[:50])
    direct = LogisticRegression(max_iter=1000).fit(X, y)
    assert list(outputs["classes"]) == list(CLASSES)
    np.testing.assert_allclose(outputs["proba"], direct.predict_proba(X[:50]), atol=1e-6)
    assert outputs["response"] == "decision_function"
    np.testing.assert_allclose(outputs["scores"], direct.decision_function(X[:50]), atol=1e-6)

  def test_unwraps_the_fit_cache_but_not_other_estimator_attributes(self):
    """A CachedFitClassifier (deep classifiers) is unwrapped; a random
    forest -- which also has an `estimator_` (its template tree) -- is not."""
    X, y, _ = _data()
    clear_fit_cache("calibration-test")
    cached = CC(classifier=CachedFitClassifier(LogisticRegression(max_iter=1000), cache_id="calibration-test"))
    forest = CC(classifier=RandomForestClassifier(n_estimators=20, random_state=0))
    for quantifier in (cached.fit(X, y), forest.fit(X, y)):
      outputs = outputs_from_quantifier(quantifier, X[:20])
      np.testing.assert_allclose(outputs["proba"].sum(axis=1), 1.0, atol=1e-9)
    clear_fit_cache("calibration-test")


class TestPlots:

  def test_figures_build(self):
    import pandas as pd
    y, proba = _probabilities(n=800, seed=8, sharpen=2.0)
    series = {"uncalibrated": proba, "temperature": calibrate_precomputed(proba, y, proba, CLASSES, "temperature")}
    assert len(plot_reliability_diagram(y, series, CLASSES).axes) == 2
    assert len(plot_reliability_diagram(y, series, CLASSES, class_label="V", show_counts=False).axes) == 1
    assert len(plot_classwise_reliability(y, series, CLASSES).axes) == 3
    results = pd.DataFrame([
      {"classifier": "LR", "protocol": "uncalibrated", "method": "none", "ece": 0.1, "classwise_ece": 0.05, "brier": 0.3},
      {"classifier": "LR", "protocol": "post-hoc", "method": "sigmoid", "ece": 0.02, "classwise_ece": 0.02, "brier": 0.28},
    ])
    assert len(plot_calibration_summary(results).axes) == 3
