"""Tests for `ecg_quantification.metrics.ZoneCrossingError`."""
import numpy as np
import pytest

from ecg_quantification import CLINICAL_RANGES
from ecg_quantification.metrics import ZoneCrossingError
from ecg_quantification.metrics._zone_crossing import _find_zone


CLASSES = np.array(['A', 'L', 'N', 'R', 'V'])  # np.unique-sorted order


@pytest.mark.filterwarnings("ignore:'p_pred' does not sum to 1.0")
def test_docstring_example_matches_documented_value():
  # exact scenario from ZoneCrossingError's own docstring: V crosses
  # exactly one zone boundary (gray -> high), A and L stay put, R is
  # skipped (single band). Locks in the documented behavior as a
  # regression guard. p_pred deliberately sums to 0.975, not 1.0 -- an
  # artifact of the source docstring's own example, kept as-is here on
  # purpose (changing it would no longer "lock in" the documented value).
  zce = ZoneCrossingError(clinical_ranges=CLINICAL_RANGES, classes=CLASSES)
  p_true = np.array([0.03, 0.005, 0.75, 0.05, 0.165])
  p_pred = np.array([0.03, 0.005, 0.80, 0.05, 0.09])

  assert zce(p_true, p_pred) == pytest.approx(0.167, abs=1e-3)


def test_perfect_prediction_scores_zero():
  zce = ZoneCrossingError(clinical_ranges=CLINICAL_RANGES, classes=CLASSES)
  p = np.array([0.03, 0.005, 0.75, 0.05, 0.165])

  assert zce(p, p) == 0.0


def test_single_band_labels_are_skipped():
  # 'R' has only one band in CLINICAL_RANGES -- any true/pred pair for
  # it must never contribute to the score, however far apart the values are.
  # The remaining mass is parked on 'A' (index 0) purely to keep these
  # valid probability vectors -- 'A' isn't in `ranges`, so its value is
  # irrelevant to the score either way.
  ranges = {'R': CLINICAL_RANGES['R']}
  zce = ZoneCrossingError(clinical_ranges=ranges, classes=CLASSES)
  p_true = np.array([0.99, 0.0, 0.0, 0.01, 0.0])
  p_pred = np.array([0.01, 0.0, 0.0, 0.99, 0.0])

  assert zce(p_true, p_pred) == 0.0


def test_missing_class_labels_are_ignored():
  # a clinical_ranges entry for a label absent from `classes` must not
  # raise and must simply not contribute to the score
  ranges = {**CLINICAL_RANGES, 'X': [(0.0, 0.5), (0.5, 1.0)]}
  zce = ZoneCrossingError(clinical_ranges=ranges, classes=CLASSES)
  p = np.array([0.03, 0.005, 0.75, 0.05, 0.165])

  assert zce(p, p) == 0.0  # identical vectors -> zero regardless


def test_returns_zero_when_no_class_has_clinical_ranges():
  # degenerate case, e.g. a purely binary N/not-N problem with no
  # clinically-defined zones at all
  classes = np.array(['N', 'not_N'])
  zce = ZoneCrossingError(clinical_ranges={}, classes=classes)

  assert zce(np.array([0.6, 0.4]), np.array([0.1, 0.9])) == 0.0


def test_full_range_miss_costs_the_maximum_for_that_class():
  # V has 3 bands (K=3): true in the lowest band, predicted in the
  # highest band -> |0 - 2| / (3 - 1) == 1.0 for V alone. Remaining mass
  # is parked on 'N' (not in `ranges`, so irrelevant to the score) purely
  # to keep these valid probability vectors.
  ranges = {'V': CLINICAL_RANGES['V']}
  zce = ZoneCrossingError(clinical_ranges=ranges, classes=CLASSES)
  n_idx = np.searchsorted(CLASSES, 'N')
  v_idx = np.searchsorted(CLASSES, 'V')

  p_true = np.zeros(5)
  p_pred = np.zeros(5)
  p_true[v_idx] = 0.01   # low-risk band: [0.00, 0.05)
  p_pred[v_idx] = 0.90   # high-risk band: [0.10, 0.40)
  p_true[n_idx] = 1.0 - p_true[v_idx]
  p_pred[n_idx] = 1.0 - p_pred[v_idx]

  assert zce(p_true, p_pred) == pytest.approx(1.0)


@pytest.mark.parametrize("value,expected_zone", [
  (0.0, 0),      # lower edge of the first band
  (0.049, 0),    # inside the first band
  (0.05, 1),     # exactly on the boundary -> falls into the next band
  (0.35, 2),     # inside the last band
  (1.0, 2),      # exactly the upper edge of the last band -> inclusive
  (2.0, 2),      # out-of-range above -> falls back to the last zone
])
def test_find_zone_boundaries(value, expected_zone):
  bands = CLINICAL_RANGES['V']  # [(0.00, 0.05), (0.05, 0.10), (0.10, 0.40)]
  # note: 1.0 and 2.0 are out of V's actual last-band upper bound (0.40);
  # exercised here purely to lock in the fallback behavior for values
  # above the last band, using the same 3-band shape.
  assert _find_zone(value, bands) == expected_zone


def test_find_zone_below_first_band_falls_back_to_zone_zero():
  bands = [(0.10, 0.40), (0.40, 1.00)]
  assert _find_zone(-0.05, bands) == 0


def test_zone_report_matches_compute():
  zce = ZoneCrossingError(clinical_ranges=CLINICAL_RANGES, classes=CLASSES)
  p_true = np.array([0.03, 0.005, 0.75, 0.05, 0.165])
  p_pred = np.array([0.03, 0.005, 0.80, 0.05, 0.09])

  report = zce.zone_report(p_true, p_pred)

  assert set(report.keys()) == {'V', 'A', 'L'}  # R excluded: single band
  assert report['V'] == {'true_zone': 2, 'pred_zone': 1, 'crossed': True}
  assert report['A']['crossed'] is False
  assert report['L']['crossed'] is False


def test_metric_metadata():
  zce = ZoneCrossingError(clinical_ranges=CLINICAL_RANGES, classes=CLASSES)
  assert zce.name == "Zone-Crossing Error"
  assert zce.lower_is_better is True
