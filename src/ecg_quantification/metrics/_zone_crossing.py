"""Zone-Crossing Error (ZCE): a clinically-anchored quantification metric.

Standard quantification metrics (AE, RAE, KLD, ...) treat every unit of
prevalence error as equally costly everywhere in [0, 1]. For arrhythmia
burden, that is not clinically true: a 2-point error inside a low-risk band
(e.g. PVC burden estimated at 3% vs a true 4%, both "low risk") changes no
clinical decision, while the same 2-point error straddling a guideline
threshold (estimated 13% vs a true 15% PVC burden, crossing the suppression-
therapy threshold) can change patient management entirely.

Zone-Crossing Error scores a prediction by whether it lands in the same
clinical risk zone as the truth (see `ecg_quantification.CLINICAL_RANGES`),
independent of how far apart the two prevalence values are numerically.
"""
from typing import Sequence
import numpy as np
from quack.metrics.base import QuantificationMetric


def _find_zone(value: float, bands: Sequence[tuple[float, float]]) -> int:
  """Returns the index of the band in `bands` containing `value`.

  Bands are assumed sorted ascending and contiguous (as in
  `CLINICAL_RANGES`); the last band's upper bound is treated as inclusive
  so a value of exactly 1.0 still resolves to a zone. Values below the
  first band's lower bound (shouldn't happen for prevalences, but guards
  against floating-point noise) fall back to zone 0; values above the
  last band's upper bound fall back to the last zone.
  """
  n_bands = len(bands)
  for idx, (lo, hi) in enumerate(bands):
    is_last = (idx == n_bands - 1)
    if lo <= value < hi or (is_last and np.isclose(value, hi)):
      return idx
  if value < bands[0][0]:
    return 0
  return n_bands - 1


class ZoneCrossingError(QuantificationMetric):
  """Zone-Crossing Error (ZCE): fraction of classes whose predicted
  prevalence falls in a different clinical risk zone than the true
  prevalence, optionally weighted by how many zone boundaries were crossed.

  For each class with clinically-defined risk bands (see
  `clinical_ranges`), the true and predicted prevalence are each mapped to
  a zone index (0 = lowest-risk band, increasing with risk). The per-class
  cost is:

    cost(c) = 0                              if same zone
    cost(c) = |zone_true - zone_pred| / (K-1) otherwise, K = n_bands

  so a miss between adjacent zones (e.g. gray -> low) costs less than a
  miss spanning the full range (e.g. low -> high), and the metric stays
  bounded in [0, 1] regardless of how many bands a class has. Classes
  without an entry in `clinical_ranges` (e.g. the base/normal class `N`)
  are excluded from the average — they carry no zone semantics.

  Parameters
  ----------
  clinical_ranges : dict[str, list[tuple[float, float]]]
    Maps a class label to its ordered, contiguous list of (low, high)
    risk bands (fractions in [0, 1]), e.g.
    `ecg_quantification.CLINICAL_RANGES`.
  classes : Sequence
    The class labels, in the same order as the `p_true`/`p_pred` vectors
    passed to `.compute()` (i.e. matching a quantifier's `classes_`
    attribute, sorted ascending per `np.unique`'s convention).

  Notes
  -----
  Unlike `AbsoluteError`/`KullbackLeiblerDivergence`, this metric requires
  knowing which vector index corresponds to which class label, since
  clinical risk bands are class-specific. `classes` must therefore be
  supplied explicitly and kept aligned with whatever produced `p_true`/
  `p_pred` (typically a quantifier's `.classes_` and a bag's realized
  prevalence vector built via `np.unique(y_bag, return_counts=True)` over
  the same sorted class set).

  If every class lacks a clinical-range entry (e.g. called on a purely
  binary N/not-N problem with no ranges defined), `compute` returns 0.0,
  since there is nothing to evaluate — this mirrors
  `NormalizedAbsoluteError`'s degenerate-case handling rather than raising.

  References
  ----------
  Risk-zone thresholds for PVC (V), PAC (A), LBBB (L), and RBBB (R) beat
  burden as described in the clinical protocol underlying
  `ecg_quantification.CLINICAL_RANGES`.

  Examples
  --------
  >>> import numpy as np
  >>> from ecg_quantification import CLINICAL_RANGES
  >>> from ecg_quantification.metrics import ZoneCrossingError
  >>> classes = np.array(['A', 'L', 'N', 'R', 'V'])
  >>> zce = ZoneCrossingError(clinical_ranges=CLINICAL_RANGES, classes=classes)
  >>> p_true = np.array([0.03, 0.005, 0.75, 0.05, 0.165])   # V in high-risk zone (>=0.10)
  >>> p_pred = np.array([0.03, 0.005, 0.80, 0.05, 0.09])    # V predicted in gray zone (0.05-0.10)
  >>> round(zce(p_true, p_pred), 3)  # A, L correctly zoned (cost 0); V crosses one zone (cost 0.5); R has a single band (skipped)
  0.167
  """

  def __init__(self, clinical_ranges: dict, classes: Sequence):
    super().__init__(name="Zone-Crossing Error", lower_is_better=True)
    self.clinical_ranges = clinical_ranges
    self.classes = np.asarray(classes)
    self._class_to_idx = {c: i for i, c in enumerate(self.classes)}

  def compute(self, p_true: np.ndarray, p_pred: np.ndarray) -> float:
    costs = []

    for label, bands in self.clinical_ranges.items():
      if label not in self._class_to_idx:
        continue

      idx = self._class_to_idx[label]
      n_bands = len(bands)
      if n_bands < 2:
        # a single band spans the whole [0, 1] range: every value is
        # trivially in the same zone, so this class contributes no signal
        continue

      zone_true = _find_zone(float(p_true[idx]), bands)
      zone_pred = _find_zone(float(p_pred[idx]), bands)

      costs.append(abs(zone_true - zone_pred) / (n_bands - 1))

    if not costs:
      return 0.0

    return float(np.mean(costs))

  def zone_report(self, p_true: np.ndarray, p_pred: np.ndarray) -> dict:
    """Returns the per-class (true_zone, predicted_zone, crossed) detail
    backing a single `compute()` call, useful for diagnostics/reporting
    beyond the aggregate score.

    Parameters
    ----------
    p_true : np.ndarray
      True prevalence vector, aligned with `self.classes`.
    p_pred : np.ndarray
      Predicted prevalence vector, aligned with `self.classes`.

    Returns
    -------
    report : dict[str, dict]
      Maps each label with clinical ranges to
      `{'true_zone': int, 'pred_zone': int, 'crossed': bool}`.
    """
    report = {}
    for label, bands in self.clinical_ranges.items():
      if label not in self._class_to_idx or len(bands) < 2:
        continue
      idx = self._class_to_idx[label]
      zone_true = _find_zone(float(p_true[idx]), bands)
      zone_pred = _find_zone(float(p_pred[idx]), bands)
      report[label] = {
        'true_zone': zone_true,
        'pred_zone': zone_pred,
        'crossed': zone_true != zone_pred,
      }
    return report