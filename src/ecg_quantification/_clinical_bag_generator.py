from typing import Generator
import numpy as np
from sklearn.utils import check_random_state
from quack.bag_generator.base import BaseBagGenerator


class ClinicalPrevalenceBagGenerator(BaseBagGenerator):
  """Simulates clinically-relevant prevalence shifts for ECG beat labels
  (e.g. V, A, R, L vs a base/residual label), sampling each label's
  prevalence independently from ranges tied to real diagnostic risk
  strata rather than the full simplex.

  For each bag, every label present in `clinical_ranges` gets a target
  prevalence drawn uniformly from one of its clinically-defined bands
  (optionally weighted by `zone_weights` to oversample gray/high-risk
  zones). Remaining probability mass is assigned to the residual/base
  class (default `'N'`). If the sampled label prevalences exceed 1.0,
  they are proportionally rescaled to fit the simplex.

  Parameters
  ----------
  clinical_ranges : dict[str, list[tuple[float, float]]]
    Maps a class label to a list of (low, high) prevalence bands
    (fractions, not percentages) representing its clinical risk zones,
    e.g. `{'V': [(0.0, 0.05), (0.05, 0.10), (0.10, 0.40)]}`. See
    `ecg_quantification.CLINICAL_RANGES` for the default V/A/L/R protocol.
  n_bags : int, default = 100
    Number of bags to generate.
  bag_size : int, default = None
    Number of instances per bag. If None, defaults to `len(y)`.
  zone_weights : dict[str, list[float]], default = None
    Sampling weights over each label's bands (same length as its band
    list), letting gray/high-risk zones be oversampled relative to a
    naive uniform pick across zones. If None, zones are picked uniformly.
  base_label : default = 'N'
    Label absorbing the residual prevalence mass (1 - sum of sampled
    label prevalences). Must be present in `y`.
  with_replacement : bool, default = True
    Whether instances are drawn with replacement from each label's pool.
    Forced to True whenever the requested count exceeds the pool size.
  random_state : int, RandomState instance or None, default = None
    Controls reproducibility of zone/prevalence sampling and resampling.

  Attributes
  ----------
  classes_ : ndarray of shape (n_classes,)
    Distinct labels found in `y` the last time `generate` was called.
  sampled_prevalences_ : ndarray of shape (n_bags, n_classes)
    Realized prevalence of each class per generated bag, aligned to
    `classes_` (sorted ascending, standard `np.unique` order).
  sampled_zones_ : dict[str, ndarray of shape (n_bags,)]
    For each label in `clinical_ranges`, the index of the risk zone
    sampled for each bag (useful for stratified downstream analysis).

  References
  ----------
  Clinical thresholds for PVC/PAC-induced cardiomyopathy, LBBB/RBBB risk
  stratification, per the protocol described in the project documentation.

  Examples
  --------
  >>> from ecg_quantification import ClinicalPrevalenceBagGenerator, CLINICAL_RANGES, ZONE_WEIGHTS
  >>> generator = ClinicalPrevalenceBagGenerator(
  ...   clinical_ranges=CLINICAL_RANGES, zone_weights=ZONE_WEIGHTS,
  ...   n_bags=200, bag_size=500, random_state=0,
  ... )
  >>> bags = generator.to_list(X, y)
  """

  def __init__(self,
               clinical_ranges: dict,
               n_bags: int = 100,
               bag_size: int = None,
               zone_weights: dict = None,
               base_label='N',
               with_replacement: bool = True,
               random_state=None):
    super().__init__(n_bags=n_bags, bag_size=bag_size, random_state=random_state)
    self.clinical_ranges = clinical_ranges
    self.zone_weights = zone_weights
    self.base_label = base_label
    self.with_replacement = with_replacement

  def _sample_label_prevalence(self, label: str, rng) -> tuple[float, int]:
    """Draws a single (prevalence, zone_index) pair for one label."""
    bands = self.clinical_ranges[label]
    n_zones = len(bands)

    weights = None
    if self.zone_weights is not None and label in self.zone_weights:
      weights = np.asarray(self.zone_weights[label], dtype=float)
      if weights.shape[0] != n_zones:
        raise ValueError(
          f"zone_weights['{label}'] must have length {n_zones} (one per band), "
          f"got {weights.shape[0]}."
        )
      weights = weights / weights.sum()

    zone_idx = rng.choice(n_zones, p=weights)
    lo, hi = bands[zone_idx]
    prevalence = rng.uniform(lo, hi)
    return prevalence, zone_idx

  def _sample_bag_prevalence(self, rng) -> tuple[np.ndarray, dict]:
    """Samples the full per-class prevalence vector (aligned to
    `self.classes_`) for a single bag, plus the zone indices drawn."""
    n_classes = len(self.classes_)
    prevalence_vec = np.zeros(n_classes)
    zones = {}

    label_prevs = {}
    for label in self.clinical_ranges:
      if label not in self._class_to_idx:
        continue
      p, zone_idx = self._sample_label_prevalence(label, rng)
      label_prevs[label] = p
      zones[label] = zone_idx

    total_labeled = sum(label_prevs.values())

    # rescale proportionally if sampled label prevalences overshoot 1.0,
    # leaving no room for the base/residual class
    if total_labeled > 1.0:
      scale = 1.0 / total_labeled
      label_prevs = {k: v * scale for k, v in label_prevs.items()}
      total_labeled = 1.0

    for label, p in label_prevs.items():
      prevalence_vec[self._class_to_idx[label]] = p

    residual = 1.0 - total_labeled
    if self.base_label in self._class_to_idx:
      prevalence_vec[self._class_to_idx[self.base_label]] += residual
    elif residual > 1e-12:
      # no explicit base class present: redistribute residual proportionally
      # among the unspecified classes, or uniformly if none were specified
      unspecified = [c for c in self.classes_ if c not in label_prevs]
      if unspecified:
        share = residual / len(unspecified)
        for c in unspecified:
          prevalence_vec[self._class_to_idx[c]] += share
      else:
        prevalence_vec += residual / n_classes

    return prevalence_vec, zones

  @staticmethod
  def _prevalence_to_counts(prevalence: np.ndarray, bag_size: int) -> np.ndarray:
    """Largest-remainder rounding of a prevalence vector into integer
    counts summing exactly to `bag_size`."""
    raw_counts = prevalence * bag_size
    counts = np.floor(raw_counts).astype(int)

    remainder = bag_size - counts.sum()
    if remainder > 0:
      fractional_parts = raw_counts - counts
      top_indices = np.argsort(fractional_parts)[::-1][:remainder]
      counts[top_indices] += 1
    elif remainder < 0:
      fractional_parts = raw_counts - counts
      drop_indices = np.argsort(fractional_parts)[:(-remainder)]
      counts[drop_indices] = np.maximum(counts[drop_indices] - 1, 0)

    return counts

  def generate(self, X: np.ndarray, y: np.ndarray) -> Generator[tuple, None, None]:
    X, y = self._validate(X, y)
    rng = check_random_state(self.random_state)

    self.classes_ = np.unique(y)
    self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
    n_classes = len(self.classes_)
    class_pools = self._group_indices_by_class(y, self.classes_)

    missing = [lbl for lbl in self.clinical_ranges if lbl not in self._class_to_idx]
    if missing:
      raise ValueError(f"Labels {missing} from clinical_ranges are not present in y.")

    bag_size = self.bag_size if self.bag_size is not None else len(y)
    self.sampled_prevalences_ = np.zeros((self.n_bags, n_classes))
    self.sampled_zones_ = {label: np.zeros(self.n_bags, dtype=int) for label in self.clinical_ranges}

    for i in range(self.n_bags):
      prevalence_vec, zones = self._sample_bag_prevalence(rng)
      counts = self._prevalence_to_counts(prevalence_vec, bag_size)
      self.sampled_prevalences_[i] = counts / bag_size

      for label, zone_idx in zones.items():
        self.sampled_zones_[label][i] = zone_idx

      bag_indices = []
      for c_idx, count in enumerate(counts):
        if count == 0:
          continue
        pool = class_pools[self.classes_[c_idx]]
        if pool.size == 0:
          raise ValueError(
            f"Class '{self.classes_[c_idx]}' has zero instances in the provided "
            "data but a positive count was sampled for it; cannot build this bag."
          )
        replace = self.with_replacement or count > len(pool)
        bag_indices.append(rng.choice(pool, size=count, replace=replace))

      bag_indices = np.concatenate(bag_indices)
      rng.shuffle(bag_indices)

      yield X[bag_indices], y[bag_indices]