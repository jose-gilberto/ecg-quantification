"""Tests for `ecg_quantification.ClinicalPrevalenceBagGenerator`."""
import numpy as np
import pytest

from ecg_quantification import ClinicalPrevalenceBagGenerator, CLINICAL_RANGES, ZONE_WEIGHTS


def _synthetic_multiclass_dataset(random_state=0):
  # generous per-class pool sizes so any sampled prevalence in
  # CLINICAL_RANGES (with_replacement=True by default, but this also
  # keeps replace=False scenarios realistic) has enough instances to draw from
  rng = np.random.default_rng(random_state)
  sizes = {'N': 2000, 'V': 800, 'A': 800, 'L': 800, 'R': 800}
  X_parts, y_parts = [], []
  for label, size in sizes.items():
    X_parts.append(rng.normal(size=(size, 5)))
    y_parts.append(np.full(size, label))
  X = np.vstack(X_parts)
  y = np.concatenate(y_parts)
  return X, y


@pytest.fixture
def dataset():
  return _synthetic_multiclass_dataset()


def test_bag_shapes_and_count(dataset):
  X, y = dataset
  generator = ClinicalPrevalenceBagGenerator(
    clinical_ranges=CLINICAL_RANGES, n_bags=20, bag_size=300, random_state=0,
  )

  bags = generator.to_list(X, y)

  assert len(bags) == 20
  for X_bag, y_bag in bags:
    assert X_bag.shape == (300, X.shape[1])
    assert y_bag.shape == (300,)


def test_sampled_prevalences_fall_within_clinical_bands(dataset):
  X, y = dataset
  generator = ClinicalPrevalenceBagGenerator(
    clinical_ranges=CLINICAL_RANGES, zone_weights=ZONE_WEIGHTS,
    n_bags=100, bag_size=500, base_label='N', random_state=0,
  )
  generator.to_list(X, y)

  class_to_idx = {c: i for i, c in enumerate(generator.classes_)}

  for label, bands in CLINICAL_RANGES.items():
    idx = class_to_idx[label]
    lo = min(b[0] for b in bands)
    hi = max(b[1] for b in bands)
    sampled = generator.sampled_prevalences_[:, idx]
    # allow a small tolerance for largest-remainder rounding to integer counts
    tolerance = 2.0 / 500
    assert sampled.min() >= lo - tolerance
    assert sampled.max() <= hi + tolerance


def test_prevalence_rows_sum_to_one(dataset):
  X, y = dataset
  generator = ClinicalPrevalenceBagGenerator(
    clinical_ranges=CLINICAL_RANGES, zone_weights=ZONE_WEIGHTS,
    n_bags=50, bag_size=200, base_label='N', random_state=1,
  )
  generator.to_list(X, y)

  row_sums = generator.sampled_prevalences_.sum(axis=1)
  np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)


def test_base_label_absorbs_residual_mass(dataset):
  X, y = dataset
  # tiny, tight ranges for V/A/L/R so most prevalence mass must land on 'N'
  tight_ranges = {
    'V': [(0.0, 0.01)],
    'A': [(0.0, 0.01)],
    'L': [(0.0, 0.01)],
    'R': [(0.0, 0.01)],
  }
  generator = ClinicalPrevalenceBagGenerator(
    clinical_ranges=tight_ranges, n_bags=10, bag_size=1000, base_label='N', random_state=2,
  )
  generator.to_list(X, y)

  class_to_idx = {c: i for i, c in enumerate(generator.classes_)}
  n_prevalence = generator.sampled_prevalences_[:, class_to_idx['N']]

  assert np.all(n_prevalence > 0.9)


def test_rescales_when_sampled_prevalences_overshoot_one(dataset):
  X, y = dataset
  # ranges deliberately wide enough that V + A + L + R can jointly exceed
  # 1.0, forcing the proportional-rescale branch
  overshoot_ranges = {
    'V': [(0.4, 0.6)],
    'A': [(0.4, 0.6)],
    'L': [(0.4, 0.6)],
    'R': [(0.4, 0.6)],
  }
  generator = ClinicalPrevalenceBagGenerator(
    clinical_ranges=overshoot_ranges, n_bags=50, bag_size=500, base_label='N', random_state=3,
  )
  generator.to_list(X, y)

  # rows must still be valid probability vectors even though the raw
  # sampled label prevalences could sum above 1.0 before rescaling
  row_sums = generator.sampled_prevalences_.sum(axis=1)
  np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)
  assert np.all(generator.sampled_prevalences_ >= 0.0)


def test_sampled_zones_are_tracked_per_label(dataset):
  X, y = dataset
  generator = ClinicalPrevalenceBagGenerator(
    clinical_ranges=CLINICAL_RANGES, zone_weights=ZONE_WEIGHTS,
    n_bags=30, bag_size=300, base_label='N', random_state=4,
  )
  generator.to_list(X, y)

  assert set(generator.sampled_zones_.keys()) == set(CLINICAL_RANGES.keys())
  for label, bands in CLINICAL_RANGES.items():
    zones = generator.sampled_zones_[label]
    assert zones.shape == (30,)
    assert zones.min() >= 0
    assert zones.max() < len(bands)


def test_reproducible_with_fixed_random_state(dataset):
  X, y = dataset
  gen_a = ClinicalPrevalenceBagGenerator(clinical_ranges=CLINICAL_RANGES, n_bags=10, bag_size=100, random_state=5)
  gen_b = ClinicalPrevalenceBagGenerator(clinical_ranges=CLINICAL_RANGES, n_bags=10, bag_size=100, random_state=5)

  bags_a = gen_a.to_list(X, y)
  bags_b = gen_b.to_list(X, y)

  np.testing.assert_array_equal(gen_a.sampled_prevalences_, gen_b.sampled_prevalences_)
  for (Xa, ya), (Xb, yb) in zip(bags_a, bags_b):
    np.testing.assert_array_equal(Xa, Xb)
    np.testing.assert_array_equal(ya, yb)


def test_every_class_covered_and_no_base_label_redistributes_uniformly(dataset):
  # every class in y has an explicit (narrow) clinical range and
  # base_label is absent from the data entirely: with no "unspecified"
  # class left to absorb the residual, it must fall back to a uniform
  # `prevalence_vec += residual / n_classes` spread across all classes
  X, y = dataset
  narrow_ranges = {label: [(0.0, 0.05)] for label in np.unique(y)}
  generator = ClinicalPrevalenceBagGenerator(
    clinical_ranges=narrow_ranges, n_bags=10, bag_size=500, base_label='not_in_data', random_state=0,
  )
  generator.to_list(X, y)

  row_sums = generator.sampled_prevalences_.sum(axis=1)
  np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)
  # residual mass (>= 0.75, since every class is capped at 5%) must have
  # been spread across all 5 classes, not concentrated arbitrarily
  assert np.all(generator.sampled_prevalences_ > 0.0)


def test_raises_when_clinical_range_label_missing_from_y():
  rng = np.random.default_rng(0)
  X = rng.normal(size=(100, 3))
  y = np.array(['N', 'V'] * 50)  # no 'A', 'L', 'R' present
  generator = ClinicalPrevalenceBagGenerator(clinical_ranges=CLINICAL_RANGES, n_bags=5, bag_size=20, random_state=0)

  with pytest.raises(ValueError, match="not present in y"):
    generator.to_list(X, y)


def test_raises_on_mismatched_zone_weights_length(dataset):
  X, y = dataset
  bad_weights = {'V': [0.5, 0.5]}  # CLINICAL_RANGES['V'] has 3 bands, not 2
  generator = ClinicalPrevalenceBagGenerator(
    clinical_ranges=CLINICAL_RANGES, zone_weights=bad_weights, n_bags=5, bag_size=100, random_state=0,
  )

  with pytest.raises(ValueError, match="must have length"):
    generator.to_list(X, y)


def test_no_explicit_base_label_redistributes_residual_uniformly():
  # base_label defaults to 'N', but if it's not in the data at all, the
  # residual mass must be redistributed instead of silently dropped
  rng = np.random.default_rng(0)
  sizes = {'V': 500, 'A': 500, 'L': 500, 'R': 500}
  X_parts, y_parts = [], []
  for label, size in sizes.items():
    X_parts.append(rng.normal(size=(size, 3)))
    y_parts.append(np.full(size, label))
  X, y = np.vstack(X_parts), np.concatenate(y_parts)

  narrow_ranges = {
    'V': [(0.0, 0.05)],
    'A': [(0.0, 0.05)],
  }
  generator = ClinicalPrevalenceBagGenerator(
    clinical_ranges=narrow_ranges, n_bags=20, bag_size=200, base_label='N', random_state=0,
  )
  generator.to_list(X, y)

  row_sums = generator.sampled_prevalences_.sum(axis=1)
  np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)
