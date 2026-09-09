"""Smoke tests for `ecg_quantification.APPBagGenerator`.

Only exercises the migration from `legacy_cbms26.bags.generate_bags`
into the `BaseBagGenerator`-conformant version. A broader test suite
(covering `OneVsRestQuantifier`, `ClinicalPrevalenceBagGenerator`,
`ZoneCrossingError`, and the multiclass preprocessing path) is tracked
as a separate, follow-up task.
"""
import numpy as np
import pytest

from ecg_quantification import APPBagGenerator


@pytest.fixture
def binary_dataset():
  rng = np.random.default_rng(0)
  n_pos, n_neg = 300, 300
  X_pos = rng.normal(loc=1.0, size=(n_pos, 10))
  X_neg = rng.normal(loc=-1.0, size=(n_neg, 10))
  X = np.vstack([X_pos, X_neg])
  y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
  return X, y


def test_generates_requested_number_of_bags(binary_dataset):
  X, y = binary_dataset
  generator = APPBagGenerator(n_bags=15, bag_size=50, random_state=0)

  bags = generator.to_list(X, y)

  assert len(bags) == 15
  for X_bag, y_bag in bags:
    assert X_bag.shape == (50, X.shape[1])
    assert y_bag.shape == (50,)


def test_realized_prevalence_matches_bag_labels(binary_dataset):
  X, y = binary_dataset
  generator = APPBagGenerator(n_bags=25, bag_size=200, positive_label=1, random_state=1)

  bags = generator.to_list(X, y)

  for i, (_, y_bag) in enumerate(bags):
    empirical_prevalence = np.mean(y_bag == 1)
    assert empirical_prevalence == pytest.approx(generator.sampled_prevalences_[i])


def test_prevalences_span_close_to_the_full_range(binary_dataset):
  # APP (unlike the clinically-constrained protocol) should realistically
  # cover close to the full [0, 1] simplex given enough bags.
  X, y = binary_dataset
  generator = APPBagGenerator(n_bags=200, bag_size=100, random_state=2)

  generator.to_list(X, y)

  assert generator.sampled_prevalences_.min() < 0.05
  assert generator.sampled_prevalences_.max() > 0.95


def test_reproducible_with_fixed_random_state(binary_dataset):
  X, y = binary_dataset

  gen_a = APPBagGenerator(n_bags=10, bag_size=40, random_state=7)
  gen_b = APPBagGenerator(n_bags=10, bag_size=40, random_state=7)

  bags_a = gen_a.to_list(X, y)
  bags_b = gen_b.to_list(X, y)

  for (Xa, ya), (Xb, yb) in zip(bags_a, bags_b):
    np.testing.assert_array_equal(Xa, Xb)
    np.testing.assert_array_equal(ya, yb)


def test_rejects_non_binary_labels():
  rng = np.random.default_rng(0)
  X = rng.normal(size=(90, 5))
  y = np.array(['N', 'V', 'A'] * 30)
  generator = APPBagGenerator(n_bags=5, bag_size=10, random_state=0)

  with pytest.raises(ValueError, match="only supports binary quantification"):
    generator.to_list(X, y)


def test_rejects_unknown_positive_label(binary_dataset):
  X, y = binary_dataset
  generator = APPBagGenerator(n_bags=5, bag_size=10, positive_label=99, random_state=0)

  with pytest.raises(ValueError, match="positive_label"):
    generator.to_list(X, y)
