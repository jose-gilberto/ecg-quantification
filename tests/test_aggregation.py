"""Tests for `ecg_quantification.aggregate_prevalence_to_binary` and
`aggregate_bag_true_prevalences_to_binary`."""
import numpy as np
import pytest

from ecg_quantification import aggregate_prevalence_to_binary, aggregate_bag_true_prevalences_to_binary


CLASSES = np.array(['A', 'L', 'N', 'R', 'V'])


def test_aggregates_everything_except_base_label():
  prevalence = np.array([0.02, 0.01, 0.90, 0.01, 0.06])
  assert aggregate_prevalence_to_binary(prevalence, CLASSES) == pytest.approx(0.10)


def test_all_mass_on_base_label_gives_zero_burden():
  prevalence = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
  assert aggregate_prevalence_to_binary(prevalence, CLASSES) == pytest.approx(0.0)


def test_no_mass_on_base_label_gives_full_burden():
  prevalence = np.array([0.25, 0.25, 0.0, 0.25, 0.25])
  assert aggregate_prevalence_to_binary(prevalence, CLASSES) == pytest.approx(1.0)


def test_custom_base_label():
  # aggregating around a different residual class must also work
  prevalence = np.array([0.7, 0.1, 0.1, 0.05, 0.05])
  assert aggregate_prevalence_to_binary(prevalence, CLASSES, base_label='A') == pytest.approx(0.3)


def test_raises_when_base_label_not_in_classes():
  prevalence = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
  with pytest.raises(ValueError, match="not found in classes"):
    aggregate_prevalence_to_binary(prevalence, CLASSES, base_label='X')


def test_vectorized_matches_scalar_version_row_by_row():
  rng = np.random.default_rng(0)
  raw = rng.dirichlet(np.ones(5), size=20)  # 20 valid prevalence rows summing to 1

  vectorized = aggregate_bag_true_prevalences_to_binary(raw, CLASSES)
  scalar = np.array([aggregate_prevalence_to_binary(row, CLASSES) for row in raw])

  np.testing.assert_allclose(vectorized, scalar)


def test_vectorized_raises_when_base_label_not_in_classes():
  raw = np.zeros((3, 5))
  with pytest.raises(ValueError, match="not found in classes"):
    aggregate_bag_true_prevalences_to_binary(raw, CLASSES, base_label='X')