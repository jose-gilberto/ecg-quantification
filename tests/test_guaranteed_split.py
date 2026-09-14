"""Tests for `ecg_quantification.model_selection.guaranteed_patient_train_test_split`."""
import numpy as np
import pytest

from ecg_quantification.model_selection import (
  guaranteed_patient_train_test_split,
  assert_no_patient_leakage,
)


def _dataset_with_rare_label_confined_to_one_patient(random_state=0):
  """6 patients contribute plenty of 'N'/'V'; exactly ONE additional
  patient ever contributes 'R' -- the MIT-BIH-style situation this
  function exists for. No group-wise split can ever satisfy full
  coverage here: it's a structural impossibility, not bad luck."""
  rng = np.random.default_rng(random_state)
  X_parts, y_parts, ids_parts = [], [], []
  for patient in range(6):
    n = 100
    X_parts.append(rng.normal(size=(n, 3)))
    y_parts.append(rng.choice(['N', 'V'], size=n))
    ids_parts.append(np.full(n, f'p{patient}'))
  n_rare = 20
  X_parts.append(rng.normal(size=(n_rare, 3)))
  y_parts.append(np.full(n_rare, 'R'))
  ids_parts.append(np.full(n_rare, 'p6'))
  return np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(ids_parts)


def _dataset_with_every_label_spread_across_many_patients(random_state=0):
  rng = np.random.default_rng(random_state)
  labels_pool = ['N', 'V', 'A', 'L', 'R']
  X_parts, y_parts, ids_parts = [], [], []
  for patient in range(20):
    n = 50
    X_parts.append(rng.normal(size=(n, 3)))
    y_parts.append(rng.choice(labels_pool, size=n))
    ids_parts.append(np.full(n, f'p{patient}'))
  return np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(ids_parts)


def test_raises_clear_error_when_rare_label_confined_to_a_single_patient():
  X, y, record_ids = _dataset_with_rare_label_confined_to_one_patient()

  with pytest.raises(RuntimeError, match="No patient-wise split"):
    guaranteed_patient_train_test_split(X, y, record_ids, test_size=0.3, random_state=0, max_attempts=5)


def test_finds_a_split_with_full_label_coverage_when_feasible():
  X, y, record_ids = _dataset_with_every_label_spread_across_many_patients()

  result = guaranteed_patient_train_test_split(X, y, record_ids, test_size=0.3, random_state=0, max_attempts=50)
  X_train, X_test, y_train, y_test, ids_train, ids_test, seed = result

  assert set(np.unique(y_train)) == set(np.unique(y))
  assert set(np.unique(y_test)) == set(np.unique(y))
  assert_no_patient_leakage(ids_train, ids_test)
  assert isinstance(seed, int)


def test_given_random_state_is_used_as_is_when_it_already_works():
  # every patient individually contains every label, so genuinely ANY
  # split has full coverage -- the given seed must be accepted on the
  # first try, with no retry needed
  rng = np.random.default_rng(0)
  labels_pool = ['N', 'V', 'A', 'L', 'R']
  X_parts, y_parts, ids_parts = [], [], []
  for patient in range(10):
    n = 50
    X_parts.append(rng.normal(size=(n, 3)))
    y = rng.choice(labels_pool, size=n)
    y[:len(labels_pool)] = labels_pool  # force every label to appear for this patient
    y_parts.append(y)
    ids_parts.append(np.full(n, f'p{patient}'))
  X, y, record_ids = np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(ids_parts)

  *_, seed = guaranteed_patient_train_test_split(X, y, record_ids, test_size=0.3, random_state=42, max_attempts=1)

  assert seed == 42


def test_ensure_labels_can_be_a_strict_subset():
  X, y, record_ids = _dataset_with_every_label_spread_across_many_patients()

  X_train, X_test, y_train, y_test, ids_train, ids_test, seed = guaranteed_patient_train_test_split(
    X, y, record_ids, ensure_labels=['L', 'R'], test_size=0.3, random_state=0, max_attempts=50,
  )

  assert {'L', 'R'} <= set(np.unique(y_train))
  assert {'L', 'R'} <= set(np.unique(y_test))