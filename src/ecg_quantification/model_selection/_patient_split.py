"""Patient-wise train/test splitting.

Standard `train_test_split` shuffles individual samples, which for beat
segments extracted from ECG records leaks the same patient's beats into
both train and test (adjacent/similar beats from one recording end up on
both sides), inflating evaluation metrics. This module splits by patient
(record) group instead: every beat from a given patient stays entirely
in one split.
"""
from typing import Sequence
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GroupKFold


def patient_train_test_split(X: np.ndarray,
                             y: np.ndarray,
                             record_ids: np.ndarray,
                             test_size: float = 0.3,
                             stratify: bool = True,
                             random_state=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray]:
  """Splits `(X, y)` into train/test sets with no patient (record) shared
  across the two splits.

  Parameters
  ----------
  X : ndarray of shape (n_samples, ...)
    Feature matrix (or raw signal segments), one row per beat.
  y : ndarray of shape (n_samples,)
    Beat labels.
  record_ids : ndarray of shape (n_samples,)
    Patient/record identifier per beat (e.g. the `record_ids` output of
    `ECGPreprocessor.process_all_records`). Rows sharing the same value
    are guaranteed to end up in the same split.
  test_size : float, default = 0.3
    Approximate fraction of *patients* (not necessarily samples, since
    patients can have very different beat counts) assigned to the test
    split.
  stratify : bool, default = True
    If True, uses `StratifiedGroupKFold` internals (via a 1-fold-style
    shuffle split) to additionally balance the label distribution across
    splits as far as the group constraint allows. If False, uses a plain
    `GroupShuffleSplit` (label distribution is only a side effect of
    which patients land where).
  random_state : int, RandomState instance or None, default = None
    Controls the randomness of the patient assignment.

  Returns
  -------
  X_train, X_test, y_train, y_test, ids_train, ids_test : ndarray, ...
    The split arrays. `set(ids_train) & set(ids_test)` is always empty.

  Examples
  --------
  >>> from ecg_quantification.model_selection import patient_train_test_split
  >>> X_train, X_test, y_train, y_test, ids_train, ids_test = patient_train_test_split(
  ...   X, y, record_ids, test_size=0.3, random_state=0,
  ... )
  >>> set(ids_train).isdisjoint(set(ids_test))
  True
  """
  X = np.asarray(X)
  y = np.asarray(y)
  record_ids = np.asarray(record_ids)

  if not (len(X) == len(y) == len(record_ids)):
    raise ValueError(
      f"X, y, and record_ids must have the same length, got {len(X)}, {len(y)}, {len(record_ids)}."
    )

  if stratify:
    # StratifiedGroupKFold with k folds ~ 1/test_size approximates a
    # stratified group split: use the first fold's test indices as the
    # held-out split, keeping every patient's beats entirely on one side
    n_splits = max(int(round(1.0 / test_size)), 2)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=record_ids))
  else:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=record_ids))

  X_train, X_test = X[train_idx], X[test_idx]
  y_train, y_test = y[train_idx], y[test_idx]
  ids_train, ids_test = record_ids[train_idx], record_ids[test_idx]

  return X_train, X_test, y_train, y_test, ids_train, ids_test


def patient_kfold_split(X: np.ndarray,
                        y: np.ndarray,
                        record_ids: np.ndarray,
                        n_splits: int = 5,
                        stratify: bool = True,
                        random_state=None):
  """Yields `n_splits` patient-wise (train_idx, test_idx) folds, with no
  patient shared between a fold's train and test indices.

  Parameters
  ----------
  X : ndarray of shape (n_samples, ...)
    Feature matrix (only its length is used for validation here).
  y : ndarray of shape (n_samples,)
    Beat labels.
  record_ids : ndarray of shape (n_samples,)
    Patient/record identifier per beat.
  n_splits : int, default = 5
    Number of folds.
  stratify : bool, default = True
    If True, uses `StratifiedGroupKFold` to additionally balance label
    distribution across folds as far as the group constraint allows.
    If False, uses plain `GroupKFold`.
  random_state : int, RandomState instance or None, default = None
    Controls fold assignment randomness. Only used when `stratify=True`
    (`GroupKFold` is deterministic given `groups`).

  Yields
  ------
  train_idx, test_idx : ndarray, ndarray
    Index arrays into `X`/`y`/`record_ids` for this fold.

  Examples
  --------
  >>> from ecg_quantification.model_selection import patient_kfold_split
  >>> for train_idx, test_idx in patient_kfold_split(X, y, record_ids, n_splits=5, random_state=0):
  ...     X_train, X_test = X[train_idx], X[test_idx]
  ...     y_train, y_test = y[train_idx], y[test_idx]
  """
  X = np.asarray(X)
  y = np.asarray(y)
  record_ids = np.asarray(record_ids)

  if not (len(X) == len(y) == len(record_ids)):
    raise ValueError(
      f"X, y, and record_ids must have the same length, got {len(X)}, {len(y)}, {len(record_ids)}."
    )

  if stratify:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
  else:
    splitter = GroupKFold(n_splits=n_splits)

  for train_idx, test_idx in splitter.split(X, y, groups=record_ids):
    yield train_idx, test_idx


def assert_no_patient_leakage(ids_train: np.ndarray, ids_test: np.ndarray) -> None:
  """Raises `AssertionError` if any patient/record id appears in both
  `ids_train` and `ids_test`. Useful as a sanity check right after any
  split, custom or otherwise.

  Parameters
  ----------
  ids_train : ndarray
    Patient/record ids in the training split.
  ids_test : ndarray
    Patient/record ids in the test split.

  Raises
  ------
  AssertionError
    If the two id sets overlap, naming the offending patient ids.
  """
  overlap = set(np.unique(ids_train)) & set(np.unique(ids_test))
  assert not overlap, f"Patient leakage detected: {sorted(overlap)} appear in both train and test."