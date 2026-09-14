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


def guaranteed_patient_train_test_split(X: np.ndarray,
                                        y: np.ndarray,
                                        record_ids: np.ndarray,
                                        ensure_labels: Sequence = None,
                                        test_size: float = 0.3,
                                        stratify: bool = True,
                                        random_state: int = 0,
                                        max_attempts: int = 50):
  """Like `patient_train_test_split`, but retries with different seeds
  until every label in `ensure_labels` is present in *both* the
  resulting train and test splits, or `max_attempts` is exhausted.

  Motivation: a patient-wise split is a group-wise split -- an entire
  patient's beats land on one side or the other, never both. If a rare
  label's beats happen to be concentrated in very few distinct patients
  (as V/A/L/R annotations are in MIT-BIH, unevenly distributed across
  the 48 records), an unlucky `random_state` can easily put every
  patient carrying that label entirely into one split, leaving the
  other with zero instances of it -- silently breaking anything
  downstream that needs every class present in both splits (a native
  multiclass quantifier's `.fit()`, every `OneVsRestQuantifier` member,
  and `ClinicalPrevalenceBagGenerator`, which raises if a
  `clinical_ranges` label is entirely absent from the bag-generation
  side's `y`).

  Parameters
  ----------
  X, y, record_ids : ndarray
    Same as `patient_train_test_split`.
  ensure_labels : Sequence, default = None
    Labels required to be present in both splits. Defaults to every
    unique label in `y` (i.e. full coverage). Pass a subset (e.g. just
    the rare ones) to only guard against those.
  test_size, stratify : same as `patient_train_test_split`.
  random_state : int, default = 0
    Tried first, as-is (so a caller's explicit seed is respected
    whenever it already works). If it doesn't give full coverage,
    subsequent candidate seeds are drawn deterministically from a
    `np.random.default_rng(random_state)`, so the *retry sequence*
    itself is reproducible even though it may land on a different
    winning seed than `random_state`.
  max_attempts : int, default = 50
    Maximum number of seeds tried (including `random_state` itself)
    before giving up.

  Returns
  -------
  X_train, X_test, y_train, y_test, ids_train, ids_test : ndarray, ...
    Same as `patient_train_test_split`.
  winning_random_state : int
    The seed that actually produced the returned split. Save this
    alongside the split artifacts -- it's what makes the split
    reproducible, not the originally-requested `random_state`.

  Raises
  ------
  RuntimeError
    If no split among `max_attempts` tried seeds gives every label in
    `ensure_labels` full coverage in both splits. This is not always a
    matter of trying more seeds: if a label's beats come from a single
    distinct patient, NO group-wise split can ever put that patient's
    beats on both sides simultaneously, and every attempt will fail
    identically regardless of `max_attempts`. The error message lists
    every attempted seed's missing labels so this structural case is
    distinguishable from "just got unlucky, try more attempts".

  Examples
  --------
  >>> from ecg_quantification.model_selection import guaranteed_patient_train_test_split
  >>> result = guaranteed_patient_train_test_split(X, y, record_ids, test_size=0.3, random_state=0)
  >>> X_train, X_test, y_train, y_test, ids_train, ids_test, seed = result
  """
  y = np.asarray(y)
  ensure_labels = set(np.unique(y)) if ensure_labels is None else set(ensure_labels)

  rng = np.random.default_rng(random_state)
  extra_seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=max(max_attempts - 1, 0))]
  candidate_seeds = [random_state] + extra_seeds

  attempts_log = []
  for seed in candidate_seeds:
    split = patient_train_test_split(X, y, record_ids, test_size=test_size, stratify=stratify, random_state=seed)
    _, _, y_train, y_test, _, _ = split

    missing = (ensure_labels - set(np.unique(y_train))) | (ensure_labels - set(np.unique(y_test)))
    attempts_log.append((seed, sorted(missing)))

    if not missing:
      return split + (seed,)

  attempts_str = "\n".join(f"  seed={s}: missing {m}" for s, m in attempts_log)
  raise RuntimeError(
    f"No patient-wise split among {len(candidate_seeds)} attempts had every label in "
    f"{sorted(ensure_labels)} present in both train and test.\n{attempts_str}\n"
    "This usually means a rare label's beats are concentrated in too few distinct "
    "patients for any group-wise split to guarantee representation on both sides. "
    "Options: (1) adjust --test-size, (2) manually pin specific patient IDs to each "
    "split instead of a random group split, or (3) drop the offending label from this "
    "experiment and document it as a limitation."
  )