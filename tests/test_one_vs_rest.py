"""Tests for `ecg_quantification.OneVsRestQuantifier`.

Two layers of testing are used deliberately:

- A `_FakeBinaryQuantifier` isolates OvR's own behavior (per-class
  fitting, prevalence aggregation, normalization, sklearn `clone()`
  compatibility) from the internals of any specific quack quantifier.
- A real `quack.quantifiers.ACC` integration test catches wiring issues
  between `OneVsRestQuantifier` and quack's actual `BaseQuantifier`
  contract (the thing a fake object could paper over).
"""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from quack.quantifiers.base import BaseQuantifier
from quack.quantifiers import ACC
from ecg_quantification import OneVsRestQuantifier


class _FakeBinaryQuantifier(BaseQuantifier):
  """Minimal binary quantifier: 'predicts' the true positive-class
  frequency in `X`'s (label-encoded) rows exactly, so OvR's aggregation
  arithmetic can be checked without any real classifier noise."""

  def __init__(self, classifier=None):
    super().__init__(classifier=classifier)

  def fit(self, X, y):
    unique = np.unique(y)
    if len(unique) > 2:
      raise ValueError("_FakeBinaryQuantifier only works for binary quantification.")
    self.classes_ = unique
    return self

  def predict(self, X):
    # X's single column directly encodes the bag's true positive rate,
    # replicated across rows for check_array's 2D shape requirement.
    positive_rate = float(X[0, 0])
    return np.array([1.0 - positive_rate, positive_rate])


def _multiclass_dataset(n_per_class=40, n_classes=4, random_state=0):
  """Builds a synthetic multiclass dataset where every class is
  linearly separable from the union of the rest (each class gets its
  own dedicated feature axis). This matters specifically for the OvR
  integration test below: if cluster centers are collinear (e.g. placed
  along a single shared direction), a "sandwiched" middle class is not
  linearly separable from the other two combined, and a linear base
  classifier (LogisticRegression) degenerates -- a real, well known
  limitation of One-vs-Rest with linear classifiers, but not what this
  test suite means to exercise.
  """
  rng = np.random.default_rng(random_state)
  labels = np.array(['N', 'V', 'A', 'L', 'R'][:n_classes])
  X_parts, y_parts = [], []
  for i, label in enumerate(labels):
    center = np.zeros(n_classes)
    center[i] = 6.0
    X_parts.append(rng.normal(loc=center, scale=1.0, size=(n_per_class, n_classes)))
    y_parts.append(np.full(n_per_class, label))
  X = np.vstack(X_parts)
  y = np.concatenate(y_parts)

  shuffle_idx = rng.permutation(len(y))
  return X[shuffle_idx], y[shuffle_idx], labels


def test_fit_creates_one_estimator_per_class():
  X, y, labels = _multiclass_dataset(n_classes=3)
  ovr = OneVsRestQuantifier(_FakeBinaryQuantifier())

  ovr.fit(X, y)

  assert ovr.n_classes_ == 3
  np.testing.assert_array_equal(ovr.classes_, sorted(labels))
  assert len(ovr.estimators_) == 3
  assert ovr.train_prevalence_.sum() == pytest.approx(1.0)


def test_fit_rejects_single_class():
  X = np.zeros((10, 2))
  y = np.array(['N'] * 10)
  ovr = OneVsRestQuantifier(_FakeBinaryQuantifier())

  with pytest.raises(ValueError, match="at least 2 distinct classes"):
    ovr.fit(X, y)


def test_predict_returns_normalized_vector_aligned_with_classes():
  X, y, labels = _multiclass_dataset(n_classes=4)
  ovr = OneVsRestQuantifier(_FakeBinaryQuantifier())
  ovr.fit(X, y)

  # bag "claims" (via the fake quantifier's single-column convention) a
  # 0.9 positive rate for every one-vs-rest member -- deliberately
  # overshooting the simplex, exercising `normalize_prevalence`.
  X_bag = np.full((5, 4), 0.9)
  prevalences = ovr.predict(X_bag)

  assert prevalences.shape == (4,)
  assert prevalences.sum() == pytest.approx(1.0)
  assert np.all(prevalences >= 0.0)
  # every member reported the same raw score, so after renormalization
  # the estimate should be uniform across classes
  np.testing.assert_allclose(prevalences, np.full(4, 0.25))


def test_predict_before_fit_raises():
  ovr = OneVsRestQuantifier(_FakeBinaryQuantifier())
  with pytest.raises(Exception):
    ovr.predict(np.zeros((5, 3)))


def test_sklearn_clone_compatibility():
  # OneVsRestQuantifier must survive sklearn.base.clone (used internally
  # by every ensemble/CV routine in quack), which round-trips get_params/
  # set_params rather than deep-copying __dict__ directly.
  ovr = OneVsRestQuantifier(_FakeBinaryQuantifier(), n_jobs=2, parallel_backend="threading")
  cloned = clone(ovr)

  assert cloned.n_jobs == 2
  assert cloned.parallel_backend == "threading"
  assert isinstance(cloned.base_quantifier, _FakeBinaryQuantifier)
  assert not hasattr(cloned, "estimators_")  # clone() must not carry fitted state


@pytest.mark.parametrize("n_jobs", [None, 2])
def test_integration_with_real_quack_quantifier(n_jobs):
  # Real integration path: ACC is structurally binary (raises internally
  # for >2 classes), matching exactly the quantifiers OneVsRestQuantifier
  # is meant to unlock for the multiclass N/V/A/L/R problem.
  X, y, labels = _multiclass_dataset(n_per_class=60, n_classes=3, random_state=1)
  base = ACC(classifier=LogisticRegression(max_iter=1000), cv=3, n_jobs=None)
  ovr = OneVsRestQuantifier(base, n_jobs=n_jobs)

  ovr.fit(X, y)
  # a bag drawn entirely from labels[0]'s cluster ('N') should be
  # estimated as overwhelmingly that class. Note: ovr.classes_ is
  # np.unique(y)-sorted ('A', 'N', 'V'), NOT `labels`' generation
  # order -- look up the expected index there, not at position 0.
  target_label = labels[0]
  X_bag = X[y == target_label]
  prevalences = ovr.predict(X_bag)
  expected_idx = np.searchsorted(ovr.classes_, target_label)

  assert prevalences.shape == (3,)
  assert prevalences.sum() == pytest.approx(1.0, abs=1e-6)
  assert np.argmax(prevalences) == expected_idx
