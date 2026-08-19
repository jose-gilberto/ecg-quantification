"""One-vs-Rest wrapper for binary-only quack quantifiers.

Several quack quantifiers are structurally binary (`ACC`, `PACC`, `HDy`,
`DyS`, `FormanMM`, `CDE`: all built on `BaseCalibratedQuantifier`
subclasses that raise `ValueError` for >2 classes). `OneVsRestQuantifier`
lets any of these participate in a multiclass problem by fitting one
independent binary quantifier per class (label vs. everything else) and
combining their positive-class prevalence estimates into a single
normalized multiclass prevalence vector.

This is a quantification-specific analogue of `sklearn.multiclass.
OneVsRestClassifier`, but aggregates *prevalence estimates* rather than
decision scores, and follows the shared `BaseQuantifier.fit`/`.predict`
contract so it drops into the same evaluation code as any native
multiclass quack quantifier.
"""
import numpy as np
from sklearn.base import clone
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from quack.quantifiers.base import BaseQuantifier, normalize_prevalence


def _fit_binary_member_job(quantifier: BaseQuantifier, X: np.ndarray, y_binary: np.ndarray) -> BaseQuantifier:
  """Fits one independent binary quantifier for a single label-vs-rest split.

  Defined at module level (rather than as a closure/method) so it can be
  pickled and dispatched to worker processes by `joblib`/`Parallel`, the
  same pattern used throughout `quack.ensembles`.
  """
  member = clone(quantifier)
  member.fit(X, y_binary)
  return member


def _predict_binary_member_job(member: BaseQuantifier, X: np.ndarray) -> float:
  """Scores one test bag against a single fitted binary member and
  extracts its estimated positive-class ("label") prevalence.

  Binary quack quantifiers place the positive class at index 1 of their
  `.classes_` (the convention `np.unique([0, 1])` guarantees), so the
  second entry of `.predict(X)` is exactly the label-vs-rest prevalence.
  """
  return member.predict(X)[1]


class OneVsRestQuantifier(BaseQuantifier):
  """Turns a binary-only quack quantifier into a multiclass one via
  One-vs-Rest decomposition.

  For each class `c` observed during `fit`, an independent clone of
  `base_quantifier` is trained on a binary relabeling of the training
  data (`1` where `y == c`, `0` otherwise). At predict time, every
  member reports its estimate of `c`'s prevalence in the test bag; the
  resulting per-class estimates are stacked and renormalized to sum to
  1.0 (they are not guaranteed to already do so, since each binary
  problem is solved independently and OvR estimates routinely
  over/undershoot a valid simplex point).

  Parameters
  ----------
  base_quantifier : BaseQuantifier
    The binary quantifier template cloned once per class (e.g. `ACC()`,
    `HDy()`, `DyS(distance_metric="HD")`). Must implement the standard
    `.fit(X, y)` / `.predict(X)` contract and accept exactly 2 classes.
  n_jobs : int, default = None
    Number of jobs to run in parallel while fitting/predicting the
    `n_classes` independent binary members, since none of them depend on
    each other. `None` means sequential; `-1` uses all available
    processors. See `joblib.Parallel`.
  parallel_backend : str, default = "loky"
    `joblib.Parallel` backend used for the per-class jobs.

  Attributes
  ----------
  classes_ : ndarray of shape (n_classes,)
    The distinct class labels found during the training phase, sorted
    ascending.
  n_classes_ : int
    The total number of unique classes.
  train_prevalence_ : ndarray of shape (n_classes,)
    The prevalence of each class in the full training dataset.
  estimators_ : list of BaseQuantifier
    The `n_classes_` fitted binary members, aligned with `classes_`.

  Notes
  -----
  Unlike `quack.ensembles.EoQ` (which resamples training bags), every
  member here is fit on the *same* full training set, just relabeled;
  there is no risk of a member missing a class, since every relabeling
  is guaranteed to contain both `0` and `1` as long as `c` is neither
  entirely absent nor the only class present in `y`.

  Examples
  --------
  >>> from quack.quantifiers import ACC, HDy
  >>> from sklearn.linear_model import LogisticRegression
  >>> from ecg_quantification.quantifiers import OneVsRestQuantifier
  >>> ovr_acc = OneVsRestQuantifier(ACC(classifier=LogisticRegression(max_iter=1000), cv=5))
  >>> ovr_acc.fit(X_train, y_train)  # y_train has classes 'N', 'V', 'A', 'L', 'R'
  >>> prevalences = ovr_acc.predict(X_test)  # shape (5,), sums to 1.0
  """

  def __init__(self,
               base_quantifier: BaseQuantifier,
               n_jobs: int = None,
               parallel_backend: str = "loky"):
    super().__init__(classifier=None)
    self.base_quantifier = base_quantifier
    self.n_jobs = n_jobs
    self.parallel_backend = parallel_backend

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'OneVsRestQuantifier':
    """Fits one independent binary quantifier per class (label vs. rest).

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
      Training data.
    y : array-like of shape (n_samples,)
      Labels for the corresponding classes. Must contain at least 2
      distinct classes.

    Returns
    -------
    self : object
      Returns the fitted estimator instance itself.
    """
    X, y = check_X_y(X, y, accept_sparse=True, dtype=None)

    self.classes_, counts = np.unique(y, return_counts=True)
    self.n_classes_ = len(self.classes_)
    self.train_prevalence_ = counts / len(y)

    if self.n_classes_ < 2:
      raise ValueError(
        f"OneVsRestQuantifier requires at least 2 distinct classes, got {self.n_classes_}."
      )

    fit_jobs = [
      delayed(_fit_binary_member_job)(self.base_quantifier, X, (y == c).astype(int))
      for c in self.classes_
    ]
    self.estimators_ = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(fit_jobs)

    return self

  def predict(self, X: np.ndarray) -> np.ndarray:
    """Aggregates every One-vs-Rest member's positive-class estimate
    into a single, normalized multiclass prevalence vector.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
      The test bag with unlabelled instances.

    Returns
    -------
    prevalences : ndarray of shape (n_classes,)
      Estimated prevalence per class, aligned with `classes_`,
      normalized to sum to 1.0.
    """
    check_is_fitted(self)
    X = check_array(X, accept_sparse=True, dtype=None)

    predict_jobs = [delayed(_predict_binary_member_job)(member, X) for member in self.estimators_]
    raw_prevalences = np.array(Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(predict_jobs))

    return normalize_prevalence(raw_prevalences, self.n_classes_)