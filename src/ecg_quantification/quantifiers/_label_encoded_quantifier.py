"""Label-encoding wrapper protecting `quack` quantifiers from a hard-label
out-of-fold dtype limitation, without touching `quack` itself.

`quack.quantifiers.base.BaseCalibratedQuantifier.fit()` allocates its
out-of-fold prediction buffer as `np.zeros(n_samples)` -- always
`float64` -- whenever a subclass's `_get_oof_method()` returns
`"predict"` (hard labels) rather than `"predict_proba"`. Every classifier
this project fits echoes back whatever label dtype it was trained on, and
this project's multiclass labels are string beat symbols ('N', 'V', 'A',
'L', 'R'), so assigning a fold's string predictions into that float64
buffer raises `ValueError: could not convert string to float`.

`GAC` is the only *natively multiclass* quantifier in this project's
registry affected (`_get_oof_method() == "predict"`); `GPAC`, `FM`, `EM`
all use `"predict_proba"`, which is numeric regardless of `y`'s dtype
(see `docs/quantifier_registry_audit.md`). `ACC` uses `"predict"` too,
but its own `.fit()` override rejects >2 classes before reaching this
code path, so it only matters in binary (OvR-member) use, where
`OneVsRestQuantifier` already sidesteps the whole issue by binarizing
labels to `int` (`(y == c).astype(int)`) before fitting each member.

`LabelEncodedQuantifier` closes the one remaining gap -- fitting a
natively multiclass, hard-label quantifier (`GAC` today; any future
`quack` quantifier with the same `"predict"`-OOF shape) directly on this
project's string labels, without a one-vs-rest decomposition -- by
encoding `y` to small integer codes before handing it to the wrapped
quantifier, and exposing `.classes_` as the *original* string labels so
callers never need to track a separate label-order array by hand (unlike
the inline `np.searchsorted(all_classes, y_train)` this replaces in
`scripts/run_multiclass_experiment.py`).

This intentionally lives in `ecg_quantification`, not `quack`: the
project has decided not to patch `quack` for this (see
`docs/dataset_limitations.md`-style call), so every multiclass quantifier
this project fits on string labels -- including future classifiers such
as `LITETimeClassifier` run on the 5-class ECG problem -- should go
through this wrapper (or `OneVsRestQuantifier`, for binary-only
quantifiers) rather than being fit directly.
"""
import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from quack.quantifiers.base import BaseQuantifier


class LabelEncodedQuantifier(BaseQuantifier):
  """Fits any `quack` quantifier on integer-encoded labels, transparently.

  Encodes `y` to `0..n_classes-1` integer codes (via `np.searchsorted`
  against the sorted unique labels -- the same rule `np.unique`/
  `sklearn.preprocessing.LabelEncoder` already use internally, so this
  introduces no discrepancy with how `quack` derives `classes_` on the
  encoded labels it actually sees) before delegating to the wrapped
  quantifier's `.fit()`. `.predict()` is a pure passthrough: quack's own
  `self.classes_ = np.unique(y_encoded)` on `[0, ..., n_classes-1]`
  reproduces the exact same sorted order used to build the encoding, so
  the wrapped quantifier's output at position `i` already corresponds to
  `self.classes_[i]` (the original label) -- no output reordering is
  needed, only exposing the right `classes_` for interpretation.

  Parameters
  ----------
  quantifier : BaseQuantifier
    The quantifier template to fit on encoded labels (e.g. `GAC(...)`,
    or any future `quack` quantifier with the same hard-label OOF
    shape). Cloned once in `.fit()`, so the instance passed here is
    never mutated and can be reused across multiple
    `LabelEncodedQuantifier`/direct-fit calls.

  Attributes
  ----------
  classes_ : ndarray of shape (n_classes,)
    The distinct *original* (string or otherwise) class labels found
    during `.fit()`, sorted ascending -- exactly what a caller would get
    from fitting `quantifier` directly on labels that happened to already
    be numeric.
  n_classes_ : int
    The total number of unique classes.
  train_prevalence_ : ndarray of shape (n_classes,)
    The prevalence of each class in the full training dataset, aligned
    with `classes_`.
  quantifier_ : BaseQuantifier
    The fitted clone of `quantifier`, fit on integer-encoded labels.

  Examples
  --------
  >>> from quack.quantifiers import GAC
  >>> from sklearn.linear_model import LogisticRegression
  >>> from ecg_quantification.quantifiers import LabelEncodedQuantifier
  >>> quantifier = LabelEncodedQuantifier(GAC(classifier=LogisticRegression(max_iter=1000), cv=5))
  >>> quantifier.fit(X_train, y_train)  # y_train has string classes 'N', 'V', 'A', 'L', 'R'
  >>> prevalences = quantifier.predict(X_test)  # shape (5,), aligned with quantifier.classes_
  """

  def __init__(self, quantifier: BaseQuantifier):
    super().__init__(classifier=None)
    self.quantifier = quantifier

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'LabelEncodedQuantifier':
    """Encodes `y` to integer codes and fits a clone of `quantifier`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
      Training data.
    y : array-like of shape (n_samples,)
      Labels of any dtype (string symbols included). Must contain at
      least 2 distinct classes.

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
        f"LabelEncodedQuantifier requires at least 2 distinct classes, got {self.n_classes_}."
      )

    y_encoded = np.searchsorted(self.classes_, y)

    self.quantifier_ = clone(self.quantifier)
    self.quantifier_.fit(X, y_encoded)

    return self

  def predict(self, X: np.ndarray) -> np.ndarray:
    """Delegates to the fitted wrapped quantifier; no output reordering
    needed (see class docstring).

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
      The test bag with unlabelled instances.

    Returns
    -------
    prevalences : ndarray of shape (n_classes,)
      Estimated prevalence per class, aligned with `classes_`.
    """
    check_is_fitted(self, 'quantifier_')
    X = check_array(X, accept_sparse=True, dtype=None)
    return self.quantifier_.predict(X)
