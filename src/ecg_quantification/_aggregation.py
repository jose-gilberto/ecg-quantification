"""Reconstructing the original binary (N vs. arrhythmic) burden estimate
from a multiclass (N/V/A/L/R) prevalence vector.

The CBMS'26 study framed arrhythmia burden as a single binary quantity
(N vs. not-N). The multiclass extension estimates each class's
prevalence individually (see `OneVsRestQuantifier`,
`build_multiclass_quantifiers`), which is clinically richer (V/A/L/R
each have their own risk zones in `CLINICAL_RANGES`) but no longer
directly comparable to the original binary framing, or to a plain
CC-style binary baseline evaluated the old way.

`aggregate_prevalence_to_binary` closes that gap: given any multiclass
prevalence vector (predicted or true) aligned to a `classes` array, it
collapses it back to the single scalar "total arrhythmic burden"
(1 - P(base_label)). This lets the extension report, side by side: (a)
the per-class multiclass breakdown + Zone-Crossing Error, and (b) a
directly CBMS'26-comparable binary MAE, computed from the *same*
multiclass fit rather than a separately-trained binary model -- i.e.
whether collapsing a good multiclass estimate loses anything relative
to a purpose-fit binary one.
"""
import numpy as np


def aggregate_prevalence_to_binary(prevalence: np.ndarray,
                                   classes,
                                   base_label='N') -> float:
  """Collapses a multiclass prevalence vector into a single "arrhythmic
  burden" scalar: the total prevalence of every class except
  `base_label`.

  Parameters
  ----------
  prevalence : ndarray of shape (n_classes,)
    A multiclass prevalence vector (predicted or true), aligned with
    `classes` -- e.g. a quantifier's `.predict(X)` output, or a row of
    `ClinicalPrevalenceBagGenerator.sampled_prevalences_`.
  classes : array-like of shape (n_classes,)
    The class labels `prevalence` is aligned with, in the same order
    (e.g. a quantifier's `.classes_`, or
    `ClinicalPrevalenceBagGenerator.classes_`).
  base_label : default = 'N'
    The residual/"healthy" class whose prevalence is subtracted from 1.
    Must be present in `classes`.

  Returns
  -------
  binary_burden : float
    `1.0 - prevalence[index of base_label]`, i.e. the combined
    prevalence of every non-`base_label` class.

  Raises
  ------
  ValueError
    If `base_label` is not found in `classes`.

  Examples
  --------
  >>> import numpy as np
  >>> from ecg_quantification import aggregate_prevalence_to_binary
  >>> classes = np.array(['A', 'L', 'N', 'R', 'V'])
  >>> prevalence = np.array([0.02, 0.01, 0.90, 0.01, 0.06])
  >>> round(aggregate_prevalence_to_binary(prevalence, classes), 2)
  0.1
  """
  classes = np.asarray(classes)
  matches = np.flatnonzero(classes == base_label)
  if matches.size == 0:
    raise ValueError(f"base_label={base_label!r} not found in classes={classes.tolist()}.")
  return float(1.0 - prevalence[matches[0]])


def aggregate_bag_true_prevalences_to_binary(sampled_prevalences: np.ndarray,
                                             classes,
                                             base_label='N') -> np.ndarray:
  """Vectorized `aggregate_prevalence_to_binary` over every bag in a
  `ClinicalPrevalenceBagGenerator.sampled_prevalences_`-style matrix.

  Parameters
  ----------
  sampled_prevalences : ndarray of shape (n_bags, n_classes)
    E.g. `ClinicalPrevalenceBagGenerator.sampled_prevalences_`.
  classes : array-like of shape (n_classes,)
    Aligned with `sampled_prevalences`'s columns.
  base_label : default = 'N'
    See `aggregate_prevalence_to_binary`.

  Returns
  -------
  binary_burdens : ndarray of shape (n_bags,)
  """
  classes = np.asarray(classes)
  matches = np.flatnonzero(classes == base_label)
  if matches.size == 0:
    raise ValueError(f"base_label={base_label!r} not found in classes={classes.tolist()}.")
  return 1.0 - sampled_prevalences[:, matches[0]]