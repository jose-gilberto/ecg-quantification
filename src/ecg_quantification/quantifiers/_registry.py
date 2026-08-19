"""Registry of quantifiers available for the multiclass ECG experiments.

Splits quack's quantifier catalog into two groups:

- Natively multiclass: usable directly on the 5-class (N, V, A, L, R)
  problem.
- Structurally binary: wrapped via `OneVsRestQuantifier` to participate
  in the multiclass evaluation.

Includes every quantifier quack ships, not just the six evaluated in the
original CBMS'26 paper (CC, ACC, PCC, PACC, EMQ, HDy) — the extension
also reports HDx, ReadME, ED, DyS, FormanMM, GAC/GPAC/FM, CDE, and the
threshold-selector family (X, Max, T50, MedianSweep).
"""
from typing import Callable
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from quack.quantifiers import (
  CC, PCC, ACC, PACC,
  MedianSweep, X, T50, Max,
  HDy, FM, FormanMM, DyS, GAC, GPAC,
  ReadMe, HDx,
  EM, CDE,
  ED,
)
from ecg_quantification.quantifiers._one_vs_rest import OneVsRestQuantifier


def default_base_classifiers() -> dict[str, Callable]:
  """Factories for the four base classifiers used in the original
  CBMS'26 study (LR, SVM, GB, RF), each a zero-argument callable
  returning a fresh, unfitted estimator instance.

  Kept as factories (not instances) so every quantifier/classifier
  combination below gets its own independent classifier object —
  sharing one instance across quantifiers would let `sklearn.base.clone`
  calls interfere with each other's hyperparameters in edge cases.
  """
  return {
    'LR': lambda: LogisticRegression(max_iter=1000),
    'SVM': lambda: SVC(probability=True),
    'GB': lambda: GradientBoostingClassifier(),
    'RF': lambda: RandomForestClassifier(n_estimators=100),
  }


def build_multiclass_quantifiers(classifier_factory: Callable,
                                 cv: int = 10,
                                 n_jobs: int = None,
                                 parallel_backend: str = "loky") -> dict[str, object]:
  """Builds every natively multiclass quack quantifier for a given base classifier.

  Parameters
  ----------
  classifier_factory : Callable
    Zero-argument callable returning a fresh classifier instance (e.g.
    from `default_base_classifiers()`).
  cv : int, default = 10
    Cross-validation folds for calibrated quantifiers.
  n_jobs : int, default = None
    Forwarded to every quantifier's `n_jobs` parameter.
  parallel_backend : str, default = "loky"
    Forwarded to every quantifier's `parallel_backend` parameter.

  Returns
  -------
  quantifiers : dict[str, BaseQuantifier]
    Maps a short quantifier name to a fresh, unfitted instance.
  """
  common_cv = dict(cv=cv, n_jobs=n_jobs, parallel_backend=parallel_backend)

  return {
    'CC': CC(classifier=classifier_factory()),
    'PCC': PCC(classifier=classifier_factory()),
    'GAC': GAC(classifier=classifier_factory(), **common_cv),
    'GPAC': GPAC(classifier=classifier_factory(), **common_cv),
    'FM': FM(classifier=classifier_factory(), **common_cv),
    'EMQ': EM(classifier=classifier_factory(), **common_cv),
  }


def build_feature_space_quantifiers(n_jobs: int = None,
                                    parallel_backend: str = "loky") -> dict[str, object]:
  """Builds classifier-free, feature-space multiclass quantifiers
  (operate directly on X; no `classifier=` parameter).

  Not part of the per-classifier grid, since they don't take a base
  classifier — evaluated once, independent of `default_base_classifiers()`.
  """
  return {
    'HDx': HDx(),
    'ReadMe': ReadMe(n_jobs=n_jobs, parallel_backend=parallel_backend),
    'ED': ED(n_jobs=n_jobs, parallel_backend=parallel_backend),
  }


def build_binary_quantifiers_for_ovr(classifier_factory: Callable,
                                     cv: int = 10,
                                     n_jobs: int = None,
                                     parallel_backend: str = "loky") -> dict[str, object]:
  """Builds every structurally-binary quack quantifier, each wrapped in
  `OneVsRestQuantifier` so it can be evaluated on the 5-class problem.

  Covers the quantifiers not natively multiclass: ACC, PACC (aggregation
  methods requiring a binary TPR/FPR or mu_pos/mu_neg calibration), the
  threshold-selector family (X, Max, T50, MedianSweep), the score-based
  distribution-matching methods (HDy, DyS, FormanMM), and CDE.

  Parameters
  ----------
  classifier_factory : Callable
    Zero-argument callable returning a fresh classifier instance.
  cv : int, default = 10
    Cross-validation folds for the underlying binary quantifiers.
  n_jobs : int, default = None
    Forwarded both to each binary quantifier's own `n_jobs` and to the
    `OneVsRestQuantifier` wrapper's `n_jobs` would double-parallelize;
    here it is forwarded only to the underlying quantifier, while the
    OvR wrapper itself runs its per-class jobs sequentially by default
    (pass a separate `ovr_n_jobs` at the call site if outer parallelism
    across classes is preferred over inner parallelism across CV folds).
  parallel_backend : str, default = "loky"
    Forwarded to every wrapped quantifier's `parallel_backend`.

  Returns
  -------
  quantifiers : dict[str, OneVsRestQuantifier]
    Maps a short quantifier name to a fresh `OneVsRestQuantifier`
    wrapping the corresponding binary quack quantifier.
  """
  common_cv = dict(cv=cv, n_jobs=n_jobs, parallel_backend=parallel_backend)

  binary_members = {
    'ACC': ACC(classifier=classifier_factory(), **common_cv),
    'PACC': PACC(classifier=classifier_factory(), **common_cv),
    'X': X(classifier=classifier_factory(), **common_cv),
    'Max': Max(classifier=classifier_factory(), **common_cv),
    'T50': T50(classifier=classifier_factory(), **common_cv),
    'MedianSweep': MedianSweep(classifier=classifier_factory(), **common_cv),
    'HDy': HDy(classifier=classifier_factory(), **common_cv),
    'DyS': DyS(classifier=classifier_factory(), **common_cv),
    'FormanMM': FormanMM(classifier=classifier_factory(), **common_cv),
    'CDE': CDE(classifier=classifier_factory(), **common_cv),
  }

  return {name: OneVsRestQuantifier(member) for name, member in binary_members.items()}


def build_all_quantifiers(classifier_name: str,
                          classifier_factory: Callable,
                          cv: int = 10,
                          n_jobs: int = None,
                          parallel_backend: str = "loky") -> dict[str, object]:
  """Builds every quantifier (native multiclass + OvR-wrapped binary)
  for a single base classifier, with names prefixed by the classifier
  (e.g. `'SVM-ACC'`, `'LR-EMQ'`), matching the reporting convention used
  in the original CBMS'26 Table I.

  Parameters
  ----------
  classifier_name : str
    Short name identifying the base classifier (e.g. `'LR'`, `'SVM'`),
    used as a naming prefix only.
  classifier_factory : Callable
    Zero-argument callable returning a fresh classifier instance.
  cv : int, default = 10
    Cross-validation folds for calibrated quantifiers.
  n_jobs : int, default = None
    Forwarded to every quantifier's `n_jobs` parameter.
  parallel_backend : str, default = "loky"
    Forwarded to every quantifier's `parallel_backend` parameter.

  Returns
  -------
  quantifiers : dict[str, BaseQuantifier]
    Maps `'{classifier_name}-{quantifier_name}'` to a fresh, unfitted instance.
  """
  native = build_multiclass_quantifiers(classifier_factory, cv=cv, n_jobs=n_jobs,
                                        parallel_backend=parallel_backend)
  ovr = build_binary_quantifiers_for_ovr(classifier_factory, cv=cv, n_jobs=n_jobs,
                                        parallel_backend=parallel_backend)

  combined = {**native, **ovr}
  return {f'{classifier_name}-{name}': quantifier for name, quantifier in combined.items()}