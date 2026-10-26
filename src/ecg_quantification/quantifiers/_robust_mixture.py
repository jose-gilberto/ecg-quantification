"""Multiclass-safe mixture quantifiers and feature discretization for the
feature-space quantifiers -- without patching `quack`.

Two problems surfaced in the first full multiclass run (crash while
evaluating HDx: `ValueError: shapes (18506640,5) and (2,) not aligned`):

1. **HDx and ReadMe need discrete features.** Both build their
   conditional matrix from *exact* feature values (HDx: one row per
   distinct value of each feature; ReadMe: one row per distinct feature
   pattern of each random subspace) -- they are meant for categorical or
   discretized data (González-Castro et al., 2013; Hopkins & King, 2010).
   On continuous, z-scored beat samples every training value is distinct
   (360 features x ~51k beats = 18.5M rows for HDx) and a test value
   essentially never matches one exactly, so the test frequency vector is
   all zeros: the problem is degenerate, the convex solver fails, and the
   estimate would be meaningless even if it did not.
   `DiscretizedQuantifier` bins every feature (quantile bins fit on the
   training data, `n_bins=8` by default) before fitting/predicting, which
   is how these methods are used in the quantification literature.

2. **quack's solver fallback is binary-only.** When CVXPY fails,
   `BaseMixtureQuantifier._solve_mixture` falls back to a Golden Section
   Search over `[p, 1 - p]` -- which crashes on any problem with more
   than two classes. That affects every natively multiclass mixture
   quantifier used here (HDx, ReadMe's sub-models, GAC, GPAC, FM), not
   just the degenerate HDx case above. `_MulticlassSafeSolveMixin`
   replaces that step for > 2 classes: retry the same problem with
   another conic solver (SCS), then, if that fails too, minimize quack's
   own distance (`_compute_distance`, valid for any number of classes)
   over the probability simplex with SciPy's SLSQP. Binary problems keep
   quack's behavior unchanged.

The classes below subclass quack's (same names, same parameters, so
`clone()`/`get_params()` and the registry names are unchanged) and are
defined at module level so fitted quantifiers stay picklable for
`ExperimentCheckpoint`.
"""
import warnings

import cvxpy as cvx
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from quack.quantifiers import GAC as _QuackGAC
from quack.quantifiers import GPAC as _QuackGPAC
from quack.quantifiers import FM as _QuackFM
from quack.quantifiers import HDx as _QuackHDx
from quack.quantifiers import ReadMe as _QuackReadMe
from quack.quantifiers._features import _RawSubspaceMixture
from quack.quantifiers.base import BaseQuantifier


# conic solvers tried, in order, after quack's default one fails
_RETRY_SOLVERS = ("SCS",)


def _simplex_minimize(distance, n_classes: int, start: np.ndarray = None) -> np.ndarray:
  """Minimizes `distance(p)` over the probability simplex (SLSQP)."""
  from scipy.optimize import minimize

  x0 = np.full(n_classes, 1.0 / n_classes) if start is None else np.asarray(start, dtype=float)
  result = minimize(
    distance, x0, method="SLSQP",
    bounds=[(0.0, 1.0)] * n_classes,
    constraints=[{"type": "eq", "fun": lambda p: np.sum(p) - 1.0}],
    options={"maxiter": 500, "ftol": 1e-10},
  )
  solution = np.clip(result.x, 0.0, 1.0)
  total = solution.sum()
  return solution / total if total > 0 else np.full(n_classes, 1.0 / n_classes)


class _MulticlassSafeSolveMixin:
  """Overrides `BaseMixtureQuantifier._solve_mixture` so a solver failure
  on a > 2-class problem degrades gracefully instead of reaching quack's
  binary-only Golden Section Search (see the module docstring)."""

  def _solve_mixture(self, test_frequencies: np.ndarray) -> np.ndarray:
    n_classes = self.conditional_matrix_.shape[1]
    if n_classes <= 2:
      return super()._solve_mixture(test_frequencies)

    if self.use_convex_solver:
      try:
        solution = self._solve_via_convex_programming(test_frequencies)
        if solution is not None:
          return np.asarray(solution).squeeze()
      except cvx.SolverError:
        pass
      # the problem (with this bag's parameter values) is cached by
      # quack's _get_convex_problem; re-solve it with another solver
      for solver in _RETRY_SOLVERS:
        if solver not in cvx.installed_solvers():
          continue
        try:
          self._cvx_problem.solve(solver=solver)
          solution = self._cvx_prevalence_var.value
          if solution is not None:
            warnings.warn(
              f"{type(self).__name__}: default convex solver failed; solved with {solver} instead."
            )
            return np.asarray(solution).squeeze()
        except cvx.SolverError:
          continue

    warnings.warn(
      f"{type(self).__name__}: convex solvers failed (or disabled) on a {n_classes}-class "
      f"problem; minimizing the {self.distance_metric} distance over the simplex with SLSQP "
      f"(quack's Golden Section Search fallback is binary-only)."
    )
    return _simplex_minimize(
      lambda p: self._compute_distance(p, test_frequencies),
      n_classes, getattr(self, "train_prevalence_", None),
    )


class HDx(_MulticlassSafeSolveMixin, _QuackHDx):
  """quack's `HDx` with a multiclass-safe solver fallback. Still expects
  discrete features: wrap it in `DiscretizedQuantifier` for continuous X."""


class GAC(_MulticlassSafeSolveMixin, _QuackGAC):
  """quack's `GAC` with a multiclass-safe solver fallback."""


class GPAC(_MulticlassSafeSolveMixin, _QuackGPAC):
  """quack's `GPAC` with a multiclass-safe solver fallback."""


class FM(_MulticlassSafeSolveMixin, _QuackFM):
  """quack's `FM` with a multiclass-safe solver fallback."""


class _SafeRawSubspaceMixture(_MulticlassSafeSolveMixin, _RawSubspaceMixture):
  """ReadMe's internal per-subspace model, with the safe fallback."""


class ReadMe(_QuackReadMe):
  """quack's `ReadMe` whose per-subspace sub-models use the multiclass-safe
  solver fallback. Still expects discrete features: wrap it in
  `DiscretizedQuantifier` for continuous X."""

  def fit(self, X: np.ndarray, y: np.ndarray) -> "ReadMe":
    super().fit(X, y)
    # quack builds the sub-models itself (in joblib workers); upgrade each
    # one in place -- same state, subclass adds only _solve_mixture
    for sub_quantifier in self.sub_quantifiers_:
      sub_quantifier.__class__ = _SafeRawSubspaceMixture
    return self


class DiscretizedQuantifier(BaseQuantifier):
  """Bins every feature before fitting/predicting a feature-space
  quantifier that needs discrete features (`HDx`, `ReadMe`).

  Parameters
  ----------
  quantifier : BaseQuantifier
    Unfitted feature-space quantifier (cloned on `fit`).
  n_bins : int, default = 8
    Bins per feature.
  strategy : {"quantile", "uniform", "kmeans"}, default = "quantile"
    `KBinsDiscretizer` strategy, fit on the training data only. Quantile
    bins are robust to the heavy tails of z-scored ECG samples (uniform
    bins would put almost every sample in the middle bins).

  Attributes
  ----------
  discretizer_ : KBinsDiscretizer
  quantifier_ : BaseQuantifier
    The fitted clone of `quantifier`.
  classes_, n_classes_, train_prevalence_
    Forwarded from `quantifier_`.
  """

  def __init__(self, quantifier: BaseQuantifier = None, n_bins: int = 8, strategy: str = "quantile"):
    super().__init__(classifier=None)
    self.quantifier = quantifier
    self.n_bins = n_bins
    self.strategy = strategy

  def _transform(self, X: np.ndarray) -> np.ndarray:
    # integer bin codes, as float (quack's check_array expects numeric)
    return self.discretizer_.transform(X).astype(np.float64)

  def fit(self, X: np.ndarray, y: np.ndarray) -> "DiscretizedQuantifier":
    X, y = check_X_y(X, y, dtype=None)
    X = np.asarray(X, dtype=np.float64)
    self.discretizer_ = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy=self.strategy)
    with warnings.catch_warnings():
      # features with repeated quantiles (e.g. zero padding) just get
      # fewer bins -- expected, not worth one warning per feature
      warnings.filterwarnings("ignore", message=".*[Bb]ins whose width are too small.*")
      self.discretizer_.fit(X)
    self.quantifier_ = clone(self.quantifier)
    self.quantifier_.fit(self._transform(X), y)
    self.classes_ = self.quantifier_.classes_
    self.n_classes_ = len(self.classes_)
    self.train_prevalence_ = getattr(self.quantifier_, "train_prevalence_", None)
    return self

  def predict(self, X: np.ndarray) -> np.ndarray:
    check_is_fitted(self, "quantifier_")
    X = np.asarray(check_array(X, dtype=None), dtype=np.float64)
    return self.quantifier_.predict(self._transform(X))
