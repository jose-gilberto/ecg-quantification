"""Artificial Prevalence Protocol (APP) bag generator.

Baseline sampling protocol for quantification evaluation: for each bag,
a target positive-class prevalence is drawn uniformly from the *entire*
[0, 1] simplex, with no clinical constraint on which prevalence values
are realistic, and instances are resampled (with replacement, by
default) from the positive/negative pools to hit that target exactly.

This is the binary-only protocol used in the original CBMS'26 study
(see `legacy_cbms26.bags.generate_bags`, kept frozen there for
reproducibility of the published results). It is reintroduced here,
conformed to the same `quack.bag_generator.base.BaseBagGenerator`
contract as `ClinicalPrevalenceBagGenerator`, so both generators can be
swapped interchangeably in experiment scripts -- letting the journal
extension report a direct "APP vs. C-APP" comparison (Reviewer 2's
concern that the full 0-100% prevalence range explored by APP removes
temporal dependencies and may not reflect realistic long-term
monitoring scenarios).
"""
from typing import Generator
import numpy as np
from sklearn.utils import check_random_state
from quack.bag_generator.base import BaseBagGenerator


class APPBagGenerator(BaseBagGenerator):
  """Binary-only quantification bag generator following the Artificial
  Prevalence Protocol (APP): every bag's positive-class prevalence is
  drawn uniformly from the entire [0, 1] simplex, independent of any
  clinical plausibility.

  Parameters
  ----------
  n_bags : int, default = 200
    Number of bags to generate.
  bag_size : int, default = 500
    Number of instances per bag.
  positive_label : default = 1
    The label in `y` treated as the positive class (e.g. `1` for
    "healthy" in `ecg_quantification`'s binary N-vs-rest formulation).
  with_replacement : bool, default = True
    Whether instances are drawn with replacement from each class' pool.
    Forced to True whenever the requested count for a bag exceeds the
    pool size, regardless of this setting.
  random_state : int, RandomState instance or None, default = None
    Controls reproducibility of the prevalence and resampling draws.

  Attributes
  ----------
  classes_ : ndarray of shape (2,)
    The two distinct labels found in `y` the last time `generate` was
    called, in `np.unique` order.
  sampled_prevalences_ : ndarray of shape (n_bags,)
    Realized positive-class prevalence of each generated bag.

  Notes
  -----
  Unlike `ClinicalPrevalenceBagGenerator`, this generator has no notion
  of clinical risk zones and only supports a strictly binary `y`; it
  exists primarily as the "unconstrained" baseline protocol to contrast
  against the clinically-constrained one.

  Examples
  --------
  >>> from ecg_quantification import APPBagGenerator
  >>> generator = APPBagGenerator(n_bags=200, bag_size=500, random_state=42)
  >>> bags = generator.to_list(X, y)  # y in {0, 1}, 1 = healthy
  >>> generator.sampled_prevalences_.shape
  (200,)
  """

  def __init__(self,
               n_bags: int = 200,
               bag_size: int = 500,
               positive_label=1,
               with_replacement: bool = True,
               random_state=None):
    super().__init__(n_bags=n_bags, bag_size=bag_size, random_state=random_state)
    self.positive_label = positive_label
    self.with_replacement = with_replacement

  def generate(self, X: np.ndarray, y: np.ndarray) -> Generator[tuple, None, None]:
    X, y = self._validate(X, y)
    rng = check_random_state(self.random_state)

    self.classes_ = np.unique(y)
    if len(self.classes_) != 2:
      raise ValueError(
        f"APPBagGenerator only supports binary quantification, got "
        f"{len(self.classes_)} classes: {self.classes_.tolist()}."
      )
    if self.positive_label not in self.classes_:
      raise ValueError(f"positive_label={self.positive_label!r} not found in y.")

    pos_idx_pool = np.flatnonzero(y == self.positive_label)
    neg_idx_pool = np.flatnonzero(y != self.positive_label)
    n_pos, n_neg = len(pos_idx_pool), len(neg_idx_pool)
    print(f'[INFO] Positive samples: {n_pos}, Negative samples: {n_neg}.')

    bag_size = self.bag_size if self.bag_size is not None else len(y)
    self.sampled_prevalences_ = np.zeros(self.n_bags)

    target_prevalences = rng.uniform(0.0, 1.0, size=self.n_bags)

    for i, p in enumerate(target_prevalences):
      n_pos_bag = int(p * bag_size)
      n_neg_bag = bag_size - n_pos_bag

      pos_replace = self.with_replacement or n_pos_bag > n_pos
      neg_replace = self.with_replacement or n_neg_bag > n_neg

      pos_idx = rng.choice(pos_idx_pool, size=n_pos_bag, replace=pos_replace)
      neg_idx = rng.choice(neg_idx_pool, size=n_neg_bag, replace=neg_replace)

      bag_indices = np.concatenate([pos_idx, neg_idx])
      rng.shuffle(bag_indices)

      self.sampled_prevalences_[i] = n_pos_bag / bag_size

      yield X[bag_indices], y[bag_indices]
