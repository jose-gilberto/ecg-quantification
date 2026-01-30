import numpy as np


def generate_bags(
	X: np.ndarray,
  y: np.ndarray,
  bag_size: int = 500,
  n_bags: int = 200,
  random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
  """ Generate artificial bags using the APP (artificial prevalence protocol)
  for quantification tasks. Each bag contains `bag_size` instances and a prevalence
  for the positive class (y == 1).

  This method only works for bianry quantification.

  Args:
    X (np.ndarray): time series instances.
    y (np.ndarray): each instance label (y = 1 -> healthy).
    bag_size (int, optional): quantity of instances in a bag. Defaults to 500.
    n_bags (int, optional): number of bags that will be generated. Defaults to 200.
    random_state (int, optional): random state to replicability. Defaults to 42.

  Returns:
    bags_X (np.ndarray): list of instances in that bag (shape [bag_size, timesteps])
    bags_prevalences (np.ndarray): list of floats indicating the positive class prevalence
      (proportion of y == 1 in each bag)
  """
  rng = np.random.default_rng(random_state)
  
  X_pos = X[y == 1]
  X_neg = X[y == 0]

  n_pos, n_neg = len(X_pos), len(X_neg)
  print(f'[INFO] Positive samples: {n_pos}, Negative samples: {n_neg}.')

  bags_X, bags_prevalences = [], []
  target_prevalences = rng.uniform(0.0, 1.0, size=n_bags)

  for p in target_prevalences:
    n_pos_bag = int(p * bag_size)
    n_neg_bag = bag_size - n_pos_bag

    pos_idx = rng.choice(n_pos, size=n_pos_bag, replace=True)
    neg_idx = rng.choice(n_neg, size=n_neg_bag, replace=True)

    bag = np.vstack([X_pos[pos_idx], X_neg[neg_idx]])

    rng.shuffle(bag)

    bags_X.append(bag)
    bags_prevalences.append(n_pos_bag / bag_size)

  return bags_X, bags_prevalences
