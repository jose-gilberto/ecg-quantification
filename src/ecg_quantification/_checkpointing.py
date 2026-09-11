"""Checkpoint/resume infrastructure for long-running quantifier
fit + evaluation experiments (e.g. `scripts/run_multiclass_experiment.py`).

Motivation
----------
Running every (base classifier x quantifier) combination across several
bag sizes, on the full multiclass registry, is a long-lived batch job.
An interruption partway through (power loss, OOM kill, a cluster job
timing out) should not force redoing work that already finished.

The classical notion of "resuming mid-training" (checkpointing model
weights at epoch N) does not apply to the classifiers used here
(`LogisticRegression`, `SVC`, `RandomForestClassifier`,
`GradientBoostingClassifier`): their `.fit()` calls are atomic, with no
internal epoch-like state to snapshot. `RandomForestClassifier`/
`GradientBoostingClassifier` support `warm_start=True` (incrementally
adding estimators), but every individual fit here is already cheap
relative to the number of (classifier, quantifier, bag_size)
combinations evaluated -- the real risk of wasted work is at the
*combination* level, not inside any single fit.

`ExperimentCheckpoint` therefore checkpoints at two coarser, still
useful granularities:

- **fit**: one unit per fitted quantifier (`joblib.dump`'d once
  `.fit()` returns).
- **evaluate**: one unit per (quantifier name, bag_size) pair -- every
  bag in that combination is scored and written together.

A genuinely epoch-level checkpoint (for LITETime/ECGFounder training,
once those adapters exist) is a separate, unrelated concern and does
not belong in this module.
"""
import os
import json
import hashlib
from typing import Any

import joblib
import pandas as pd


class ExperimentCheckpoint:
  """Crash-safe checkpoint/resume layer for long quantifier fit and
  bag-evaluation experiments.

  Every write (manifest, fitted quantifier, result fragment) is done
  via write-to-a-temp-path then `os.replace`, so an interruption
  mid-write can never leave a corrupted/partial file behind: either the
  previous valid state remains, or the new one is atomically in place.
  A unit is only marked done in the manifest *after* its artifact is
  safely on disk, so resuming after an interruption only ever redoes
  the one unit that was in progress, never anything already completed.

  Parameters
  ----------
  checkpoint_dir : str
    Directory the checkpoint state is persisted to (manifest, fitted
    quantifiers, per-unit result fragments). Created if missing.

  Attributes
  ----------
  fitted_names : list[str]
    Names of every quantifier already fitted and checkpointed.
  n_evaluated_units : int
    Number of (quantifier, bag_size) units already evaluated.

  Examples
  --------
  >>> from ecg_quantification import ExperimentCheckpoint
  >>> ckpt = ExperimentCheckpoint('results/multiclass_capp/_checkpoint')
  >>> ckpt.record_run_config({'random_state': 0, 'bag_sizes': [100, 200, 500]})
  >>> fitted = {}
  >>> for name, quantifier in registry.items():
  ...     if ckpt.is_fitted(name):
  ...         fitted[name] = ckpt.load_fitted(name)
  ...         continue
  ...     quantifier.fit(X_train, y_train)  # doctest: +SKIP
  ...     ckpt.save_fitted(name, quantifier)
  ...     fitted[name] = quantifier
  >>> for bag_size in [100, 200, 500]:
  ...     for name, quantifier in fitted.items():
  ...         if ckpt.is_evaluated(name, bag_size):
  ...             continue
  ...         rows = evaluate_quantifier(name, quantifier, bags, ...)  # doctest: +SKIP
  ...         ckpt.save_evaluation(name, bag_size, rows)
  >>> results_df = ckpt.load_results()
  """

  def __init__(self, checkpoint_dir: str):
    self.checkpoint_dir = checkpoint_dir
    self.models_dir = os.path.join(checkpoint_dir, 'fitted_quantifiers')
    self.results_dir = os.path.join(checkpoint_dir, 'results')
    self.manifest_path = os.path.join(checkpoint_dir, 'manifest.json')
    self.config_path = os.path.join(checkpoint_dir, 'run_config.json')

    os.makedirs(self.models_dir, exist_ok=True)
    os.makedirs(self.results_dir, exist_ok=True)

    self.manifest = self._load_manifest()

  # ---------- run configuration ----------

  def record_run_config(self, config: dict) -> None:
    """Persists `config` the first time it's called for this checkpoint
    directory; on every subsequent (resumed) call, verifies `config`
    matches exactly what was recorded before.

    Guards against silently mixing partial results generated under
    different settings across an interrupted-and-resumed run: e.g.
    resuming with a different bag-generator `random_state` would
    silently make already-checkpointed bag evaluations inconsistent
    with newly generated bags under the same (name, bag_size) key.

    Parameters
    ----------
    config : dict
      JSON-serializable configuration to persist/verify, e.g.
      `{'random_state': 0, 'bag_sizes': [100, 200, 500], 'n_bags': 200}`.

    Raises
    ------
    ValueError
      If `config` doesn't match the configuration already recorded in
      this checkpoint directory.
    """
    if os.path.exists(self.config_path):
      with open(self.config_path) as f:
        recorded = json.load(f)
      if recorded != config:
        raise ValueError(
          "Run configuration does not match the one recorded in this "
          f"checkpoint directory ({self.config_path}).\n"
          f"Recorded: {recorded}\nGiven:    {config}\n"
          "Resuming with a different configuration risks mixing "
          "inconsistent partial results. Use a fresh checkpoint_dir "
          "for a genuinely different run."
        )
      return

    tmp_path = self.config_path + '.tmp'
    with open(tmp_path, 'w') as f:
      json.dump(config, f, indent=2, sort_keys=True)
    os.replace(tmp_path, self.config_path)

  # ---------- manifest plumbing ----------

  def _load_manifest(self) -> dict:
    if os.path.exists(self.manifest_path):
      with open(self.manifest_path) as f:
        return json.load(f)
    return {'fitted': {}, 'evaluated': {}}

  def _save_manifest(self) -> None:
    tmp_path = self.manifest_path + '.tmp'
    with open(tmp_path, 'w') as f:
      json.dump(self.manifest, f, indent=2)
    os.replace(tmp_path, self.manifest_path)

  @staticmethod
  def _key(*parts: Any) -> str:
    raw = '|'.join(map(str, parts))
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]

  # ---------- fit stage ----------

  def is_fitted(self, name: str) -> bool:
    """Whether a quantifier named `name` was already fitted and
    persisted in this checkpoint directory."""
    return name in self.manifest['fitted']

  def load_fitted(self, name: str):
    """Loads a previously-fitted quantifier by name.

    Parameters
    ----------
    name : str
      Quantifier name previously passed to `save_fitted`.

    Returns
    -------
    quantifier : object
      The fitted quantifier, exactly as it was pickled by
      `save_fitted`.

    Raises
    ------
    KeyError
      If `name` was never checkpointed as fitted (check `is_fitted`
      first).
    """
    if not self.is_fitted(name):
      raise KeyError(f"No fitted checkpoint found for '{name}'.")
    return joblib.load(self.manifest['fitted'][name]['path'])

  def save_fitted(self, name: str, quantifier) -> None:
    """Persists an already-fitted `quantifier` under `name`, and marks
    it as fitted in the manifest. Overwrites any previous checkpoint
    stored under the same `name`.

    Parameters
    ----------
    name : str
      Quantifier name (e.g. `'LR-EMQ'`, matching
      `build_all_quantifiers`'s naming convention).
    quantifier : object
      The fitted quantifier instance to persist (via `joblib.dump`).
    """
    path = os.path.join(self.models_dir, f'{self._key(name)}.joblib')
    tmp_path = path + '.tmp'
    joblib.dump(quantifier, tmp_path)
    os.replace(tmp_path, path)

    self.manifest['fitted'][name] = {'path': path}
    self._save_manifest()

  # ---------- evaluation stage ----------

  def is_evaluated(self, name: str, bag_size: int) -> bool:
    """Whether `name`'s evaluation at `bag_size` was already run and
    persisted in this checkpoint directory."""
    return self._key(name, bag_size) in self.manifest['evaluated']

  def save_evaluation(self, name: str, bag_size: int, rows: list) -> None:
    """Persists the evaluation result rows for one (`name`, `bag_size`)
    unit, and marks it as evaluated in the manifest.

    Parameters
    ----------
    name : str
      Quantifier name this evaluation unit covers.
    bag_size : int
      Bag size this evaluation unit covers.
    rows : list[dict]
      One dict per scored bag (see e.g.
      `scripts/run_multiclass_experiment.py`'s `evaluate_quantifier`).
      Written as a single CSV fragment; every dict should share the
      same keys.
    """
    key = self._key(name, bag_size)
    path = os.path.join(self.results_dir, f'{key}.csv')
    tmp_path = path + '.tmp'

    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)

    self.manifest['evaluated'][key] = {'path': path, 'name': name, 'bag_size': bag_size}
    self._save_manifest()

  def load_results(self) -> pd.DataFrame:
    """Concatenates every persisted evaluation fragment into a single
    DataFrame.

    Returns
    -------
    results : pd.DataFrame
      All checkpointed evaluation rows, concatenated. Empty DataFrame
      if nothing has been evaluated yet.
    """
    fragments = [pd.read_csv(entry['path']) for entry in self.manifest['evaluated'].values()]
    if not fragments:
      return pd.DataFrame()
    return pd.concat(fragments, ignore_index=True)

  # ---------- introspection ----------

  @property
  def fitted_names(self) -> list:
    """Names of every quantifier already fitted and checkpointed."""
    return sorted(self.manifest['fitted'].keys())

  @property
  def n_evaluated_units(self) -> int:
    """Number of (quantifier, bag_size) units already evaluated."""
    return len(self.manifest['evaluated'])
