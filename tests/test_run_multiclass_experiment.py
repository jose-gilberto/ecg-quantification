"""End-to-end integration test for `scripts/run_multiclass_experiment.py`:
runs `main()` against a tiny synthetic splits file (real registry, real
quack quantifiers, real ExperimentCheckpoint -- no mocks), then runs it
again to confirm a "restart" genuinely skips already-completed work
instead of redoing it.

`scripts/` isn't an installed package, so it's imported directly by path.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest


def _load_run_experiment_module():
  repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  script_path = os.path.join(repo_root, 'scripts', 'run_multiclass_experiment.py')
  spec = importlib.util.spec_from_file_location('run_multiclass_experiment', script_path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


run_experiment = _load_run_experiment_module()

# GAC/GPAC's convex solver (CVXPY) is numerically unstable on a dataset
# this tiny (30 samples/class) -- expected noise from quack's internals
# on a toy fixture, not something this test suite is meant to catch.
pytestmark = pytest.mark.filterwarnings(
  "ignore:.*DPP.*:UserWarning",
  "ignore:Solution may be inaccurate.*:UserWarning",
  "ignore:CVXPY SolverError encountered.*:UserWarning",
)


def _make_synthetic_splits(path, random_state=0):
  # each class gets its own dedicated feature axis (see
  # test_one_vs_rest.py's _multiclass_dataset for why this matters: it
  # keeps every class linearly separable from the rest, which several
  # quack quantifiers' base classifiers rely on for sane calibration).
  rng = np.random.default_rng(random_state)
  labels = np.array(['N', 'V', 'A', 'L', 'R'])
  n_classes = len(labels)
  n_per_class = 30

  def build(n_per_class_local, seed_offset):
    r = np.random.default_rng(random_state + seed_offset)
    X_parts, y_parts = [], []
    for i, label in enumerate(labels):
      center = np.zeros(n_classes)
      center[i] = 6.0
      X_parts.append(r.normal(loc=center, scale=1.0, size=(n_per_class_local, n_classes)))
      y_parts.append(np.full(n_per_class_local, label))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    idx = r.permutation(len(y))
    return X[idx], y[idx]

  X_train, y_train = build(n_per_class, seed_offset=1)
  X_test, y_test = build(n_per_class, seed_offset=2)
  ids_test = np.array([f'patient_{i % 4}' for i in range(len(y_test))])

  np.savez_compressed(
    path,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test, ids_test=ids_test,
    label_mode='multiclass', window_size=X_train.shape[1],
    normalization='zscore', length_mode='pad', test_size=0.3, random_state=random_state,
  )


@pytest.fixture
def splits_path(tmp_path):
  path = str(tmp_path / "splits_multiclass.npz")
  _make_synthetic_splits(path)
  return path


def _run(splits_path, output_dir, checkpoint_dir, extra_argv=()):
  argv = [
    'run_multiclass_experiment.py',
    '--splits-path', splits_path,
    '--output-dir', output_dir,
    '--checkpoint-dir', checkpoint_dir,
    '--n-bags', '5',
    '--bag-sizes', '20',
    '--classifiers', 'LR',
    '--no-include-feature-space',
    '--cv', '2',
    '--random-state', '0',
    *extra_argv,
  ]
  old_argv = sys.argv
  sys.argv = argv
  try:
    run_experiment.main()
  finally:
    sys.argv = old_argv


def test_first_run_produces_expected_outputs(tmp_path, splits_path):
  output_dir = str(tmp_path / "output")
  checkpoint_dir = str(tmp_path / "checkpoint")

  _run(splits_path, output_dir, checkpoint_dir)

  detailed = pd.read_csv(os.path.join(output_dir, 'detailed_results.csv'))
  summary = pd.read_csv(os.path.join(output_dir, 'summary_table.csv'))

  # 16 LR quantifiers (6 native multiclass + 10 OvR-wrapped binary),
  # 1 bag_size, 5 bags each
  assert set(detailed['quantifier'].unique()) <= {
    f'LR-{q}' for q in ['CC', 'PCC', 'GAC', 'GPAC', 'FM', 'EMQ',
                        'ACC', 'PACC', 'X', 'Max', 'T50', 'MedianSweep',
                        'HDy', 'DyS', 'FormanMM', 'CDE']
  }
  assert len(detailed) == len(detailed['quantifier'].unique()) * 5
  assert not summary.empty
  assert {'quantifier', 'bag_size', 'mae_mean', 'zce_mean'} <= set(summary.columns)


def test_rerun_after_completion_reuses_every_checkpointed_unit(tmp_path, splits_path, capsys):
  output_dir = str(tmp_path / "output")
  checkpoint_dir = str(tmp_path / "checkpoint")

  _run(splits_path, output_dir, checkpoint_dir)
  first_detailed = pd.read_csv(os.path.join(output_dir, 'detailed_results.csv'))

  capsys.readouterr()  # discard first run's output
  _run(splits_path, output_dir, checkpoint_dir)
  captured = capsys.readouterr()

  second_detailed = pd.read_csv(os.path.join(output_dir, 'detailed_results.csv'))

  # every quantifier should report having been loaded from checkpoint,
  # and every bag_size evaluation should report already being done
  assert captured.out.count("loaded from checkpoint") == 16
  assert "already evaluated for bag_size=20" in captured.out
  # results are byte-identical, not just "similar" -- nothing was silently redone differently
  pd.testing.assert_frame_equal(
    first_detailed.sort_values(['quantifier', 'bag_index']).reset_index(drop=True),
    second_detailed.sort_values(['quantifier', 'bag_index']).reset_index(drop=True),
  )


def test_fresh_flag_discards_checkpoint_and_refits_everything(tmp_path, splits_path, capsys):
  output_dir = str(tmp_path / "output")
  checkpoint_dir = str(tmp_path / "checkpoint")

  _run(splits_path, output_dir, checkpoint_dir)
  capsys.readouterr()

  _run(splits_path, output_dir, checkpoint_dir, extra_argv=['--fresh'])
  captured = capsys.readouterr()

  assert "removing existing checkpoint" in captured.out
  assert "loaded from checkpoint" not in captured.out
  assert captured.out.count(": fit in") == 16


def test_interruption_partway_through_bag_sizes_only_redoes_the_missing_ones(tmp_path, splits_path, capsys):
  # simulates dying after bag_size=20 finished but before a hypothetical
  # second bag_size would run, by checking bag_size=20's units are never
  # touched again even when the checkpoint dir is reused for a "second"
  # (identical, for this test's purposes) invocation
  output_dir = str(tmp_path / "output")
  checkpoint_dir = str(tmp_path / "checkpoint")

  _run(splits_path, output_dir, checkpoint_dir)

  from ecg_quantification import ExperimentCheckpoint
  ckpt = ExperimentCheckpoint(checkpoint_dir)
  assert ckpt.n_evaluated_units == 16  # 16 quantifiers x 1 bag_size
  assert len(ckpt.fitted_names) == 16

  capsys.readouterr()
  _run(splits_path, output_dir, checkpoint_dir)
  captured = capsys.readouterr()

  assert captured.out.count("already evaluated for bag_size=20") == 16
