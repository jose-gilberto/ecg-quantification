"""End-to-end integration test for `scripts/run_multiclass_experiment.py`
with `--classifiers ... LITETime` (Fase 3, item 14): confirms the full
CLI path -- registry building via `cached_classifier_factory`, fitting,
`ExperimentCheckpoint.save_fitted`/`load_fitted`, and evaluation -- works
for a LITETime-based run, not just the classic sklearn classifiers
already covered by `test_run_multiclass_experiment.py`.

Kept in its own module (guarded by `pytest.importorskip`) so the
TensorFlow-free classic-classifier integration tests stay fast and
always collected.

This specifically exercises the item 14 fix: `LITETimeClassifier`'s
`model_factory` used to be an unpicklable `lambda`, which would have
made `ExperimentCheckpoint.save_fitted` (`joblib.dump`) fail for every
LITETime-based quantifier the first time this script tried to
checkpoint one -- see `_lite_time.py`'s module docstring, "NOTE on
checkpointing".
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("aeon")


def _load_run_experiment_module():
  repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  script_path = os.path.join(repo_root, 'scripts', 'run_multiclass_experiment.py')
  spec = importlib.util.spec_from_file_location('run_multiclass_experiment_lt', script_path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


run_experiment = _load_run_experiment_module()

pytestmark = pytest.mark.filterwarnings(
  "ignore:.*DPP.*:UserWarning",
  "ignore:Solution may be inaccurate.*:UserWarning",
  "ignore:CVXPY SolverError encountered.*:UserWarning",
  "ignore:Skipping variable loading for optimizer:UserWarning",
)


def _make_synthetic_splits(path, random_state=0, n_classes=5, n_per_class=15, length=16):
  # n_classes defaults to all 5 (not a smaller subset): the script
  # unconditionally uses ecg_quantification's CLINICAL_RANGES/ZONE_WEIGHTS
  # (keyed by all of N, V, A, L, R) to build ClinicalPrevalenceBagGenerator,
  # which raises if any of those labels is absent from y.
  rng = np.random.default_rng(random_state)
  labels = np.array(['N', 'V', 'A', 'L', 'R'])[:n_classes]

  def build(seed_offset):
    r = np.random.default_rng(random_state + seed_offset)
    X_parts, y_parts = [], []
    for i, label in enumerate(labels):
      center = np.zeros(length)
      center[i % length] = 6.0
      X_parts.append(r.normal(loc=center, scale=1.0, size=(n_per_class, length)))
      y_parts.append(np.full(n_per_class, label))
    X = np.vstack(X_parts).astype(np.float64)
    y = np.concatenate(y_parts)
    idx = r.permutation(len(y))
    return X[idx], y[idx]

  X_train, y_train = build(seed_offset=1)
  X_test, y_test = build(seed_offset=2)
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
    '--n-bags', '3',
    '--bag-sizes', '10',
    '--classifiers', 'LITETime',
    '--no-include-feature-space',
    '--cv', '2',
    '--random-state', '0',
    '--lite-time-n-classifiers', '1',
    '--lite-time-n-epochs', '1',
    '--lite-time-batch-size', '8',
    # this suite must run on CPU-only machines/CI; the GPU default and its
    # fail-fast path are covered separately below
    '--lite-time-device', 'cpu',
    *extra_argv,
  ]
  old_argv = sys.argv
  sys.argv = argv
  try:
    run_experiment.main()
  finally:
    sys.argv = old_argv


def test_lite_time_run_fits_checkpoints_and_evaluates(tmp_path, splits_path, capsys):
  output_dir = str(tmp_path / "output")
  checkpoint_dir = str(tmp_path / "checkpoint")

  _run(splits_path, output_dir, checkpoint_dir)
  out = capsys.readouterr().out

  # progress reporting: device, per-quantifier counters before/after each
  # fit, and every real LITETime training (cv=2, 5 classes ->
  # (2 + 1) x (1 + 5) = 18 trainings), each logged once when it starts
  assert "LITETime device: CPU" in out
  assert "[1/16] LITETime-CC: fitting..." in out
  assert "[16/16] LITETime-CDE: fit in" in out
  assert out.count("training #") == 2 * 18  # one 'started' + one 'done' line each
  assert "training #18 done" in out
  assert "training #19" not in out
  assert "new LITETime trainings" in out

  detailed = pd.read_csv(os.path.join(output_dir, 'detailed_results.csv'))
  summary = pd.read_csv(os.path.join(output_dir, 'summary_table.csv'))

  expected_names = {
    f'LITETime-{q}' for q in ['CC', 'PCC', 'GAC', 'GPAC', 'FM', 'EMQ',
                              'ACC', 'PACC', 'X', 'Max', 'T50', 'MedianSweep',
                              'HDy', 'DyS', 'FormanMM', 'CDE']
  }
  assert set(detailed['quantifier'].unique()) <= expected_names
  assert len(detailed) == len(detailed['quantifier'].unique()) * 3
  assert not summary.empty

  from ecg_quantification import ExperimentCheckpoint
  ckpt = ExperimentCheckpoint(checkpoint_dir)
  assert len(ckpt.fitted_names) == 16
  # the item 14 fix, concretely: loading a checkpointed LITETime-based
  # quantifier back and using it must actually work, not just "not
  # raise on save" -- joblib.dump silently producing a broken pickle
  # would only surface here, at load+predict time.
  reloaded = ckpt.load_fitted('LITETime-CC')
  data = np.load(splits_path, allow_pickle=True)
  preds = reloaded.predict(data['X_test'])
  assert preds.shape[0] == len(np.unique(data['y_train']))


def test_lite_time_rerun_reuses_checkpoint_without_refitting(tmp_path, splits_path, capsys):
  output_dir = str(tmp_path / "output")
  checkpoint_dir = str(tmp_path / "checkpoint")

  _run(splits_path, output_dir, checkpoint_dir)
  capsys.readouterr()

  _run(splits_path, output_dir, checkpoint_dir)
  captured = capsys.readouterr()

  assert captured.out.count("loaded from checkpoint") == 16
  assert "already evaluated for bag_size=10" in captured.out


def test_default_gpu_device_fails_fast_without_a_gpu(tmp_path, splits_path, capsys):
  """The default --lite-time-device is 'gpu': on a machine where
  TensorFlow sees no GPU the run must stop before fitting anything,
  instead of silently training for hours on the CPU."""
  import tensorflow as tf
  if tf.config.list_physical_devices("GPU"):
    pytest.skip("a GPU is visible here; the fail-fast path can't be exercised")

  output_dir = str(tmp_path / "output")
  checkpoint_dir = str(tmp_path / "checkpoint")
  argv = [
    'run_multiclass_experiment.py',
    '--splits-path', splits_path, '--output-dir', output_dir,
    '--checkpoint-dir', checkpoint_dir,
    '--classifiers', 'LITETime', '--no-include-feature-space',
    '--n-bags', '3', '--bag-sizes', '10', '--cv', '2',
    '--lite-time-n-classifiers', '1', '--lite-time-n-epochs', '1',
  ]
  old_argv = sys.argv
  sys.argv = argv
  try:
    with pytest.raises(SystemExit):
      run_experiment.main()
  finally:
    sys.argv = old_argv

  out = capsys.readouterr().out
  assert "ERROR: device='gpu' was requested but no GPU is visible" in out
  assert "--lite-time-device cpu" in out
  assert "fitting..." not in out
