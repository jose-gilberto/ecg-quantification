"""End-to-end integration test for `scripts/compare_bag_protocols.py`."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest


def _load_module():
  repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  script_path = os.path.join(repo_root, 'scripts', 'compare_bag_protocols.py')
  spec = importlib.util.spec_from_file_location('compare_bag_protocols', script_path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


compare_protocols = _load_module()

pytestmark = pytest.mark.filterwarnings(
  "ignore:.*DPP.*:UserWarning",
  "ignore:Solution may be inaccurate.*:UserWarning",
)


def _make_synthetic_splits(path, random_state=0):
  labels = np.array(['N', 'V', 'A', 'L', 'R'])
  n_classes = len(labels)
  n_per_class = 40

  def build(seed_offset):
    r = np.random.default_rng(random_state + seed_offset)
    X_parts, y_parts = [], []
    for i, label in enumerate(labels):
      center = np.zeros(n_classes)
      center[i] = 6.0
      X_parts.append(r.normal(loc=center, scale=1.0, size=(n_per_class, n_classes)))
      y_parts.append(np.full(n_per_class, label))
    X = np.vstack(X_parts)
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
    'compare_bag_protocols.py',
    '--splits-path', splits_path,
    '--output-dir', output_dir,
    '--checkpoint-dir', checkpoint_dir,
    '--n-bags', '5',
    '--bag-sizes', '20',
    '--classifiers', 'LR',
    '--quantifiers', 'CC', 'ACC', 'EMQ',
    '--cv', '2',
    '--random-state', '0',
    *extra_argv,
  ]
  old_argv = sys.argv
  sys.argv = argv
  try:
    compare_protocols.main()
  finally:
    sys.argv = old_argv


def test_produces_both_protocols_for_every_quantifier(tmp_path, splits_path):
  output_dir = str(tmp_path / "output")
  checkpoint_dir = str(tmp_path / "checkpoint")

  _run(splits_path, output_dir, checkpoint_dir)

  detailed = pd.read_csv(os.path.join(output_dir, 'detailed_results.csv'))
  assert set(detailed['protocol'].unique()) == {'APP', 'C-APP'}
  assert set(detailed['quantifier'].unique()) == {'LR-CC', 'LR-ACC', 'LR-EMQ'}
  # 3 quantifiers x 2 protocols x 5 bags
  assert len(detailed) == 30

  ranking = pd.read_csv(os.path.join(output_dir, 'ranking_comparison.csv'))
  assert set(ranking.columns) >= {'bag_size', 'quantifier', 'app_rank', 'capp_rank', 'rank_shift'}
  assert len(ranking) == 3  # one row per quantifier, single bag_size


def test_burdens_are_valid_probabilities(tmp_path, splits_path):
  output_dir = str(tmp_path / "output")
  checkpoint_dir = str(tmp_path / "checkpoint")

  _run(splits_path, output_dir, checkpoint_dir)

  detailed = pd.read_csv(os.path.join(output_dir, 'detailed_results.csv'))
  assert (detailed['true_burden'] >= 0).all() and (detailed['true_burden'] <= 1).all()
  assert (detailed['pred_burden'] >= 0).all() and (detailed['pred_burden'] <= 1).all()


def test_rerun_reuses_every_checkpointed_unit(tmp_path, splits_path, capsys):
  output_dir = str(tmp_path / "output")
  checkpoint_dir = str(tmp_path / "checkpoint")

  _run(splits_path, output_dir, checkpoint_dir)
  first = pd.read_csv(os.path.join(output_dir, 'detailed_results.csv'))

  capsys.readouterr()
  _run(splits_path, output_dir, checkpoint_dir)
  captured = capsys.readouterr()

  second = pd.read_csv(os.path.join(output_dir, 'detailed_results.csv'))

  assert captured.out.count("loaded from checkpoint") == 3
  assert "already evaluated -- skipping bag generation" in captured.out
  pd.testing.assert_frame_equal(
    first.sort_values(['quantifier', 'protocol', 'bag_index']).reset_index(drop=True),
    second.sort_values(['quantifier', 'protocol', 'bag_index']).reset_index(drop=True),
  )