"""End-to-end test of `scripts/run_multiclass_experiment.py` with
`--classifiers ECGFounder` (Fase 4): registry through the fit cache,
fitting, `ExperimentCheckpoint` persistence/reload and evaluation, using
a fake checkpoint shaped like the official `1_lead_ECGFounder.pth`
(random weights; the real one is a 370 MB download).

Linear-probe mode only: full fine-tuning is covered by
`test_ecg_founder_classifier.py` -- running it per training subset here
would write ~18 x 123 MB of weights for a 5-class, cv=2 toy problem.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


def _load_run_experiment_module():
  repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  script_path = os.path.join(repo_root, 'scripts', 'run_multiclass_experiment.py')
  spec = importlib.util.spec_from_file_location('run_multiclass_experiment_ecgf', script_path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


run_experiment = _load_run_experiment_module()

pytestmark = pytest.mark.filterwarnings(
  "ignore:.*DPP.*:UserWarning",
  "ignore:Solution may be inaccurate.*:UserWarning",
  "ignore:CVXPY SolverError encountered.*:UserWarning",
)

QUANTIFIERS = ['CC', 'PCC', 'GAC', 'GPAC', 'FM', 'EMQ', 'ACC', 'PACC', 'X', 'Max', 'T50',
               'MedianSweep', 'HDy', 'DyS', 'FormanMM', 'CDE']


@pytest.fixture(scope="module")
def fake_checkpoint(tmp_path_factory):
  from ecg_quantification.quantifiers._ecg_founder import _NET1D_ARCHITECTURE, _import_torch
  from ecg_quantification.quantifiers._net1d import Net1D
  _import_torch()  # TensorFlow may be loaded by earlier tests
  torch.manual_seed(0)
  network = Net1D(n_classes=150, **_NET1D_ARCHITECTURE)
  path = str(tmp_path_factory.mktemp("ecgf") / "1_lead_ECGFounder.pth")
  torch.save({"state_dict": network.state_dict(), "val_auroc": np.float64(0.9)}, path)
  return path


@pytest.fixture
def splits_path(tmp_path):
  # all 5 labels: the script's CLINICAL_RANGES need every one of them;
  # 360-sample (1 s at 360 Hz) windows, like prepare_splits.py's default
  labels = np.array(['N', 'V', 'A', 'L', 'R'])

  def build(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(15 * len(labels), 360))
    y = np.repeat(labels, 15)
    for i, label in enumerate(labels):
      X[y == label, 40 + 60 * i: 70 + 60 * i] += 5.0
    order = rng.permutation(len(y))
    return X[order], y[order]

  X_train, y_train = build(1)
  X_test, y_test = build(2)
  path = str(tmp_path / "splits.npz")
  np.savez_compressed(
    path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    ids_test=np.array([f'p{i % 4}' for i in range(len(y_test))]),
    label_mode='multiclass', window_size=360, normalization='zscore',
    length_mode='pad', test_size=0.3, random_state=0,
  )
  return path


def _run(splits_path, output_dir, checkpoint, extra_argv=()):
  argv = [
    'run_multiclass_experiment.py',
    '--splits-path', splits_path, '--output-dir', output_dir,
    '--n-bags', '3', '--bag-sizes', '10',
    '--classifiers', 'ECGFounder', '--no-include-feature-space',
    '--cv', '2', '--random-state', '0',
    '--ecg-founder-checkpoint', checkpoint,
    '--ecg-founder-device', 'cpu',
    '--ecg-founder-batch-size', '64',
    *extra_argv,
  ]
  old_argv = sys.argv
  sys.argv = argv
  try:
    run_experiment.main()
  finally:
    sys.argv = old_argv


def test_ecg_founder_linear_probe_run(tmp_path, splits_path, fake_checkpoint, capsys):
  output_dir = str(tmp_path / "out")
  _run(splits_path, output_dir, fake_checkpoint)
  out = capsys.readouterr().out

  assert "ECGFounder device: CPU" in out
  assert "[1/16] ECGFounder-CC: fitting..." in out
  # cv=2, 5 classes: (2 + 1) x (1 + 5) = 18 real trainings
  assert "training #18 done" in out and "training #19" not in out
  assert "new ECGFounder trainings" in out

  detailed = pd.read_csv(os.path.join(output_dir, 'detailed_results.csv'))
  assert set(detailed['quantifier']) == {f'ECGFounder-{q}' for q in QUANTIFIERS}

  from ecg_quantification import ExperimentCheckpoint
  checkpoint = ExperimentCheckpoint(os.path.join(output_dir, '_checkpoint'))
  assert len(checkpoint.fitted_names) == 16
  # lean pickles: linear-probe models never store the 123 MB backbone
  fitted_dir = os.path.join(output_dir, '_checkpoint')
  sizes = [os.path.getsize(os.path.join(root, f))
           for root, _, files in os.walk(fitted_dir) for f in files if f.endswith('.joblib')]
  assert sizes and max(sizes) < 5_000_000
  data = np.load(splits_path, allow_pickle=True)
  prevalences = checkpoint.load_fitted('ECGFounder-EMQ').predict(data['X_test'])
  np.testing.assert_allclose(prevalences.sum(), 1, atol=1e-6)


def test_cuda_default_fails_fast_without_gpu(tmp_path, splits_path, fake_checkpoint, capsys):
  if torch.cuda.is_available():
    pytest.skip("a CUDA GPU is visible here")
  argv = [
    'run_multiclass_experiment.py', '--splits-path', splits_path,
    '--output-dir', str(tmp_path / "out"), '--classifiers', 'ECGFounder',
    '--ecg-founder-checkpoint', fake_checkpoint,
  ]
  old_argv = sys.argv
  sys.argv = argv
  try:
    with pytest.raises(SystemExit):
      run_experiment.main()
  finally:
    sys.argv = old_argv
  out = capsys.readouterr().out
  assert "ERROR: device='cuda' was requested but no GPU is visible to PyTorch" in out
  assert "fitting..." not in out


def test_deep_classifiers_get_separate_fit_cache_scopes():
  """The fit cache keys on the training data only: LITETime and
  ECGFounder fitting the same subset must never share a scope, or one
  would be served the other's model."""
  registry = run_experiment.build_registry(
    ['LITETime', 'ECGFounder'], include_feature_space=False, cv=2, n_jobs=None,
    parallel_backend='loky', cache_id='run',
  )
  lite = registry['LITETime-CC'].quantifier.classifier
  founder = registry['ECGFounder-CC'].quantifier.classifier
  assert lite.cache_id == 'run::LITETime'
  assert founder.cache_id == 'run::ECGFounder'
