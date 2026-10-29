"""End-to-end test of `scripts/calibration_study.py` (Fase 5, T34) on a
small synthetic multiclass split with patient ids: uncalibrated outputs
taken from a `run_multiclass_experiment.py` checkpoint (no refit) or fitted,
post-hoc and CalibratedClassifierCV protocols, metrics CSV, figures, and
resuming from the cached outputs."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.filterwarnings(
  "ignore:.*DPP.*:UserWarning",
  "ignore:Solution may be inaccurate.*:UserWarning",
  "ignore:CVXPY SolverError encountered.*:UserWarning",
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = np.array(['N', 'V', 'A', 'L', 'R'])


def _load(script, module_name):
  spec = importlib.util.spec_from_file_location(module_name, os.path.join(REPO, 'scripts', script))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


study = _load('calibration_study.py', 'calibration_study')
experiment = _load('run_multiclass_experiment.py', 'run_multiclass_experiment_for_calibration')


@pytest.fixture
def splits_path(tmp_path):
  def build(seed, n_patients, per_patient):
    rng = np.random.default_rng(seed)
    X, y, ids = [], [], []
    for patient in range(n_patients):
      labels = LABELS[rng.choice(5, per_patient, p=[0.6, 0.1, 0.1, 0.1, 0.1])]
      features = rng.normal(size=(per_patient, 20)) + rng.normal(scale=0.3, size=(1, 20))
      for i, label in enumerate(LABELS):
        features[labels == label, 4 * i:4 * i + 3] += 1.5
      X.append(features)
      y.append(labels)
      ids += [f'p{seed}_{patient}'] * per_patient
    return np.vstack(X), np.concatenate(y), np.array(ids)

  X_train, y_train, ids_train = build(1, 9, 60)
  X_test, y_test, ids_test = build(2, 6, 60)
  path = str(tmp_path / 'splits.npz')
  np.savez_compressed(path, X_train=X_train, y_train=y_train, ids_train=ids_train,
                      X_test=X_test, y_test=y_test, ids_test=ids_test, label_mode='multiclass')
  return path


def _run(module, argv):
  old = sys.argv
  sys.argv = ['script.py', *argv]
  try:
    module.main()
  finally:
    sys.argv = old


def test_calibration_study_end_to_end(tmp_path, splits_path, capsys):
  experiment_dir = str(tmp_path / 'experiment')
  _run(experiment, ['--splits-path', splits_path, '--output-dir', experiment_dir, '--classifiers', 'LR',
                    '--no-include-feature-space', '--cv', '2', '--n-bags', '2', '--bag-sizes', '20'])
  checkpoint_dir = os.path.join(experiment_dir, '_checkpoint')
  capsys.readouterr()

  output_dir = str(tmp_path / 'calibration')
  argv = ['--splits-path', splits_path, '--output-dir', output_dir, '--classifiers', 'LR', 'RF',
          '--experiment-checkpoint', checkpoint_dir, '--cv', '3', '--posthoc-folds', '3', '--n-bins', '10']
  _run(study, argv)
  out = capsys.readouterr().out
  assert "uncalibrated outputs from experiment checkpoint" in out  # LR: taken, not refit
  assert "no fitted RF-CC" in out                                  # RF: fitted here

  results = pd.read_csv(os.path.join(output_dir, 'calibration_metrics.csv'))
  assert len(results) == 2 * (1 + 3 + 3)  # uncalibrated + 3 post-hoc + 3 CalibratedClassifierCV
  assert set(results['protocol']) == {'uncalibrated', 'post-hoc', 'cv'}
  for column in ('accuracy', 'brier', 'log_loss', 'ece', 'ece_equal_mass', 'classwise_ece',
                 *(f'ece_{c}' for c in sorted(LABELS))):
    assert results[column].notna().all(), column

  # the cached uncalibrated LR outputs are the checkpointed LR-CC's
  from ecg_quantification import ExperimentCheckpoint
  from ecg_quantification.calibration import outputs_from_quantifier
  data = np.load(splits_path)
  expected = outputs_from_quantifier(ExperimentCheckpoint(checkpoint_dir).load_fitted('LR-CC'), data['X_test'])
  with np.load(os.path.join(output_dir, 'probabilities', 'LR.npz')) as cached:
    np.testing.assert_allclose(cached['test_proba'], expected['proba'])

  figures = set(os.listdir(os.path.join(output_dir, 'figures')))
  for name in ('LR', 'RF'):
    for protocol in ('post-hoc', 'cv'):
      assert f'reliability_{name}_{protocol}.pdf' in figures
      assert f'reliability_classwise_{name}_{protocol}.png' in figures
  assert 'calibration_summary.pdf' in figures

  # re-running reuses every cached output: nothing is refit
  _run(study, argv)
  out = capsys.readouterr().out
  assert out.count("uncalibrated outputs: cached") == 2
  assert "fitted in" not in out
  pd.testing.assert_frame_equal(pd.read_csv(os.path.join(output_dir, 'calibration_metrics.csv')), results)


def test_deep_classifiers_require_an_experiment_checkpoint(tmp_path, splits_path):
  with pytest.raises(SystemExit, match="experiment-checkpoint"):
    _run(study, ['--splits-path', splits_path, '--output-dir', str(tmp_path / 'c'),
                 '--classifiers', 'LITETime'])


def _logistic_regression():
  from sklearn.linear_model import LogisticRegression
  return LogisticRegression(max_iter=1000)


def _predict_proba(model, X):
  return model.predict_proba(X)


def test_deep_classifier_is_calibrated_post_hoc_only_from_its_checkpoint(tmp_path, splits_path, capsys):
  """Deep classifiers are never refit: their outputs come from the
  checkpointed `<name>-CC` (here a stand-in with the same structure as the
  experiment's -- LabelEncodedQuantifier(CC(CachedFitClassifier(...))),
  predict_proba only), and only the post-hoc protocol applies."""
  from quack.utils.wrappers import SklearnClassifierWrapper
  from quack.quantifiers import CC
  from ecg_quantification import ExperimentCheckpoint
  from ecg_quantification.quantifiers import CachedFitClassifier, LabelEncodedQuantifier

  data = np.load(splits_path)
  probability_only = SklearnClassifierWrapper(
    model_factory=_logistic_regression, predict_proba_fn=_predict_proba, supports_predict_proba=True,
  )
  stand_in = LabelEncodedQuantifier(CC(classifier=CachedFitClassifier(probability_only, cache_id='calib-deep')))
  stand_in.fit(data['X_train'], data['y_train'])
  checkpoint_dir = str(tmp_path / 'deep_checkpoint')
  ExperimentCheckpoint(checkpoint_dir).save_fitted('LITETime-CC', stand_in)

  output_dir = str(tmp_path / 'calibration')
  _run(study, ['--splits-path', splits_path, '--output-dir', output_dir, '--classifiers', 'LITETime',
               '--experiment-checkpoint', checkpoint_dir, '--posthoc-folds', '3'])
  results = pd.read_csv(os.path.join(output_dir, 'calibration_metrics.csv'))
  assert set(results['protocol']) == {'uncalibrated', 'post-hoc'}
  assert len(results) == 1 + 3
  assert "uncalibrated outputs from experiment checkpoint" in capsys.readouterr().out
