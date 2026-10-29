"""Portability of experiment artifacts (checkpoints that can be moved to
another machine) and `scripts/export_bundle.py` / `ExportedBundle`: a
bundle unpacked somewhere else must reproduce the run's recorded
per-bag predictions from its fitted quantifiers and bag memberships."""
import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ecg_quantification import APPBagGenerator, ClinicalPrevalenceBagGenerator, CLINICAL_RANGES, ZONE_WEIGHTS
from ecg_quantification import ExperimentCheckpoint

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


exporter = _load('export_bundle.py', 'export_bundle')
experiment = _load('run_multiclass_experiment.py', 'run_multiclass_experiment_for_bundle')


def _run(module, argv):
  old = sys.argv
  sys.argv = ['script.py', *argv]
  try:
    module.main()
  finally:
    sys.argv = old


def _splits(path, seed_shift=0):
  def build(seed, n_patients, per_patient):
    rng = np.random.default_rng(seed + seed_shift)
    X, y, ids = [], [], []
    for patient in range(n_patients):
      labels = LABELS[rng.choice(5, per_patient, p=[0.6, 0.1, 0.1, 0.1, 0.1])]
      features = rng.normal(size=(per_patient, 12))
      for i, label in enumerate(LABELS):
        features[labels == label, 2 * i:2 * i + 2] += 2.0
      X.append(features)
      y.append(labels)
      ids += [f'p{seed}_{patient}'] * per_patient
    return np.vstack(X), np.concatenate(y), np.array(ids)

  X_train, y_train, ids_train = build(1, 6, 50)
  X_test, y_test, ids_test = build(2, 5, 50)
  np.savez_compressed(path, X_train=X_train, y_train=y_train, ids_train=ids_train,
                      X_test=X_test, y_test=y_test, ids_test=ids_test, label_mode='multiclass')
  return path


class TestPortableCheckpoint:

  def _fill(self, directory):
    checkpoint = ExperimentCheckpoint(directory)
    X = np.random.default_rng(0).normal(size=(40, 3))
    y = np.arange(40) % 2
    checkpoint.save_fitted('model', LogisticRegression().fit(X, y))
    checkpoint.save_evaluation('model', 10, [{'quantifier': 'model', 'bag_index': 0, 'mae': 0.1}])
    checkpoint.save_artifact('thing', {'a': 1})
    return X

  def test_moved_directory_still_loads(self, tmp_path):
    X = self._fill(str(tmp_path / 'original'))
    shutil.move(str(tmp_path / 'original'), str(tmp_path / 'moved'))
    checkpoint = ExperimentCheckpoint(str(tmp_path / 'moved'))
    assert checkpoint.load_fitted('model').predict(X).shape == (40,)
    assert len(checkpoint.load_results()) == 1
    assert checkpoint.load_artifact('thing') == {'a': 1}

  def test_manifest_stores_paths_relative_to_the_checkpoint(self, tmp_path):
    self._fill(str(tmp_path / 'ckpt'))
    with open(tmp_path / 'ckpt' / 'manifest.json') as handle:
      manifest = json.load(handle)
    paths = [e['path'] for section in ('fitted', 'evaluated', 'artifacts') for e in manifest[section].values()]
    assert all(not os.path.isabs(p) and not p.startswith('..') for p in paths)

  def test_older_manifests_with_stale_paths_resolve_by_file_name(self, tmp_path):
    X = self._fill(str(tmp_path / 'ckpt'))
    manifest_path = tmp_path / 'ckpt' / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for section in ('fitted', 'evaluated', 'artifacts'):
      for entry in manifest[section].values():  # as an old run recorded them: another machine's paths
        entry['path'] = os.path.join('/home/someone/results/_checkpoint', entry['path'])
    manifest_path.write_text(json.dumps(manifest))
    checkpoint = ExperimentCheckpoint(str(tmp_path / 'ckpt'))
    assert checkpoint.load_fitted('model').predict(X).shape == (40,)
    assert len(checkpoint.load_results()) == 1


class TestBagIndices:

  def test_clinical_generator_records_the_indices_it_yields(self):
    rng = np.random.default_rng(0)
    y = LABELS[rng.choice(5, 500, p=[0.6, 0.1, 0.1, 0.1, 0.1])]
    X = np.arange(500)[:, None].astype(float)
    generator = ClinicalPrevalenceBagGenerator(clinical_ranges=CLINICAL_RANGES, zone_weights=ZONE_WEIGHTS,
                                               n_bags=6, bag_size=40, base_label='N', random_state=3)
    bags = generator.to_list(X, y)
    assert generator.sampled_indices_.shape == (6, 40)
    for (X_bag, y_bag), indices in zip(bags, generator.sampled_indices_):
      np.testing.assert_array_equal(X_bag[:, 0], indices)
      np.testing.assert_array_equal(y_bag, y[indices])

  def test_app_generator_records_the_indices_it_yields(self):
    y = np.array([0, 1] * 100)
    X = np.arange(200)[:, None].astype(float)
    generator = APPBagGenerator(n_bags=4, bag_size=30, positive_label=1, random_state=0)
    for (X_bag, _), indices in zip(generator.to_list(X, y), generator.sampled_indices_):
      np.testing.assert_array_equal(X_bag[:, 0], indices)


@pytest.fixture
def finished_experiment(tmp_path):
  splits = _splits(str(tmp_path / 'splits.npz'))
  experiment_dir = str(tmp_path / 'multiclass_capp')
  _run(experiment, ['--splits-path', splits, '--output-dir', experiment_dir, '--classifiers', 'LR',
                    '--cv', '2', '--n-bags', '4', '--bag-sizes', '30', '60', '--hdx-bins', '4'])
  return splits, experiment_dir


def test_bundle_reproduces_recorded_predictions_after_moving(tmp_path, finished_experiment, capsys):
  splits, experiment_dir = finished_experiment
  bundle_dir = str(tmp_path / 'bundle')
  _run(exporter, ['--splits-path', splits, '--experiment-dir', experiment_dir,
                  '--output-dir', bundle_dir, '--archive'])
  capsys.readouterr()

  # unpack on "another machine": the original run directory is gone
  shutil.rmtree(experiment_dir)
  elsewhere = tmp_path / 'elsewhere'
  elsewhere.mkdir()
  shutil.unpack_archive(str(tmp_path / 'bundle.tar.gz'), str(elsewhere))
  from ecg_quantification.bundle import ExportedBundle
  bundle = ExportedBundle(str(elsewhere / 'bundle'))

  assert bundle.experiments == ['multiclass_capp']
  results = bundle.results()
  X_test = bundle.splits()['X_test']
  for bag_size in (30, 60):
    bags = bundle.bags(bag_size)
    assert bags['indices'].shape == (4, bag_size)
    for name in ('LR-EMQ', 'LR-ACC', 'LR-GAC', 'HDx_4bins', 'ED'):
      quantifier = bundle.load_quantifier(name)
      rows = results[(results['quantifier'] == name) & (results['bag_size'] == bag_size)]
      for _, row in rows.iterrows():
        predicted = quantifier.predict(X_test[bags['indices'][int(row['bag_index'])]])
        recorded = row[[f'pred_{c}' for c in bags['classes']]].to_numpy(float)
        np.testing.assert_allclose(predicted, recorded, atol=1e-8, err_msg=f'{name} bag {row["bag_index"]}')

  # default export: per-beat outputs of LR-CC (test + train); others on demand
  cc = bundle.instance_outputs('LR-CC', 'train', compute=False)
  assert cc['kind'] == 'multiclass' and cc['proba'].shape[1] == 5
  with pytest.raises(KeyError):
    bundle.instance_outputs('LR-ACC', compute=False)
  ovr = bundle.instance_outputs('LR-ACC')
  assert ovr['kind'] == 'one-vs-rest' and ovr['proba'].shape == (len(X_test), 5)
  assert os.path.exists(elsewhere / 'bundle' / 'instances' / 'multiclass_capp' / 'LR-ACC.npz')  # cached

  beats = bundle.bag_instances(30, 2, quantifier='LR-CC')
  assert len(beats) == 30 and {'patient', 'y_true', 'pred', 'p_N'} <= set(beats.columns)
  table = pd.read_csv(elsewhere / 'bundle' / 'instances' / 'classifier_outputs_test.csv')
  assert {'index', 'patient', 'y_true', 'LR_pred', 'LR_p_V'} <= set(table.columns)
  environment = json.loads((elsewhere / 'bundle' / 'environment.json').read_text())
  assert 'scikit-learn' in environment['packages']


def test_export_refuses_a_different_splits_file(tmp_path, finished_experiment):
  _, experiment_dir = finished_experiment
  other = _splits(str(tmp_path / 'other_splits.npz'), seed_shift=10)
  with pytest.raises(SystemExit, match="do not match"):
    _run(exporter, ['--splits-path', other, '--experiment-dir', experiment_dir,
                    '--output-dir', str(tmp_path / 'bundle'), '--instance-outputs', 'none'])
