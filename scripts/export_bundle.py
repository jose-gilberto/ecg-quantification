#!/usr/bin/env python
"""Packs finished experiments into a portable bundle for instance-level
analysis on another machine: the fitted classifiers/quantifiers, every
bag's beat indices, per-beat classifier outputs, the per-bag quantifier
results, and the environment they were produced with.

Read it back with `ecg_quantification.bundle.ExportedBundle` (see the
README.md written into the bundle).

Usage
-----
python scripts/export_bundle.py \\
    --splits-path data/processed/splits_multiclass.npz \\
    --experiment-dir results/multiclass_capp results/multiclass_capp_ecgfounder \\
    --output-dir export/ecg_bundle \\
    --archive

`--experiment-dir` takes the `--output-dir` of `run_multiclass_experiment.py`
runs (containing `_checkpoint/`, `detailed_results.csv`, ...).

Per-beat outputs: by default, for every base classifier, the outputs of
the multiclass classifier inside its `<classifier>-CC` (test and train
splits). `--instance-outputs all` also exports, for every quantifier, the
outputs of the classifier(s) inside *it* (the One-vs-Rest quantifiers have
their own binary classifiers) -- exact, but slow for SVM; the bundle can
also compute any of them later, on demand (`ExportedBundle.instance_outputs`).
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time

# deep models (LITETime) loaded from checkpoints predict through TF/XLA:
# same environment preparation as the experiment scripts
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
if '--xla_gpu_enable_triton_gemm' not in os.environ.get('XLA_FLAGS', ''):
  os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_enable_triton_gemm=false').strip()
from ecg_quantification.quantifiers._lite_time import configure_xla_ptxas  # noqa: E402
configure_xla_ptxas()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ecg_quantification import ExperimentCheckpoint  # noqa: E402
from ecg_quantification._clinical_bag_generator import capp_bag_generator  # noqa: E402
from ecg_quantification.bundle import (  # noqa: E402
  bag_config_key, quantifier_instance_outputs, save_instance_outputs,
)

PACKAGES = ('numpy', 'scipy', 'scikit-learn', 'pandas', 'joblib', 'cvxpy', 'quack', 'aeon',
            'tensorflow', 'tensorflow-cpu', 'keras', 'torch', 'huggingface_hub', 'ecg-quantification')


def log(message: str = "") -> None:
  if message:
    message = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
  print(message, flush=True)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--splits-path', type=str, required=True)
  parser.add_argument('--experiment-dir', type=str, nargs='+', required=True,
                       help="Output dir(s) of run_multiclass_experiment.py runs.")
  parser.add_argument('--output-dir', type=str, required=True, help="Bundle directory to create.")
  parser.add_argument('--include-models', action=argparse.BooleanOptionalAction, default=True,
                       help="Copy the fitted quantifiers (checkpoints).")
  parser.add_argument('--include-splits', action=argparse.BooleanOptionalAction, default=True,
                       help="Copy the train/test splits (needed to recompute outputs or re-run quantifiers).")
  parser.add_argument('--instance-outputs', choices=['cc', 'all', 'none'], default='cc',
                       help="Per-beat outputs to precompute: of each classifier's <clf>-CC (default), "
                            "of every quantifier's own classifier(s), or none.")
  parser.add_argument('--train-outputs', action=argparse.BooleanOptionalAction, default=True,
                       help="Also export in-sample outputs on the training split.")
  parser.add_argument('--calibration-dir', type=str, default=None,
                       help="Optionally include a calibration_study.py output dir.")
  parser.add_argument('--archive', action='store_true', help="Also write <output-dir>.tar.gz.")
  return parser.parse_args()


def _sha256(path: str) -> str:
  digest = hashlib.sha256()
  with open(path, 'rb') as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _git(*args) -> str | None:
  repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  try:
    return subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True, timeout=10,
                          check=True).stdout.strip()
  except (OSError, subprocess.SubprocessError):
    return None


def environment() -> dict:
  versions = {}
  for name in PACKAGES:
    try:
      versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      continue
  quack_source = None
  try:
    quack_source = json.loads(importlib.metadata.distribution('quack').read_text('direct_url.json') or 'null')
  except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
    pass
  return {
    'exported_at': datetime.datetime.now().isoformat(timespec='seconds'),
    'python': sys.version.split()[0],
    'platform': platform.platform(),
    'packages': versions,
    'quack_source': quack_source,
    'ecg_quantification_commit': _git('rev-parse', 'HEAD'),
    'ecg_quantification_uncommitted_changes': bool(_git('status', '--porcelain')),
  }


def _checkpoint_dir(experiment_dir: str) -> str:
  inner = os.path.join(experiment_dir, '_checkpoint')
  if os.path.exists(os.path.join(inner, 'manifest.json')):
    return inner
  if os.path.exists(os.path.join(experiment_dir, 'manifest.json')):
    return experiment_dir
  raise SystemExit(f"ERROR: no experiment checkpoint (manifest.json) found in {experiment_dir}.")


def _copy_checkpoint(source: str, target: str) -> None:
  """Copies a checkpoint directory, skipping temporary files, and rewrites
  its manifest with paths relative to the checkpoint (portable)."""
  shutil.copytree(source, target, ignore=shutil.ignore_patterns('*.tmp', '*.tmp.npz'), dirs_exist_ok=True)
  manifest_path = os.path.join(target, 'manifest.json')
  with open(manifest_path) as handle:
    manifest = json.load(handle)
  subdirs = {'fitted': 'fitted_quantifiers', 'artifacts': 'fitted_quantifiers', 'evaluated': 'results'}
  for section, subdir in subdirs.items():
    for entry in manifest.get(section, {}).values():
      entry['path'] = os.path.join(subdir, os.path.basename(entry['path']))
  with open(manifest_path, 'w') as handle:
    json.dump(manifest, handle, indent=2)


def _directory_size(path: str) -> int:
  return sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(path) for f in files)


def export_bags(run_config: dict, X_test, y_test, results: pd.DataFrame, target_root: str, classes) -> str:
  """Regenerates the run's C-APP bags (same settings, same seed), checks
  them against the run's recorded true prevalences, and saves the beat
  indices of every bag. Returns the bag-config key."""
  key = bag_config_key(run_config['n_bags'], run_config['random_state'])
  directory = os.path.join(target_root, key)
  os.makedirs(directory, exist_ok=True)
  for bag_size in run_config['bag_sizes']:
    path = os.path.join(directory, f'bags_{bag_size}.npz')
    if os.path.exists(path):
      continue
    generator = capp_bag_generator(run_config['n_bags'], bag_size, run_config['random_state'])
    generator.to_list(X_test, y_test)
    recorded = results[results['bag_size'] == bag_size].drop_duplicates('bag_index').sort_values('bag_index')
    true_columns = [f'true_{c}' for c in generator.classes_]
    if len(recorded) and set(true_columns) <= set(recorded.columns):
      if not np.allclose(recorded[true_columns].to_numpy(), generator.sampled_prevalences_[recorded['bag_index']]):
        raise SystemExit(
          f"ERROR: regenerated bags (bag_size={bag_size}) do not match the run's recorded true "
          f"prevalences -- is --splits-path the file this run used?"
        )
    np.savez_compressed(
      path,
      indices=generator.sampled_indices_,
      true_prevalences=generator.sampled_prevalences_,
      classes=np.asarray(generator.classes_).astype(str),
      **{f'zones_{label}': zones for label, zones in generator.sampled_zones_.items()},
    )
    log(f"  bags: bag_size={bag_size}, {run_config['n_bags']} bags -> {os.path.relpath(path, target_root)}")
  return key


def main() -> None:
  args = parse_args()
  start = time.time()
  bundle = args.output_dir
  os.makedirs(bundle, exist_ok=True)

  log(f"=== Loading splits from {args.splits_path} ===")
  data = np.load(args.splits_path, allow_pickle=True)
  X_train, y_train = data['X_train'], data['y_train'].astype(str)
  X_test, y_test = data['X_test'], data['y_test'].astype(str)
  ids_train = data['ids_train'] if 'ids_train' in data.files else None
  ids_test = data['ids_test'] if 'ids_test' in data.files else None
  classes = np.unique(np.concatenate([y_train, y_test]))
  splits_hash = _sha256(args.splits_path)
  if args.include_splits:
    os.makedirs(os.path.join(bundle, 'data'), exist_ok=True)
    shutil.copy2(args.splits_path, os.path.join(bundle, 'data', 'splits.npz'))
    log("  copied splits -> data/splits.npz")

  index = {
    'format': 1,
    'splits_sha256': splits_hash,
    'splits_included': args.include_splits,
    'classes': classes.tolist(),
    'n_train': int(len(y_train)),
    'n_test': int(len(y_test)),
    'experiments': {},
  }

  for experiment_dir in args.experiment_dir:
    name = os.path.basename(os.path.normpath(experiment_dir))
    log()
    log(f"=== Experiment {name} ({experiment_dir}) ===")
    checkpoint_dir = _checkpoint_dir(experiment_dir)
    results_root = experiment_dir if checkpoint_dir != experiment_dir else os.path.dirname(experiment_dir)
    checkpoint = ExperimentCheckpoint(checkpoint_dir)
    target = os.path.join(bundle, 'experiments', name)
    os.makedirs(target, exist_ok=True)

    with open(os.path.join(checkpoint_dir, 'run_config.json')) as handle:
      run_config = json.load(handle)
    shutil.copy2(os.path.join(checkpoint_dir, 'run_config.json'), os.path.join(target, 'run_config.json'))
    results = checkpoint.load_results()
    results.to_csv(os.path.join(target, 'detailed_results.csv'), index=False)
    summary_path = os.path.join(results_root, 'summary_table.csv')
    if os.path.exists(summary_path):
      shutil.copy2(summary_path, os.path.join(target, 'summary_table.csv'))
    log(f"  results: {len(results)} quantifier x bag rows")

    if args.include_models:
      _copy_checkpoint(checkpoint_dir, os.path.join(target, 'checkpoint'))
      size = _directory_size(os.path.join(target, 'checkpoint'))
      log(f"  models: {len(checkpoint.fitted_names)} fitted quantifiers copied ({size / 1e9:.2f} GB)")

    bag_key = export_bags(run_config, X_test, y_test, results, os.path.join(bundle, 'bags'), classes)

    names = checkpoint.fitted_names
    if args.instance_outputs == 'cc':
      wanted = [n for n in names if n.endswith('-CC')]
    elif args.instance_outputs == 'all':
      wanted = names
    else:
      wanted = []
    exported = []
    for i, quantifier_name in enumerate(wanted, start=1):
      t0 = time.time()
      quantifier = checkpoint.load_fitted(quantifier_name)
      test_outputs = quantifier_instance_outputs(quantifier, X_test)
      if test_outputs is None:
        log(f"  [{i}/{len(wanted)}] {quantifier_name}: no classifier (feature-space quantifier), skipped")
        continue
      by_split = {'test': test_outputs}
      if args.train_outputs:
        by_split['train'] = quantifier_instance_outputs(quantifier, X_train)
      save_instance_outputs(os.path.join(bundle, 'instances', name, f'{quantifier_name}.npz'), by_split)
      exported.append(quantifier_name)
      log(f"  [{i}/{len(wanted)}] {quantifier_name}: per-beat outputs ({', '.join(by_split)}) "
          f"in {time.time() - t0:.1f}s")

    index['experiments'][name] = {
      'run_config': run_config,
      'bag_config': bag_key,
      'quantifiers': names,
      'models_included': args.include_models,
      'instance_outputs': exported,
    }

  # one flat table per split with every <clf>-CC's outputs, for quick looks
  for split, labels, ids in (('test', y_test, ids_test), ('train', y_train, ids_train)):
    if split == 'train' and not args.train_outputs:
      continue
    table = pd.DataFrame({'index': np.arange(len(labels)), 'patient': ids, 'y_true': labels})
    for name, entry in index['experiments'].items():
      for quantifier_name in entry['instance_outputs']:
        if not quantifier_name.endswith('-CC'):
          continue
        clf = quantifier_name[:-3]
        with np.load(os.path.join(bundle, 'instances', name, f'{quantifier_name}.npz')) as outputs:
          table[f'{clf}_pred'] = outputs[f'{split}_pred']
          for column, label in enumerate(outputs['classes']):
            table[f'{clf}_p_{label}'] = outputs[f'{split}_proba'][:, column]
    if table.shape[1] > 3:
      table.to_csv(os.path.join(bundle, 'instances', f'classifier_outputs_{split}.csv'), index=False)

  if args.calibration_dir:
    shutil.copytree(args.calibration_dir, os.path.join(bundle, 'calibration'), dirs_exist_ok=True)
    log("  copied calibration study -> calibration/")

  with open(os.path.join(bundle, 'bundle.json'), 'w') as handle:
    json.dump(index, handle, indent=2)
  with open(os.path.join(bundle, 'environment.json'), 'w') as handle:
    json.dump(environment(), handle, indent=2)
  with open(os.path.join(bundle, 'README.md'), 'w') as handle:
    handle.write(bundle_readme(index, os.path.basename(os.path.normpath(bundle))))

  log()
  log(f"=== Bundle written to {bundle} ({_directory_size(bundle) / 1e9:.2f} GB) ===")
  if args.archive:
    archive = shutil.make_archive(os.path.normpath(bundle), 'gztar', root_dir=os.path.dirname(os.path.abspath(bundle)),
                                  base_dir=os.path.basename(os.path.normpath(bundle)))
    log(f"Archive: {archive} ({os.path.getsize(archive) / 1e9:.2f} GB)")
  log(f"Done in {datetime.timedelta(seconds=int(time.time() - start))}.")


def bundle_readme(index: dict, bundle_name: str) -> str:
  experiments = '\n'.join(
    f"- `{name}`: {len(entry['quantifiers'])} fitted quantifiers; bags `{entry['bag_config']}` "
    f"(sizes {entry['run_config']['bag_sizes']}); per-beat outputs exported for "
    f"{len(entry['instance_outputs'])} quantifiers"
    for name, entry in index['experiments'].items()
  )
  return f"""# ECG quantification -- experiment bundle

Fitted classifiers/quantifiers, bag memberships and per-beat outputs of
the multiclass C-APP experiments, for instance-level analysis.

Classes: {', '.join(index['classes'])}. Train beats: {index['n_train']}; test beats: {index['n_test']}.
Splits included: {index['splits_included']} (sha256 `{index['splits_sha256'][:16]}...`).

{experiments}

## Setup

Install `ecg-quantification` at the commit in `environment.json`
(`ecg_quantification_commit`), with the same `quack` (`quack_source`) and,
ideally, the same scikit-learn version -- fitted models are pickles.

- LITETime models need the `[deep]` extra (TensorFlow); a GPU is not needed
  to predict.
- ECGFounder models need the `[ecgfounder]` extra (PyTorch). They do not
  contain the pretrained backbone: it is downloaded from Hugging Face on
  first use (or set `ECGFOUNDER_CHECKPOINT=/path/to/1_lead_ECGFounder.pth`).
  Without a GPU, set `ECGFOUNDER_DEVICE=cpu`.

## Reading it

```python
from ecg_quantification.bundle import ExportedBundle

bundle = ExportedBundle("{bundle_name}")  # path to this folder
results = bundle.results()        # one row per quantifier x bag: true_*/pred_* prevalences, mae, zce
bags = bundle.bags(100)           # indices (n_bags x 100) into the test split, true_prevalences, zones_*
beats = bundle.bag_instances(100, bag_index=0, quantifier="LR-CC")  # one bag's beats + classifier outputs
outputs = bundle.instance_outputs("LR-CC")          # per-beat proba/pred on the test split
outputs = bundle.instance_outputs("LR-ACC")         # computed on demand if not exported
quantifier = bundle.load_quantifier("LR-EMQ")       # fitted model: quantifier.predict(X_bag)
splits = bundle.splits()                            # X_train, y_train, ids_*, X_test, y_test
```

Re-scoring a modified bag (e.g. without some beats):

```python
X_test = splits["X_test"]
idx = bags["indices"][0]
keep = idx[beats["y_true"].values != "V"]           # any instance-level edit
print(quantifier.predict(X_test[idx]), quantifier.predict(X_test[keep]))
```

## Files

- `experiments/<name>/detailed_results.csv`: per-bag predictions and errors.
- `experiments/<name>/checkpoint/`: fitted quantifiers (`ExperimentCheckpoint`).
- `bags/<config>/bags_<size>.npz`: bag membership (regenerated with the run's
  seed and checked against its recorded true prevalences).
- `instances/<experiment>/<quantifier>.npz`: per-beat outputs
  (`test_proba`, `test_pred`, `train_proba`, `train_pred`, `classes`, `kind`).
  `kind = one-vs-rest`: column k is P(class k vs. rest) from that class's own
  binary classifier (rows do not sum to 1).
- `instances/classifier_outputs_<split>.csv`: every `<clf>-CC` side by side.
- `environment.json`: versions and commits used.
"""


if __name__ == '__main__':
  main()
