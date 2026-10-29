#!/usr/bin/env python
"""Calibration study of the base classifiers (Fase 5, T34; Reviewer 2,
point 6): ECE and Brier score with and without calibration, plus
reliability diagrams.

Probability-based quantifiers (PCC, PACC, EMQ, ...) are only as good as
the probabilities they average, so this measures how calibrated each base
classifier is on the held-out patients, and how much sklearn's calibrators
(`CalibratedClassifierCV`: sigmoid, isotonic, temperature) change that.

Protocols (all scored on the same beats: the whole test pool)
------------------------------------------------------------
- `uncalibrated`: the base classifier trained on the full training split,
  as used by the quantifiers. Taken from an experiment checkpoint
  (`--experiment-checkpoint`, the fitted `<classifier>-CC` quantifier) when
  given -- the only option for the deep models -- else refit (classical).
- `post-hoc`: calibration of those *precomputed* test-pool outputs -- no
  retraining, so it applies to LITETime/ECGFounder too. The test pool is
  cross-fitted in `--posthoc-folds` folds grouped by patient: each beat is
  calibrated by a map fit on *other* patients' beats.
- `cv` (classical classifiers, `--cv-classifiers`): sklearn's
  `CalibratedClassifierCV(estimator, method, cv=..., ensemble=True)` fit on
  the training split, with patient-grouped folds (`--cv-groups`). Each
  fold's model is trained once and reused for every method (identical to
  running `CalibratedClassifierCV` per method).

Outputs (`--output-dir`)
-----------------------
- `calibration_metrics.csv`: one row per classifier x protocol x method:
  accuracy, Brier score, log loss, top-label ECE (equal-width and
  equal-mass bins), classwise ECE and per-class ECE.
- `figures/`: reliability diagrams per classifier and protocol (top-label
  and per class) and `calibration_summary` (PDF + PNG).
- `probabilities/<classifier>.npz`: cached classifier outputs (uncalibrated
  test-pool outputs, CV fold outputs). Re-running reuses them, so changing
  bins/methods or resuming after an interruption never refits a model.

Usage
-----
python scripts/calibration_study.py \\
    --splits-path data/processed/splits_multiclass.npz \\
    --output-dir results/calibration \\
    --classifiers LR SVM GB RF LITETime ECGFounder \\
    --experiment-checkpoint results/multiclass_capp/_checkpoint \\
                            results/multiclass_capp_ecgfounder/_checkpoint \\
    --random-state 0 2>&1 | tee calibration.log
"""
import argparse
import datetime
import json
import os
import shutil
import time

# LITETime models loaded from a checkpoint predict through TensorFlow/XLA:
# same environment preparation as run_multiclass_experiment.py (quiet TF
# start-up, Triton GEMM off, XLA pointed at the pip ptxas). Importing
# ecg_quantification does not load TensorFlow.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
if '--xla_gpu_enable_triton_gemm' not in os.environ.get('XLA_FLAGS', ''):
  os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_enable_triton_gemm=false').strip()
from ecg_quantification.quantifiers._lite_time import configure_xla_ptxas  # noqa: E402
configure_xla_ptxas()

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, log_loss  # noqa: E402

from ecg_quantification import ExperimentCheckpoint, default_base_classifiers  # noqa: E402
from ecg_quantification.calibration import (  # noqa: E402
  CALIBRATION_METHODS, calibrate_from_folds, cross_fit_calibration, fold_scores,
  make_splits, outputs_from_quantifier, response_values,
)
from ecg_quantification.metrics import (  # noqa: E402
  brier_score, classwise_calibration_error, expected_calibration_error,
)
from ecg_quantification.visualization import (  # noqa: E402
  plot_calibration_summary, plot_classwise_reliability, plot_reliability_diagram,
)

CLASSICAL = ('LR', 'SVM', 'GB', 'RF')
DEEP = ('LITETime', 'ECGFounder')


def log(message: str = "") -> None:
  if message:
    message = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
  print(message, flush=True)


def format_duration(seconds: float) -> str:
  return f"{seconds:.1f}s" if seconds < 60 else str(datetime.timedelta(seconds=int(round(seconds))))


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--splits-path', type=str, required=True,
                       help="The .npz from prepare_splits.py (label_mode=multiclass).")
  parser.add_argument('--output-dir', type=str, default='results/calibration')
  parser.add_argument('--classifiers', type=str, nargs='+', default=list(CLASSICAL),
                       choices=list(CLASSICAL + DEEP))
  parser.add_argument('--experiment-checkpoint', type=str, nargs='*', default=[],
                       help="Checkpoint dir(s) of run_multiclass_experiment.py runs. The fitted "
                            "'<classifier>-CC' found there provides the uncalibrated classifier "
                            "(required for LITETime/ECGFounder; avoids refitting the classical ones).")
  parser.add_argument('--methods', type=str, nargs='+', default=list(CALIBRATION_METHODS),
                       choices=list(CALIBRATION_METHODS))
  parser.add_argument('--cv-classifiers', type=str, nargs='*', default=list(CLASSICAL),
                       choices=list(CLASSICAL),
                       help="Classifiers also calibrated with CalibratedClassifierCV on the training "
                            "split (refits --cv models each; SVM is by far the slowest). Pass the "
                            "flag with no names to skip this protocol.")
  parser.add_argument('--cv', type=int, default=5, help="Folds for the CalibratedClassifierCV protocol.")
  parser.add_argument('--cv-groups', type=str, default='patient', choices=['patient', 'beat'],
                       help="Group CalibratedClassifierCV folds by training patient (default) or not.")
  parser.add_argument('--posthoc-folds', type=int, default=5,
                       help="Cross-fitting folds for post-hoc calibration on the test pool.")
  parser.add_argument('--posthoc-groups', type=str, default='patient', choices=['patient', 'beat'],
                       help="Group post-hoc folds by test patient (default: every beat is calibrated "
                            "by a map fit on other patients) or not.")
  parser.add_argument('--n-bins', type=int, default=15, help="Bins for ECE / reliability diagrams.")
  parser.add_argument('--random-state', type=int, default=0)
  parser.add_argument('--fresh', action='store_true', help="Discard cached outputs and start over.")
  return parser.parse_args()


# ---------------------------------------------------------------------------
# cached classifier outputs
# ---------------------------------------------------------------------------

def _load_cache(path: str) -> dict:
  if not os.path.exists(path):
    return {}
  with np.load(path, allow_pickle=False) as data:
    return {key: data[key] for key in data.files}


def _save_cache(path: str, cache: dict) -> None:
  tmp = path + '.tmp.npz'
  np.savez_compressed(tmp, **cache)
  os.replace(tmp, path)


def _base_estimator(clf_name: str, random_state: int):
  """A fresh classical base classifier, seeded when it has a random_state
  (RF/GB/SVM's Platt CV) so the study is reproducible."""
  estimator = default_base_classifiers()[clf_name]()
  if 'random_state' in estimator.get_params():
    estimator.set_params(random_state=random_state)
  return estimator


def _find_fitted(name: str, checkpoint_dirs: list[str]):
  for directory in checkpoint_dirs:
    checkpoint = ExperimentCheckpoint(directory)
    if checkpoint.is_fitted(name):
      return checkpoint.load_fitted(name), directory
  return None, None


def _quiet_deep_frameworks() -> None:
  try:
    import keras
    keras.config.disable_interactive_logging()  # no per-call predict progress bars
  except ImportError:
    pass


def uncalibrated_outputs(clf_name, cache, args, X_train, y_train, X_test, classes) -> dict:
  """Full-train classifier outputs on the test pool (cached)."""
  meta = json.dumps({'splits_path': os.path.abspath(args.splits_path), 'n_train': int(len(y_train)),
                     'n_test': int(len(X_test))}, sort_keys=True)
  if 'test_proba' in cache and str(cache.get('test_meta', '')) == meta:
    log(f"  uncalibrated outputs: cached ({cache['source']})")
    return cache
  cache.clear()  # different splits (or nothing cached yet): nothing cached is valid

  start = time.time()
  if clf_name in DEEP:
    _quiet_deep_frameworks()
  quantifier, directory = _find_fitted(f'{clf_name}-CC', args.experiment_checkpoint)
  if quantifier is not None:
    outputs = outputs_from_quantifier(quantifier, X_test)
    source = f'experiment checkpoint {directory} ({clf_name}-CC)'
  elif clf_name in CLASSICAL:
    log(f"  no fitted {clf_name}-CC in --experiment-checkpoint: fitting {clf_name} on the training split")
    model = _base_estimator(clf_name, args.random_state).fit(X_train, y_train)
    scores, response = response_values(model, X_test)
    order = [int(np.flatnonzero(model.classes_ == c)[0]) for c in classes]
    outputs = {
      'classes': classes,
      'proba': model.predict_proba(X_test)[:, order],
      'scores': scores[:, order] if scores.ndim == 2 else scores,
      'response': response,
    }
    source = 'fitted on the training split'
  else:
    raise SystemExit(
      f"ERROR: {clf_name} is only taken from a run_multiclass_experiment.py checkpoint (no "
      f"retraining): pass --experiment-checkpoint <dir> containing a fitted {clf_name}-CC."
    )
  if list(outputs['classes']) != list(classes):
    raise SystemExit(f"ERROR: {clf_name}'s classes {list(outputs['classes'])} != {list(classes)}.")

  cache.update({
    'classes': np.asarray(classes).astype(str),
    'test_proba': outputs['proba'],
    'test_scores': outputs['scores'],
    'response': np.asarray(outputs['response']),
    'source': np.asarray(source),
    'test_meta': np.asarray(meta),
  })
  log(f"  uncalibrated outputs from {source} in {format_duration(time.time() - start)}")
  return cache


def cv_fold_outputs(clf_name, cache, path, args, X_train, y_train, ids_train, X_test) -> list[dict]:
  """CalibratedClassifierCV's per-fold outputs, cached fold by fold."""
  groups = ids_train if (args.cv_groups == 'patient' and ids_train is not None) else None
  meta = json.dumps({'cv': args.cv, 'groups': groups is not None, 'random_state': args.random_state,
                     'n_train': int(len(y_train)), 'n_test': int(len(X_test))}, sort_keys=True)
  if str(cache.get('cv_meta', '')) != meta:
    for key in [k for k in cache if k.startswith('cv_')]:
      del cache[key]
    cache['cv_meta'] = np.asarray(meta)
  splits = make_splits(y_train, args.cv, groups=groups, random_state=args.random_state)
  estimator = _base_estimator(clf_name, args.random_state)

  folds = []
  for k, split in enumerate(splits):
    prefix = f'cv_fold{k}_'
    if prefix + 'cal_idx' not in cache:
      start = time.time()
      fold = fold_scores(estimator, X_train, y_train, [split], X_test)[0]
      cache.update({prefix + key: np.asarray(value) for key, value in fold.items()})
      _save_cache(path, cache)
      missing = sorted(set(np.unique(y_train)) - set(fold['classes'].tolist()))
      note = f"; classes absent from this fold's training part: {missing}" if missing else ""
      log(f"  CalibratedClassifierCV fold {k + 1}/{len(splits)} fitted in "
          f"{format_duration(time.time() - start)}{note}")
    else:
      log(f"  CalibratedClassifierCV fold {k + 1}/{len(splits)}: cached")
    folds.append({key: cache[prefix + key] for key in ('cal_idx', 'classes', 'response', 'cal_scores', 'apply_scores')})
  for fold in folds:
    fold['response'] = str(fold['response'])
  return folds


# ---------------------------------------------------------------------------
# metrics and figures
# ---------------------------------------------------------------------------

def metric_row(clf_name, protocol, method, y_test, proba, classes, n_bins, note="") -> dict:
  per_class = classwise_calibration_error(y_test, proba, classes, n_bins=n_bins, average=False)
  return {
    'classifier': clf_name,
    'protocol': protocol,
    'method': method,
    'accuracy': accuracy_score(y_test, classes[np.argmax(proba, axis=1)]),
    'brier': brier_score(y_test, proba, classes),
    'log_loss': log_loss(y_test, np.clip(proba, 1e-15, 1.0), labels=classes),
    'ece': expected_calibration_error(y_test, proba, classes, n_bins=n_bins),
    'ece_equal_mass': expected_calibration_error(y_test, proba, classes, n_bins=n_bins, strategy='quantile'),
    'classwise_ece': float(np.mean(list(per_class.values()))),
    **{f'ece_{label}': value for label, value in per_class.items()},
    'note': note,
  }


def save_figure(fig, directory: str, stem: str) -> None:
  for extension in ('pdf', 'png'):
    fig.savefig(os.path.join(directory, f'{stem}.{extension}'), dpi=200, bbox_inches='tight')
  plt.close(fig)


def main() -> None:
  args = parse_args()
  run_start = time.time()
  output_dir = args.output_dir
  cache_dir = os.path.join(output_dir, 'probabilities')
  figures_dir = os.path.join(output_dir, 'figures')
  if args.fresh and os.path.exists(cache_dir):
    log(f"--fresh: removing cached outputs at {cache_dir}")
    shutil.rmtree(cache_dir)
  for directory in (cache_dir, figures_dir):
    os.makedirs(directory, exist_ok=True)
  missing_deep = [c for c in args.classifiers if c in DEEP] if not args.experiment_checkpoint else []
  if missing_deep:
    raise SystemExit(f"ERROR: {missing_deep} need --experiment-checkpoint (outputs are not retrained here).")

  log(f"=== Loading splits from {args.splits_path} ===")
  data = np.load(args.splits_path, allow_pickle=True)
  X_train, y_train = data['X_train'], data['y_train'].astype(str)
  X_test, y_test = data['X_test'], data['y_test'].astype(str)
  ids_train = data['ids_train'] if 'ids_train' in data.files else None
  ids_test = data['ids_test'] if 'ids_test' in data.files else None
  classes = np.unique(np.concatenate([y_train, y_test]))
  log(f"Train: {len(y_train)} beats; test pool: {len(y_test)} beats from "
      f"{len(np.unique(ids_test)) if ids_test is not None else '?'} patients; classes {classes.tolist()}")
  if args.cv_groups == 'patient' and ids_train is None:
    log("NOTE: the splits file has no ids_train; CalibratedClassifierCV folds are not patient-grouped.")
  posthoc_groups = ids_test if args.posthoc_groups == 'patient' else None

  with open(os.path.join(output_dir, 'calibration_config.json'), 'w') as handle:
    json.dump({key: value for key, value in vars(args).items()}, handle, indent=2)

  rows = []
  for index, clf_name in enumerate(args.classifiers, start=1):
    log()
    log(f"=== [{index}/{len(args.classifiers)}] {clf_name} ===")
    path = os.path.join(cache_dir, f'{clf_name}.npz')
    cache = _load_cache(path)
    cache = uncalibrated_outputs(clf_name, cache, args, X_train, y_train, X_test, classes)
    _save_cache(path, cache)
    response = str(cache['response'])

    uncalibrated = cache['test_proba']
    rows.append(metric_row(clf_name, 'uncalibrated', 'none', y_test, uncalibrated, classes, args.n_bins))
    log(f"  uncalibrated: ECE={rows[-1]['ece']:.4f}  classwise ECE={rows[-1]['classwise_ece']:.4f}  "
        f"Brier={rows[-1]['brier']:.4f}")

    posthoc = {}
    for method in args.methods:
      calibrated, missing = cross_fit_calibration(
        cache['test_scores'], y_test, classes, method, groups=posthoc_groups,
        n_splits=args.posthoc_folds, random_state=args.random_state, response=response,
      )
      folds_missing = [m for m in missing if m]
      note = (f"calibrated without positives of {sorted(set(sum(folds_missing, [])))} in "
              f"{len(folds_missing)}/{len(missing)} folds") if folds_missing else ""
      posthoc[method] = calibrated
      rows.append(metric_row(clf_name, 'post-hoc', method, y_test, calibrated, classes, args.n_bins, note))
      log(f"  post-hoc {method}: ECE={rows[-1]['ece']:.4f}  classwise ECE={rows[-1]['classwise_ece']:.4f}  "
          f"Brier={rows[-1]['brier']:.4f}" + (f"  ({note})" if note else ""))

    figures = {'post-hoc': posthoc}
    if clf_name in args.cv_classifiers:
      folds = cv_fold_outputs(clf_name, cache, path, args, X_train, y_train, ids_train, X_test)
      _save_cache(path, cache)
      cv_calibrated = {}
      for method in args.methods:
        cv_calibrated[method] = calibrate_from_folds(folds, y_train, classes, method)
        rows.append(metric_row(clf_name, 'cv', method, y_test, cv_calibrated[method], classes, args.n_bins))
        log(f"  CalibratedClassifierCV {method}: ECE={rows[-1]['ece']:.4f}  "
            f"classwise ECE={rows[-1]['classwise_ece']:.4f}  Brier={rows[-1]['brier']:.4f}")
      figures['cv'] = cv_calibrated

    for protocol, calibrated in figures.items():
      label = 'post-hoc (cross-fitted on test patients)' if protocol == 'post-hoc' else 'CalibratedClassifierCV'
      series = {'uncalibrated': uncalibrated, **calibrated}
      fig = plot_reliability_diagram(y_test, series, classes, n_bins=args.n_bins,
                                     title=f"{clf_name} — {label}")
      save_figure(fig, figures_dir, f'reliability_{clf_name}_{protocol}')
      fig = plot_classwise_reliability(y_test, series, classes, n_bins=args.n_bins,
                                       title=f"{clf_name} — {label}, per class")
      save_figure(fig, figures_dir, f'reliability_classwise_{clf_name}_{protocol}')

  results = pd.DataFrame(rows)
  results_path = os.path.join(output_dir, 'calibration_metrics.csv')
  results.to_csv(results_path, index=False)
  save_figure(plot_calibration_summary(results), figures_dir, 'calibration_summary')

  log()
  log(f"=== Saved {results_path} and figures under {figures_dir} ===")
  table = results.pivot_table(index='classifier', columns=['protocol', 'method'], values='ece', sort=False)
  print(table.round(4).to_string(), flush=True)
  log(f"Done in {format_duration(time.time() - run_start)}.")


if __name__ == '__main__':
  main()
