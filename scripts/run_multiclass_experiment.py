#!/usr/bin/env python
"""Runs the multiclass (N, V, A, L, R) quantification experiment under
the Clinically-Constrained Artificial Prevalence Protocol (C-APP).

Loads the patient-wise splits produced by `prepare_splits.py`, trains
every registered quantifier (native multiclass + One-vs-Rest wrapped
binary methods, across every requested base classifier) on the training
split, generates clinically-constrained test bags from the held-out
patients via `ClinicalPrevalenceBagGenerator`, and reports both the
standard Mean Absolute Error and the clinically-weighted
Zone-Crossing Error for every quantifier/classifier/bag-size
combination — mirroring the structure of the CBMS'26 Table I, extended
to the full multiclass, clinically-anchored setting.

Checkpointed and resumable: every fitted quantifier and every
(quantifier, bag_size) evaluation is persisted as soon as it completes
(see `ecg_quantification.ExperimentCheckpoint`). Re-running this exact
command after an interruption (power loss, OOM kill, job timeout) picks
up right where it left off instead of refitting/re-evaluating anything
already done. Use `--fresh` to discard an existing checkpoint and start
over instead.

Usage
-----
python scripts/run_multiclass_experiment.py \
    --splits-path data/processed/splits_multiclass.npz \
    --output-dir results/multiclass_capp \
    --n-bags 200 \
    --bag-sizes 100 200 500 \
    --classifiers LR SVM GB RF \
    --random-state 0

# after an interruption, just run the same command again:
python scripts/run_multiclass_experiment.py \
    --splits-path data/processed/splits_multiclass.npz \
    --output-dir results/multiclass_capp \
    --n-bags 200 --bag-sizes 100 200 500 --classifiers LR SVM GB RF --random-state 0
"""
import argparse
import os
import shutil
import time
import numpy as np

from ecg_quantification import (
  ClinicalPrevalenceBagGenerator,
  CLINICAL_RANGES,
  ZONE_WEIGHTS,
  ZoneCrossingError,
  ExperimentCheckpoint,
  default_base_classifiers,
  build_all_quantifiers,
  build_feature_space_quantifiers,
)
from quack.metrics import ae


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('--splits-path', type=str, required=True,
                       help="Path to the .npz produced by prepare_splits.py (label_mode=multiclass).")
  parser.add_argument('--output-dir', type=str, default='results/multiclass_capp',
                       help="Directory to write the results table(s) to.")
  parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help="Directory for fit/evaluation checkpoints. Defaults to "
                            "'<output-dir>/_checkpoint'.")
  parser.add_argument('--fresh', action='store_true',
                       help="Discard any existing checkpoint under --checkpoint-dir and start over, "
                            "instead of resuming from it.")

  parser.add_argument('--n-bags', type=int, default=200,
                       help="Number of C-APP test bags to generate per bag size.")
  parser.add_argument('--bag-sizes', type=int, nargs='+', default=[100, 200, 500],
                       help="Bag sizes to evaluate, mirroring CBMS'26 Table I.")

  parser.add_argument('--classifiers', type=str, nargs='+', default=['LR', 'SVM', 'GB', 'RF'],
                       choices=['LR', 'SVM', 'GB', 'RF'],
                       help="Base classifiers to evaluate every quantifier with.")
  parser.add_argument('--include-feature-space', action=argparse.BooleanOptionalAction, default=True,
                       help="Also evaluate classifier-free quantifiers (HDx, ReadMe, ED).")

  parser.add_argument('--cv', type=int, default=10,
                       help="Cross-validation folds for calibrated quantifiers.")
  parser.add_argument('--n-jobs', type=int, default=None,
                       help="Parallel jobs forwarded to every quantifier's own n_jobs.")
  parser.add_argument('--parallel-backend', type=str, default='loky', choices=['loky', 'threading'])

  parser.add_argument('--random-state', type=int, default=0)

  return parser.parse_args()


def build_registry(classifier_names: list[str],
                   include_feature_space: bool,
                   cv: int,
                   n_jobs: int,
                   parallel_backend: str) -> dict[str, object]:
  """Assembles the full quantifier registry to evaluate: every
  classifier x quantifier combination, plus classifier-free feature-space
  quantifiers if requested."""
  classifiers = default_base_classifiers()
  registry = {}

  for clf_name in classifier_names:
    registry.update(build_all_quantifiers(
      clf_name, classifiers[clf_name], cv=cv, n_jobs=n_jobs, parallel_backend=parallel_backend,
    ))

  if include_feature_space:
    registry.update(build_feature_space_quantifiers(n_jobs=n_jobs, parallel_backend=parallel_backend))

  return registry


def evaluate_quantifier(name: str,
                        quantifier,
                        bags: list[tuple[np.ndarray, np.ndarray]],
                        true_prevalences: np.ndarray,
                        classes: np.ndarray,
                        zce_metric: ZoneCrossingError) -> list[dict]:
  """Scores every test bag against a single fitted quantifier, returning
  one result row (dict) per bag with both MAE and Zone-Crossing Error."""
  rows = []
  for i, (X_bag, _) in enumerate(bags):
    predicted = quantifier.predict(X_bag)

    # align predicted vector to the full class set: a quantifier fit on
    # the training data already saw every class (patient-wise split
    # preserves all 5 labels in train), so this is a direct comparison
    mae = ae(true_prevalences[i], predicted)
    zce = zce_metric(true_prevalences[i], predicted)

    rows.append({
      'quantifier': name,
      'bag_index': i,
      'mae': mae,
      'zce': zce,
      **{f'true_{c}': true_prevalences[i][j] for j, c in enumerate(classes)},
      **{f'pred_{c}': predicted[j] for j, c in enumerate(classes)},
    })
  return rows


def main() -> None:
  args = parse_args()

  output_dir = args.output_dir
  checkpoint_dir = args.checkpoint_dir or os.path.join(output_dir, '_checkpoint')

  if args.fresh and os.path.exists(checkpoint_dir):
    print(f"=== --fresh given: removing existing checkpoint at {checkpoint_dir} ===")
    shutil.rmtree(checkpoint_dir)

  os.makedirs(output_dir, exist_ok=True)
  checkpoint = ExperimentCheckpoint(checkpoint_dir)

  # only settings that change *what* gets fitted/evaluated go into the
  # run config -- n_jobs/parallel_backend only affect speed, not results,
  # and forcing an exact match on those would make resuming with a
  # different parallelism setting needlessly fail
  run_config = {
    'splits_path': args.splits_path,
    'classifiers': sorted(args.classifiers),
    'include_feature_space': args.include_feature_space,
    'cv': args.cv,
    'random_state': args.random_state,
    'n_bags': args.n_bags,
    'bag_sizes': sorted(args.bag_sizes),
  }
  checkpoint.record_run_config(run_config)

  print(f"=== Loading splits from {args.splits_path} ===")
  data = np.load(args.splits_path, allow_pickle=True)
  X_train, y_train = data['X_train'], data['y_train']
  X_test, y_test, ids_test = data['X_test'], data['y_test'], data['ids_test']
  print(f"Train: {X_train.shape[0]} segments. Test pool: {X_test.shape[0]} segments "
        f"from {len(np.unique(ids_test))} patients.")

  print(f"\n=== Building quantifier registry ===")
  registry = build_registry(
    args.classifiers, args.include_feature_space,
    cv=args.cv, n_jobs=args.n_jobs, parallel_backend=args.parallel_backend,
  )
  print(f"{len(registry)} quantifiers registered: {sorted(registry.keys())}")
  if checkpoint.fitted_names:
    print(f"Checkpoint found: {len(checkpoint.fitted_names)}/{len(registry)} already fitted, will be skipped.")

  # GAC (BaseCalibratedQuantifier, _get_oof_method() == "predict") writes
  # its classifier's raw out-of-fold label predictions into a preallocated
  # array inside quack's fit() -- which breaks on string labels ('N', 'V',
  # ...) with "could not convert string to float". Verified empirically:
  # of every quantifier in the registry, only GAC is affected (the other
  # BaseCalibratedQuantifier subclasses used here -- GPAC, FM, EM -- use
  # _get_oof_method() == "predict_proba", which is numeric regardless of
  # y's dtype). Encoding to integer codes fixes GAC and is a no-op change
  # in behavior for everything else: `all_classes` (sorted, original
  # strings) stays the single source of truth for column naming and
  # ZoneCrossingError, and lines up positionally with the encoded codes
  # by construction (np.searchsorted against the same sorted array both
  # quantifiers and ClinicalPrevalenceBagGenerator ultimately key off of).
  all_classes = np.unique(np.concatenate([y_train, y_test]))
  print(f"\n=== Encoding {len(all_classes)} class labels for fitting: {all_classes.tolist()} ===")
  y_train_encoded = np.searchsorted(all_classes, y_train)

  print(f"\n=== Fitting all quantifiers on training data ===")
  fitted = {}
  for name, quantifier in registry.items():
    if checkpoint.is_fitted(name):
      fitted[name] = checkpoint.load_fitted(name)
      print(f"  {name}: loaded from checkpoint")
      continue

    start = time.time()
    quantifier.fit(X_train, y_train_encoded)
    elapsed = time.time() - start
    checkpoint.save_fitted(name, quantifier)
    fitted[name] = quantifier
    print(f"  {name}: fit in {elapsed:.1f}s")

  classes = all_classes
  zce_metric = ZoneCrossingError(clinical_ranges=CLINICAL_RANGES, classes=classes)

  for bag_size in args.bag_sizes:
    print(f"\n=== C-APP bags: bag_size={bag_size}, n_bags={args.n_bags} ===")
    # regenerated every run (cheap) rather than checkpointed themselves:
    # with a fixed random_state, ClinicalPrevalenceBagGenerator produces
    # byte-identical bags every time, so this stays consistent with
    # whatever was already evaluated and checkpointed for this bag_size
    generator = ClinicalPrevalenceBagGenerator(
      clinical_ranges=CLINICAL_RANGES,
      zone_weights=ZONE_WEIGHTS,
      n_bags=args.n_bags,
      bag_size=bag_size,
      base_label='N',
      random_state=args.random_state,
    )
    bags = generator.to_list(X_test, y_test)
    true_prevalences = generator.sampled_prevalences_

    for name, quantifier in fitted.items():
      if checkpoint.is_evaluated(name, bag_size):
        print(f"  {name}: already evaluated for bag_size={bag_size}, skipping")
        continue

      rows = evaluate_quantifier(name, quantifier, bags, true_prevalences, classes, zce_metric)
      for row in rows:
        row['bag_size'] = bag_size
      checkpoint.save_evaluation(name, bag_size, rows)

      maes = [r['mae'] for r in rows]
      zces = [r['zce'] for r in rows]
      print(f"  {name}: MAE={np.mean(maes):.4f} (±{np.std(maes):.4f})  "
            f"ZCE={np.mean(zces):.4f} (±{np.std(zces):.4f})")

  print(f"\n=== Saving results to {output_dir} ===")
  results_df = checkpoint.load_results()
  detailed_path = os.path.join(output_dir, 'detailed_results.csv')
  results_df.to_csv(detailed_path, index=False)
  print(f"Saved per-bag detail: {detailed_path}")

  summary = (
    results_df.groupby(['quantifier', 'bag_size'])
    .agg(mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
         zce_mean=('zce', 'mean'), zce_std=('zce', 'std'))
    .reset_index()
  )
  summary_path = os.path.join(output_dir, 'summary_table.csv')
  summary.to_csv(summary_path, index=False)
  print(f"Saved summary table: {summary_path}")

  print("\n=== Best quantifier per bag size (by MAE) ===")
  for bag_size in args.bag_sizes:
    subset = summary[summary['bag_size'] == bag_size].sort_values('mae_mean')
    best = subset.iloc[0]
    print(f"  bag_size={bag_size}: {best['quantifier']} "
          f"(MAE={best['mae_mean']:.4f}, ZCE={best['zce_mean']:.4f})")

  print("\n=== Best quantifier per bag size (by ZCE) ===")
  for bag_size in args.bag_sizes:
    subset = summary[summary['bag_size'] == bag_size].sort_values('zce_mean')
    best = subset.iloc[0]
    print(f"  bag_size={bag_size}: {best['quantifier']} "
          f"(ZCE={best['zce_mean']:.4f}, MAE={best['mae_mean']:.4f})")


if __name__ == '__main__':
  main()