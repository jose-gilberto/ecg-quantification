#!/usr/bin/env python
"""Compares the Artificial Prevalence Protocol (APP) against the
Clinically-Constrained APP (C-APP) as bag-generation protocols for
quantifier evaluation, holding the quantifier itself fixed.

Motivation (Reviewer 2, CBMS'26): APP draws each bag's target
prevalence uniformly from the full [0, 1] simplex, removing temporal
dependencies and exploring prevalence ranges that may be clinically
unrealistic. This script isolates the *protocol* as the only varying
factor -- the same fitted quantifier is scored against bags generated
both ways, from the same held-out patients -- to answer, quantitatively,
whether the protocol choice changes which quantifier looks best.

Design
------
Both protocols are reduced to a common binary "arrhythmic burden"
(1 - P(N)) so they're directly comparable:

- APP bags: generated directly in binary (N vs. rest), matching the
  original CBMS'26 framing exactly.
- C-APP bags: generated in the full multiclass space (respecting each
  of V/A/L/R's own CLINICAL_RANGES band independently), then both the
  true and predicted multiclass prevalence vectors are collapsed to the
  same binary burden via `aggregate_prevalence_to_binary` /
  `aggregate_bag_true_prevalences_to_binary`. This is a more clinically
  grounded way to generate a "binary" bag than a naive uniform binary
  draw: the arrhythmic mix inside each bag is realistically composed
  from independently-constrained V/A/L/R bands rather than an
  undifferentiated block of "not-N".

The same 6 quantifiers from the original CBMS'26 study (CC, PCC, ACC,
PACC, EMQ, HDy) are fit once per base classifier directly on
`y_train_binary` -- no OneVsRestQuantifier needed, since the fitting
target is already binary. The *same fitted quantifier instance* is then
scored against both protocols' bags, so any difference in MAE or
ranking is attributable only to the bag-generation protocol.

Usage
-----
python scripts/compare_bag_protocols.py \
    --splits-path data/processed/splits_multiclass.npz \
    --output-dir results/app_vs_capp \
    --n-bags 1000 \
    --bag-sizes 100 200 500 \
    --classifiers LR SVM GB RF \
    --random-state 0
"""
import argparse
import os
import shutil
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from quack.quantifiers import CC, PCC, ACC, PACC, EM, HDy

from ecg_quantification import (
  APPBagGenerator,
  ClinicalPrevalenceBagGenerator,
  CLINICAL_RANGES,
  ZONE_WEIGHTS,
  ExperimentCheckpoint,
  aggregate_prevalence_to_binary,
  aggregate_bag_true_prevalences_to_binary,
  default_base_classifiers,
)

import warnings
warnings.filterwarnings('ignore')

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('--splits-path', type=str, required=True,
                       help="Path to the .npz produced by prepare_splits.py (label_mode=multiclass). "
                            "y is collapsed to binary (N vs. rest) internally -- no separate binary "
                            "splits file needed, and this keeps both protocols evaluated on the exact "
                            "same instances/patients.")
  parser.add_argument('--output-dir', type=str, default='results/app_vs_capp')
  parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help="Defaults to '<output-dir>/_checkpoint'.")
  parser.add_argument('--use-checkpoint', action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument('--fresh', action='store_true',
                       help="Discard any existing checkpoint under --checkpoint-dir before starting.")

  parser.add_argument('--n-bags', type=int, default=200)
  parser.add_argument('--bag-sizes', type=int, nargs='+', default=[100, 200, 500])

  parser.add_argument('--classifiers', type=str, nargs='+', default=['LR', 'SVM', 'GB', 'RF'],
                       choices=['LR', 'SVM', 'GB', 'RF'])
  parser.add_argument('--quantifiers', type=str, nargs='+',
                       default=['CC', 'PCC', 'ACC', 'PACC', 'EMQ', 'HDy'],
                       choices=['CC', 'PCC', 'ACC', 'PACC', 'EMQ', 'HDy'],
                       help="The original CBMS'26 six binary quantifiers. All are natively binary-capable "
                            "in quack (or, for CC/PCC, classifier-agnostic), so no OneVsRestQuantifier "
                            "wrapping is needed here.")
  parser.add_argument('--cv', type=int, default=10)
  parser.add_argument('--n-jobs', type=int, default=None)
  parser.add_argument('--parallel-backend', type=str, default='loky', choices=['loky', 'threading'])
  parser.add_argument('--random-state', type=int, default=0)

  return parser.parse_args()


def build_binary_quantifiers(quantifier_names: list[str],
                             classifier_factory,
                             cv: int,
                             n_jobs: int,
                             parallel_backend: str) -> dict[str, object]:
  """Builds the requested subset of the CBMS'26 six, for one base classifier."""
  common_cv = dict(cv=cv, n_jobs=n_jobs, parallel_backend=parallel_backend)
  factories = {
    'CC': lambda: CC(classifier=classifier_factory()),
    'PCC': lambda: PCC(classifier=classifier_factory()),
    'ACC': lambda: ACC(classifier=classifier_factory(), **common_cv),
    'PACC': lambda: PACC(classifier=classifier_factory(), **common_cv),
    'EMQ': lambda: EM(classifier=classifier_factory(), **common_cv),
    'HDy': lambda: HDy(classifier=classifier_factory(), **common_cv),
  }
  return {name: factories[name]() for name in quantifier_names}


def evaluate_on_bags(quantifier, bags: list, true_burdens: np.ndarray) -> list[dict]:
  """Scores `quantifier` against every bag, comparing its predicted
  binary burden (1 - P(N), index 1 = 'healthy' by the np.unique([0,1])
  convention) against `true_burdens[i]`."""
  rows = []
  for i, (X_bag, _) in enumerate(bags):
    predicted = quantifier.predict(X_bag)
    pred_burden = 1.0 - predicted[1]
    rows.append({
      'bag_index': i,
      'true_burden': float(true_burdens[i]),
      'pred_burden': float(pred_burden),
      'abs_error': float(abs(true_burdens[i] - pred_burden)),
    })
  return rows


def main() -> None:
  args = parse_args()

  checkpoint_dir = args.checkpoint_dir or os.path.join(args.output_dir, '_checkpoint')
  if args.fresh and os.path.exists(checkpoint_dir):
    print(f"--fresh: removing existing checkpoint at {checkpoint_dir}")
    shutil.rmtree(checkpoint_dir)

  print(f"=== Loading splits from {args.splits_path} ===")
  data = np.load(args.splits_path, allow_pickle=True)
  X_train, y_train = data['X_train'], data['y_train']
  X_test, y_test = data['X_test'], data['y_test']
  print(f"Train: {X_train.shape[0]} segments. Test pool: {X_test.shape[0]} segments.")

  if str(data['label_mode']) != 'multiclass':
    raise ValueError(
      f"{args.splits_path} was generated with label_mode='{data['label_mode']}', not 'multiclass'. "
      "This script needs the multiclass labels to build C-APP bags (it derives the binary "
      "N-vs-rest labels internally)."
    )

  y_train_binary = (y_train == 'N').astype(int)
  y_test_binary = (y_test == 'N').astype(int)
  print(f"Binary collapse: train {y_train_binary.mean():.2%} N, test {y_test_binary.mean():.2%} N.")

  checkpoint = None
  if args.use_checkpoint:
    checkpoint = ExperimentCheckpoint(checkpoint_dir)
    checkpoint.record_run_config({
      'splits_path': args.splits_path,
      'classifiers': sorted(args.classifiers),
      'quantifiers': sorted(args.quantifiers),
      'cv': args.cv,
      'random_state': args.random_state,
      'bag_sizes': sorted(args.bag_sizes),
      'n_bags': args.n_bags,
    })
    print(f"Checkpoint dir: {checkpoint_dir}")

  print(f"\n=== Building registry ({len(args.classifiers)} classifiers x {len(args.quantifiers)} quantifiers) ===")
  classifiers = default_base_classifiers()
  registry = {}
  for clf_name in args.classifiers:
    built = build_binary_quantifiers(args.quantifiers, classifiers[clf_name],
                                     cv=args.cv, n_jobs=args.n_jobs, parallel_backend=args.parallel_backend)
    registry.update({f'{clf_name}-{name}': q for name, q in built.items()})
  print(f"{len(registry)} quantifiers registered: {sorted(registry.keys())}")

  print(f"\n=== Fitting all quantifiers on y_train_binary ===")
  fitted = {}
  for name, quantifier in registry.items():
    if checkpoint and checkpoint.is_fitted(name):
      fitted[name] = checkpoint.load_fitted(name)
      print(f"  {name}: loaded from checkpoint")
      continue
    start = time.time()
    quantifier.fit(X_train, y_train_binary)
    elapsed = time.time() - start
    fitted[name] = quantifier
    print(f"  {name}: fit in {elapsed:.1f}s")
    if checkpoint:
      checkpoint.save_fitted(name, quantifier)

  os.makedirs(args.output_dir, exist_ok=True)
  all_results = []

  for bag_size in args.bag_sizes:
    print(f"\n=== bag_size={bag_size}, n_bags={args.n_bags} ===")

    # --- APP protocol ---
    app_key_suffix = '::APP'
    pending_app = [n for n in fitted if not (checkpoint and checkpoint.is_evaluated(n + app_key_suffix, bag_size))]
    if pending_app:
      app_generator = APPBagGenerator(n_bags=args.n_bags, bag_size=bag_size,
                                      positive_label=1, random_state=args.random_state)
      app_bags = app_generator.to_list(X_test, y_test_binary)
      app_true_burdens = 1.0 - app_generator.sampled_prevalences_  # generator tracks P(1)=P(N)

      for name in pending_app:
        rows = evaluate_on_bags(fitted[name], app_bags, app_true_burdens)
        for row in rows:
          row.update(quantifier=name, protocol='APP', bag_size=bag_size)
        if checkpoint:
          checkpoint.save_evaluation(name + app_key_suffix, bag_size, rows)
        else:
          all_results.extend(rows)
        mae = np.mean([r['abs_error'] for r in rows])
        print(f"  [APP]   {name}: MAE={mae:.4f}")
    else:
      print(f"  [APP]   all {len(fitted)} quantifiers already evaluated -- skipping bag generation.")

    # --- C-APP protocol ---
    capp_key_suffix = '::C-APP'
    pending_capp = [n for n in fitted if not (checkpoint and checkpoint.is_evaluated(n + capp_key_suffix, bag_size))]
    if pending_capp:
      capp_generator = ClinicalPrevalenceBagGenerator(
        clinical_ranges=CLINICAL_RANGES, zone_weights=ZONE_WEIGHTS,
        n_bags=args.n_bags, bag_size=bag_size, base_label='N', random_state=args.random_state,
      )
      capp_bags = capp_generator.to_list(X_test, y_test)
      capp_true_burdens = aggregate_bag_true_prevalences_to_binary(
        capp_generator.sampled_prevalences_, capp_generator.classes_, base_label='N',
      )

      for name in pending_capp:
        # the quantifier was fit binary, so .predict(X_bag) always returns a
        # 2-entry vector regardless of which protocol built X_bag
        rows = evaluate_on_bags(fitted[name], capp_bags, capp_true_burdens)
        for row in rows:
          row.update(quantifier=name, protocol='C-APP', bag_size=bag_size)
        if checkpoint:
          checkpoint.save_evaluation(name + capp_key_suffix, bag_size, rows)
        else:
          all_results.extend(rows)
        mae = np.mean([r['abs_error'] for r in rows])
        print(f"  [C-APP] {name}: MAE={mae:.4f}")
    else:
      print(f"  [C-APP] all {len(fitted)} quantifiers already evaluated -- skipping bag generation.")

  print(f"\n=== Saving results to {args.output_dir} ===")
  results_df = checkpoint.load_results() if checkpoint else pd.DataFrame(all_results)
  detailed_path = os.path.join(args.output_dir, 'detailed_results.csv')
  results_df.to_csv(detailed_path, index=False)
  print(f"Saved per-bag detail: {detailed_path}")

  summary = (
    results_df.groupby(['quantifier', 'protocol', 'bag_size'])
    .agg(mae_mean=('abs_error', 'mean'), mae_std=('abs_error', 'std'))
    .reset_index()
  )
  summary_path = os.path.join(args.output_dir, 'summary_table.csv')
  summary.to_csv(summary_path, index=False)
  print(f"Saved summary table: {summary_path}")

  print("\n=== Ranking comparison: does protocol choice change which quantifier looks best? ===")
  ranking_rows = []
  for bag_size in args.bag_sizes:
    subset = summary[summary['bag_size'] == bag_size]
    app_ranked = subset[subset['protocol'] == 'APP'].sort_values('mae_mean')
    capp_ranked = subset[subset['protocol'] == 'C-APP'].sort_values('mae_mean')

    app_rank = {row.quantifier: i + 1 for i, row in enumerate(app_ranked.itertuples())}
    capp_rank = {row.quantifier: i + 1 for i, row in enumerate(capp_ranked.itertuples())}
    common = sorted(set(app_rank) & set(capp_rank))

    app_ranks = [app_rank[q] for q in common]
    capp_ranks = [capp_rank[q] for q in common]
    corr, pvalue = spearmanr(app_ranks, capp_ranks)

    print(f"\n  bag_size={bag_size}: Spearman rank correlation (APP vs C-APP) = {corr:.3f} (p={pvalue:.3f})")
    print(f"    APP best:   {app_ranked.iloc[0]['quantifier']} (MAE={app_ranked.iloc[0]['mae_mean']:.4f})")
    print(f"    C-APP best: {capp_ranked.iloc[0]['quantifier']} (MAE={capp_ranked.iloc[0]['mae_mean']:.4f})")
    if app_ranked.iloc[0]['quantifier'] != capp_ranked.iloc[0]['quantifier']:
      print("    ^ DIFFERENT best quantifier depending on protocol.")

    for q in common:
      ranking_rows.append({
        'bag_size': bag_size, 'quantifier': q,
        'app_rank': app_rank[q], 'capp_rank': capp_rank[q],
        'rank_shift': app_rank[q] - capp_rank[q],
      })

  ranking_df = pd.DataFrame(ranking_rows)
  ranking_path = os.path.join(args.output_dir, 'ranking_comparison.csv')
  ranking_df.to_csv(ranking_path, index=False)
  print(f"\nSaved ranking comparison: {ranking_path}")


if __name__ == '__main__':
  main()