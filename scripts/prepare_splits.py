#!/usr/bin/env python
"""Prepares the MIT-BIH beat dataset for experiments: downloads (if
needed), preprocesses into fixed-length segments, splits patient-wise
into train/test with no patient leakage, and persists the resulting
arrays to disk.

Preprocessing is checkpointed per-record (see --checkpoint-dir): an
interruption partway through the 48 MIT-BIH records only costs the
record that was in progress, not a full restart. Downloading is already
idempotent on its own (MITBIHDownloader skips re-downloading if a
manifest exists in --data-dir).

Usage
-----
python scripts/prepare_splits.py \
    --data-dir data/mitbih \
    --output-dir data/processed \
    --label-mode multiclass \
    --test-size 0.3 \
    --random-state 0
"""
import argparse
import os
import numpy as np

from ecg_quantification import ExperimentCheckpoint
from ecg_quantification.datasets import MITBIHDownloader
from ecg_quantification.preprocessing import ECGPreprocessor
from ecg_quantification.model_selection import (
  patient_train_test_split,
  guaranteed_patient_train_test_split,
  assert_no_patient_leakage,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('--data-dir', type=str, default='data/mitbih',
                       help="Directory to download/read the MIT-BIH records from.")
  parser.add_argument('--output-dir', type=str, default='data/processed',
                       help="Directory to write the split .npz artifacts to.")

  parser.add_argument('--label-mode', type=str, default='binary', choices=['binary', 'multiclass'],
                       help="'binary': N=1 vs rest=0. 'multiclass': keep only N/V/A/L/R beats.")
  parser.add_argument('--window-size', type=int, default=360,
                       help="Number of samples per extracted beat segment.")
  parser.add_argument('--normalization', type=str, default='zscore', choices=['zscore', 'minmax'])
  parser.add_argument('--length-mode', type=str, default='pad', choices=['pad', 'interpolate'])

  parser.add_argument('--test-size', type=float, default=0.3,
                       help="Approximate fraction of patients assigned to the test split.")
  parser.add_argument('--stratify', action=argparse.BooleanOptionalAction, default=True,
                       help="Balance label distribution across the patient-wise split.")
  parser.add_argument('--random-state', type=int, default=0)

  parser.add_argument('--limit-records', type=int, default=None,
                       help="If set, only process the first N records (useful for quick smoke runs).")
  parser.add_argument('--overwrite-download', action='store_true',
                       help="Force re-download even if the dataset already exists in --data-dir.")

  parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help="Directory to persist per-record preprocessing checkpoints to. "
                            "Defaults to '<data-dir>/.checkpoint_prepare_splits'.")
  parser.add_argument('--use-checkpoint', action=argparse.BooleanOptionalAction, default=True,
                       help="Skip already-preprocessed records on rerun (via --checkpoint-dir). "
                            "Pass --no-use-checkpoint to always reprocess every record from scratch.")
  parser.add_argument('--max-split-attempts', type=int, default=50,
                       help="Only used when --label-mode=multiclass: number of random-state "
                            "retries allowed to find a patient split with every label present "
                            "in both train and test (see guaranteed_patient_train_test_split).")

  return parser.parse_args()


def process_all_records_with_checkpoint(preprocessor: ECGPreprocessor,
                                        checkpoint: ExperimentCheckpoint,
                                        limit: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Same contract as `ECGPreprocessor.process_all_records`, but caches
  every processed record in `checkpoint` (keyed `'record:{name}'`), so a
  rerun after an interruption only reprocesses records not already
  cached -- everything else is loaded straight from disk."""
  all_records = sorted([f.replace('.dat', '') for f in os.listdir(preprocessor.records_dir) if f.endswith('.dat')])
  if limit:
    all_records = all_records[:limit]

  X_all, y_all, record_all = [], [], []
  for rec in all_records:
    artifact_key = f'record:{rec}'
    if checkpoint.has_artifact(artifact_key):
      X, y, record_ids = checkpoint.load_artifact(artifact_key)
      print(f"{rec}: {len(X)} segments loaded from checkpoint.")
    else:
      X, y, record_ids = preprocessor.process_record(rec)
      checkpoint.save_artifact(artifact_key, (X, y, record_ids))

    X_all.append(X)
    y_all.append(y)
    record_all.append(record_ids)

  X_all = np.vstack(X_all)
  y_all = np.concatenate(y_all)
  record_all = np.concatenate(record_all)

  print(f"\nTotal: {len(X_all)} extracted segments from {len(all_records)} records.")
  return X_all, y_all, record_all


def main() -> None:
  args = parse_args()

  print(f"=== 1/4: Downloading dataset (data_dir={args.data_dir}) ===")
  downloader = MITBIHDownloader()
  manifest = downloader.download(args.data_dir, overwrite=args.overwrite_download)
  print(f"Dataset ready: {manifest['n_records']} records available.")

  print(f"\n=== 2/4: Preprocessing (label_mode={args.label_mode}) ===")
  preprocessor = ECGPreprocessor(
    data_dir=args.data_dir,
    window_size=args.window_size,
    normalization=args.normalization,
    length_mode=args.length_mode,
    label_mode=args.label_mode,
  )

  if args.use_checkpoint:
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.data_dir, '.checkpoint_prepare_splits')
    checkpoint = ExperimentCheckpoint(checkpoint_dir)
    # only the parameters that actually change a record's preprocessed
    # output belong here -- mismatching any of them against a previous
    # run under the same checkpoint_dir means cached records would be
    # silently wrong for the new run, so fail loudly instead
    checkpoint.record_run_config({
      'label_mode': args.label_mode,
      'window_size': args.window_size,
      'normalization': args.normalization,
      'length_mode': args.length_mode,
    })
    X, y, record_ids = process_all_records_with_checkpoint(preprocessor, checkpoint, limit=args.limit_records)
  else:
    X, y, record_ids = preprocessor.process_all_records(limit=args.limit_records)

  print(f"Preprocessed {X.shape[0]} segments from {len(np.unique(record_ids))} patients.")

  labels, counts = np.unique(y, return_counts=True)
  print("Class distribution (full dataset):")
  for label, count in zip(labels, counts):
    print(f"  {label}: {count} ({count / len(y):.2%})")

  print(f"\n=== 3/4: Patient-wise train/test split (test_size={args.test_size}) ===")
  if args.label_mode == 'multiclass':
    split = guaranteed_patient_train_test_split(
      X, y, record_ids,
      test_size=args.test_size,
      stratify=args.stratify,
      random_state=args.random_state,
      max_attempts=args.max_split_attempts,
    )
    X_train, X_test, y_train, y_test, ids_train, ids_test, effective_random_state = split
    if effective_random_state != args.random_state:
      print(f"Note: --random-state={args.random_state} didn't give every label in both "
            f"splits; retried and used random_state={effective_random_state} instead. "
            f"Recorded below as the actual reproducible seed for this split.")
  else:
    X_train, X_test, y_train, y_test, ids_train, ids_test = patient_train_test_split(
      X, y, record_ids,
      test_size=args.test_size,
      stratify=args.stratify,
      random_state=args.random_state,
    )
    effective_random_state = args.random_state if args.random_state is not None else -1

  assert_no_patient_leakage(ids_train, ids_test)
  print("Leakage check passed: no patient appears in both splits.")

  print(f"\n=== 4/4: Saving artifacts to {args.output_dir} ===")
  os.makedirs(args.output_dir, exist_ok=True)
  output_path = os.path.join(args.output_dir, f'splits_{args.label_mode}.npz')

  np.savez_compressed(
    output_path,
    X_train=X_train, y_train=y_train, ids_train=ids_train,
    X_test=X_test, y_test=y_test, ids_test=ids_test,
    label_mode=args.label_mode,
    window_size=args.window_size,
    normalization=args.normalization,
    length_mode=args.length_mode,
    test_size=args.test_size,
    random_state=effective_random_state,
  )
  print(f"Saved: {output_path}")


if __name__ == '__main__':
  main()