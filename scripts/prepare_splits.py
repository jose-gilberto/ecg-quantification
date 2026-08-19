#!/usr/bin/env python
"""Prepares the MIT-BIH beat dataset for experiments: downloads (if
needed), preprocesses into fixed-length segments, splits patient-wise
into train/test with no patient leakage, and persists the resulting
arrays to disk.

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

from ecg_quantification.datasets import MITBIHDownloader
from ecg_quantification.preprocessing import ECGPreprocessor
from ecg_quantification.model_selection import patient_train_test_split, assert_no_patient_leakage


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

  return parser.parse_args()


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
  X, y, record_ids = preprocessor.process_all_records(limit=args.limit_records)
  print(f"Preprocessed {X.shape[0]} segments from {len(np.unique(record_ids))} patients.")

  labels, counts = np.unique(y, return_counts=True)
  print("Class distribution (full dataset):")
  for label, count in zip(labels, counts):
    print(f"  {label}: {count} ({count / len(y):.2%})")

  print(f"\n=== 3/4: Patient-wise train/test split (test_size={args.test_size}) ===")
  X_train, X_test, y_train, y_test, ids_train, ids_test = patient_train_test_split(
    X, y, record_ids,
    test_size=args.test_size,
    stratify=args.stratify,
    random_state=args.random_state,
  )
  assert_no_patient_leakage(ids_train, ids_test)
  print("Leakage check passed: no patient appears in both splits.")

  print(f"Train: {X_train.shape[0]} segments from {len(np.unique(ids_train))} patients")
  print(f"Test:  {X_test.shape[0]} segments from {len(np.unique(ids_test))} patients")

  for split_name, y_split in (('train', y_train), ('test', y_test)):
    split_labels, split_counts = np.unique(y_split, return_counts=True)
    dist = ', '.join(f"{lbl}={cnt / len(y_split):.2%}" for lbl, cnt in zip(split_labels, split_counts))
    print(f"  {split_name} class distribution: {dist}")

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
    random_state=args.random_state if args.random_state is not None else -1,
  )
  print(f"Saved: {output_path}")


if __name__ == '__main__':
  main()