#!/usr/bin/env python
"""Validates class balance and label coverage of a prepare_splits.py
multiclass output, before committing to a long fit+evaluate run.

Usage
-----
python scripts/validate_multiclass_balance.py --splits-path data/processed/splits_multiclass.npz
"""
import argparse
import numpy as np

from ecg_quantification.preprocessing import MULTICLASS_SYMBOLS


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--splits-path', required=True)
  parser.add_argument('--min-count', type=int, default=30,
                       help="Minimum instances per class per split before warning "
                            "(rule of thumb: comfortably above --cv folds used later).")
  return parser.parse_args()


def report_split(name: str, y: np.ndarray, min_count: int) -> set:
  labels, counts = np.unique(y, return_counts=True)
  total = len(y)
  print(f"\n{name}: {total} beats")
  for label, count in zip(labels, counts):
    flag = "  <-- LOW" if count < min_count else ""
    print(f"  {label}: {count:6d} ({count / total:6.2%}){flag}")

  missing = set(MULTICLASS_SYMBOLS) - set(labels)
  if missing:
    print(f"  WARNING: {name} is missing {sorted(missing)} entirely.")
  return set(labels)


def main() -> None:

  def report_patient_diversity(name: str, y: np.ndarray, record_ids: np.ndarray, min_patients: int = 3) -> None:
    """For each label, reports how many *distinct patients* contribute it
    to this split -- a label can look well-represented by raw beat count
    while actually coming from a single or handful of patients, which is
    a real external-validity risk (a quantifier "learning" that label
    could really just be learning that one patient's signal morphology)."""
    print(f"\n{name}: patient diversity per label")
    labels = np.unique(y)
    for label in labels:
        patients = np.unique(record_ids[y == label])
        flag = "  <-- LOW DIVERSITY" if len(patients) < min_patients else ""
        print(f"  {label}: {len(patients):3d} distinct patients {sorted(patients.tolist())}{flag}")

  args = parse_args()
  data = np.load(args.splits_path, allow_pickle=True)

  if str(data['label_mode']) != 'multiclass':
    raise ValueError(
      f"{args.splits_path} was generated with label_mode="
      f"'{data['label_mode']}', not 'multiclass'. Re-run prepare_splits.py "
      f"with --label-mode multiclass."
    )

  train_labels = report_split('Train', data['y_train'], args.min_count)
  test_labels = report_split('Test', data['y_test'], args.min_count)

  report_patient_diversity('Train', data['y_train'], data['ids_train'])
  report_patient_diversity('Test', data['y_test'], data['ids_test'])

  only_in_train = train_labels - test_labels
  only_in_test = test_labels - train_labels
  if only_in_train:
    print(f"\nWARNING: {sorted(only_in_train)} present in train but absent from test.")
  if only_in_test:
    print(f"\nWARNING: {sorted(only_in_test)} present in test but absent from train -- "
          f"quantifiers fit on train never saw this class; native multiclass fits and "
          f"OneVsRestQuantifier members will be missing an estimator for it entirely, "
          f"and ClinicalPrevalenceBagGenerator will raise when generating test bags "
          f"that need it.")

  if not only_in_train and not only_in_test and set(MULTICLASS_SYMBOLS) <= train_labels:
    print("\nAll good: every MULTICLASS_SYMBOLS label is present in both splits.")


if __name__ == '__main__':
  main()