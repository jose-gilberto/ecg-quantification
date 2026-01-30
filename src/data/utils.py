import numpy as np
import os
from sklearn.model_selection import train_test_split


def split_by_record_id(
  X: np.ndarray,
  y: np.ndarray,
  record_ids: np.ndarray,
  test_size: float = 0.3,
  random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  unique_records = np.unique(record_ids)

  train_records, test_records = train_test_split(
    unique_records, test_size=test_size, random_state=random_state
  )

  train_mask = np.isin(record_ids, train_records)
  test_mask = np.isin(record_ids, test_records)

  X_train, y_train = X[train_mask], y[train_mask]
  X_test, y_test = X[test_mask], y[test_mask]
  rec_train, rec_test = record_ids[train_mask], record_ids[test_mask]

  # TODO: this may check verbosity
  print(f"[INFO] Train patients: {len(train_records)} | Test patients: {len(test_records)}")
  print(f"[INFO] Train samples: {len(X_train)} | Test samples: {len(X_test)}")

  return X_train, X_test, y_train, y_test, rec_train, rec_test


def save_preprocessed_dataset(
  X_train: np.ndarray,
  X_test: np.ndarray,
  y_train: np.ndarray,
  y_test: np.ndarray,
  rec_train: np.ndarray,
  rec_test: np.ndarray,
  output_dir: str="data/processed"
) -> None:
  os.makedirs(output_dir, exist_ok=True)
  output_path = os.path.join(output_dir, "mitdb_preprocessed.npz")

  np.savez_compressed(
    output_path,
    X_train=X_train, X_test=X_test,
    y_train=y_train, y_test=y_test,
    rec_train=rec_train, rec_test=rec_test
  )

  print(f"[INFO] Preprocessed dataset saved to {output_path}")