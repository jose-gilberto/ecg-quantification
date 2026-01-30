import os
import numpy as np
import wfdb
from scipy import interpolate
import matplotlib.pyplot as plt
from typing import Literal

NormalizationType = Literal['zscore', 'minmax']
LengthHandlingType = Literal['pad', 'interpolate']


class ECGPreprocessor:
    
  def __init__(self,
               data_dir: str,
               window_size: int = 360,
               normalization: NormalizationType = 'zscore',
               length_mode: LengthHandlingType = 'pad'):
    self.data_dir = data_dir
    self.records_dir = os.path.join(data_dir, 'records')
    self.window_size = window_size
    self.normalization = normalization
    self.length_mode = length_mode

  def load_record(self, record_name: str):
    record_path = os.path.join(self.records_dir, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    return record, annotation
  
  def segment_record(self, record, annotation, record_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = record.p_signal[:, 0] # channel 1
    ann_samples = annotation.sample
    ann_symbols = np.array(annotation.symbol)

    X, y, record_ids = [], [], []
    for i, sample in enumerate(ann_samples):
      start = max(sample - self.window_size // 2, 0)
      end = start + self.window_size
      segment = signal[start:end]

      if len(segment) < self.window_size:
        segment = self._handle_length(segment)

      segment = self._normalize(segment)
      X.append(segment)
      y.append(ann_symbols[i])
      record_ids.append(record_name)

    return np.array(X), np.array(y), np.array(record_ids)
  
  def _normalize(self, segment: np.ndarray) -> np.ndarray:
    if self.normalization == 'zscore':
      mean, std = np.mean(segment), np.std(segment)
      return (segment - mean) / (std + 1e-8)
    elif self.normalization == "minmax":
      min_val, max_val = np.min(segment), np.max(segment)
      return (segment - min_val) / (max_val - min_val + 1e-8)
    else:
      raise ValueError(f"Invalid normalization: {self.normalization}")
    
  def _handle_length(self, segment: np.ndarray) -> np.ndarray:
    if self.length_mode == "pad":
      pad_width = self.window_size - len(segment)
      return np.pad(segment, (0, pad_width), mode="constant")
    elif self.length_mode == "interpolate":
      x_old = np.linspace(0, 1, len(segment))
      x_new = np.linspace(0, 1, self.window_size)
      f = interpolate.interp1d(x_old, segment, kind="linear")
      return f(x_new)
    else:
      raise ValueError(f"Invalid length_mode: {self.length_mode}")
    
  def visualize_segment(self, raw_segment: np.ndarray, processed_segment: np.ndarray):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(raw_segment)
    plt.title("Original segment")
    plt.subplot(1, 2, 2)
    plt.plot(processed_segment)
    plt.title("Processed segment")
    plt.tight_layout()
    plt.show()

  def process_record(self, record_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    record, annotation = self.load_record(record_name)
    X, y, record_ids = self.segment_record(record, annotation, record_name)
    print(f"{record_name}: {len(X)} extracted segments.")
    return X, y, record_ids
  
  def process_all_records(self, limit: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_records = sorted([f.replace(".dat", "") for f in os.listdir(self.records_dir) if f.endswith(".dat")])
    if limit:
      all_records = all_records[:limit]

    X_all, y_all, record_all = [], [], []
    for rec in all_records:
      X, y, record_ids = self.process_record(rec)
      X_all.append(X)
      y_all.append(y)
      record_all.append(record_ids)

    X_all = np.vstack(X_all)

    y_all = np.concatenate(y_all)
    y_all = np.array([1 if s == 'N' else 0 for s in y_all])

    record_all = np.concatenate(record_all)
    print(f"\nTotal: {len(X_all)} extracted segments from {len(all_records)} records.")
    return X_all, y_all, record_all