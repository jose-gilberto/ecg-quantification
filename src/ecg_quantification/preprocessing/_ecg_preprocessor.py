import os
from typing import Literal
import numpy as np
import wfdb
from scipy import interpolate

NormalizationType = Literal['zscore', 'minmax']
LengthHandlingType = Literal['pad', 'interpolate']
LabelModeType = Literal['binary', 'multiclass']

# beat symbols kept for the multiclass problem (N + the clinically-relevant
# V/A/L/R labels used by ecg_quantification's prevalence protocol)
MULTICLASS_SYMBOLS: tuple[str, ...] = ('N', 'V', 'A', 'L', 'R')


class ECGPreprocessor:
  """Loads, segments, and normalizes WFDB ECG records into fixed-length
  beat-centered windows, ready for classification/quantification.

  Parameters
  ----------
  data_dir : str
    Dataset directory containing a `records/` subfolder with `.dat`/
    `.hea`/`.atr` files (e.g. the output of `MITBIHDownloader.download`).
  window_size : int, default = 360
    Number of samples per extracted beat segment.
  normalization : {'zscore', 'minmax'}, default = 'zscore'
    Per-segment normalization strategy.
  length_mode : {'pad', 'interpolate'}, default = 'pad'
    Strategy used when a segment is shorter than `window_size` (e.g. a
    beat annotated near the start of the recording).
  label_mode : {'binary', 'multiclass'}, default = 'binary'
    - `'binary'`: every beat is kept; label is `1` for `'N'` (normal) and
      `0` for anything else.
    - `'multiclass'`: only beats whose symbol is in `MULTICLASS_SYMBOLS`
      (`N`, `V`, `A`, `L`, `R`) are kept; label is the raw symbol string,
      matching the labels expected by `ClinicalPrevalenceBagGenerator`.
  channel : int, default = 0
    Index of the signal channel extracted from `record.p_signal`.

  Examples
  --------
  >>> from ecg_quantification.preprocessing import ECGPreprocessor
  >>> preprocessor = ECGPreprocessor('data/mitbih', label_mode='multiclass')
  >>> X, y, record_ids = preprocessor.process_all_records()
  """

  def __init__(self,
               data_dir: str,
               window_size: int = 360,
               normalization: NormalizationType = 'zscore',
               length_mode: LengthHandlingType = 'pad',
               label_mode: LabelModeType = 'binary',
               channel: int = 0):
    self.data_dir = data_dir
    self.records_dir = os.path.join(data_dir, 'records')
    self.window_size = window_size
    self.normalization = normalization
    self.length_mode = length_mode
    self.label_mode = label_mode
    self.channel = channel

  def load_record(self, record_name: str):
    """Reads a WFDB record and its beat annotations from `records_dir`."""
    record_path = os.path.join(self.records_dir, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    return record, annotation

  def _keep_mask(self, ann_symbols: np.ndarray) -> np.ndarray:
    """Boolean mask of annotations to keep, depending on `label_mode`."""
    if self.label_mode == 'binary':
      return np.ones(len(ann_symbols), dtype=bool)
    if self.label_mode == 'multiclass':
      return np.isin(ann_symbols, MULTICLASS_SYMBOLS)
    raise ValueError(f"Invalid label_mode: {self.label_mode}")

  def _encode_labels(self, ann_symbols: np.ndarray) -> np.ndarray:
    """Encodes kept annotation symbols into the final label array."""
    if self.label_mode == 'binary':
      return (ann_symbols == 'N').astype(int)
    # 'multiclass': raw symbol strings ('N', 'V', 'A', 'L', 'R')
    return ann_symbols.astype(str)

  def segment_record(self,
                     record,
                     annotation,
                     record_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts, pads/interpolates, and normalizes one fixed-length
    window per kept beat annotation in a single record.

    Parameters
    ----------
    record : wfdb.Record
      Record object returned by `load_record`.
    annotation : wfdb.Annotation
      Annotation object returned by `load_record`.
    record_name : str
      Record identifier, echoed into the `record_ids` output.

    Returns
    -------
    X : ndarray of shape (n_kept_beats, window_size)
      Normalized, fixed-length signal segments.
    y : ndarray of shape (n_kept_beats,)
      Encoded labels (see `label_mode`).
    record_ids : ndarray of shape (n_kept_beats,)
      `record_name` repeated once per kept beat.
    """
    signal = record.p_signal[:, self.channel]
    ann_samples = annotation.sample
    ann_symbols = np.array(annotation.symbol)

    keep = self._keep_mask(ann_symbols)
    kept_samples = ann_samples[keep]
    kept_symbols = ann_symbols[keep]

    n_kept = len(kept_samples)
    X = np.empty((n_kept, self.window_size), dtype=float)

    half_window = self.window_size // 2
    for i, sample in enumerate(kept_samples):
      start = max(sample - half_window, 0)
      end = start + self.window_size
      segment = signal[start:end]

      if len(segment) < self.window_size:
        segment = self._handle_length(segment)

      X[i] = self._normalize(segment)

    y = self._encode_labels(kept_symbols)
    record_ids = np.full(n_kept, record_name)

    return X, y, record_ids

  def _normalize(self, segment: np.ndarray) -> np.ndarray:
    if self.normalization == 'zscore':
      mean, std = np.mean(segment), np.std(segment)
      return (segment - mean) / (std + 1e-8)
    if self.normalization == 'minmax':
      min_val, max_val = np.min(segment), np.max(segment)
      return (segment - min_val) / (max_val - min_val + 1e-8)
    raise ValueError(f"Invalid normalization: {self.normalization}")

  def _handle_length(self, segment: np.ndarray) -> np.ndarray:
    if self.length_mode == 'pad':
      pad_width = self.window_size - len(segment)
      return np.pad(segment, (0, pad_width), mode='constant')
    if self.length_mode == 'interpolate':
      x_old = np.linspace(0, 1, len(segment))
      x_new = np.linspace(0, 1, self.window_size)
      f = interpolate.interp1d(x_old, segment, kind='linear')
      return f(x_new)
    raise ValueError(f"Invalid length_mode: {self.length_mode}")

  def visualize_segment(self, raw_segment: np.ndarray, processed_segment: np.ndarray):
    """Plots a raw vs. processed segment side by side. Requires
    `matplotlib` (optional dependency, imported lazily)."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(raw_segment)
    plt.title('Original segment')
    plt.subplot(1, 2, 2)
    plt.plot(processed_segment)
    plt.title('Processed segment')
    plt.tight_layout()
    plt.show()

  def process_record(self, record_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads and segments a single record by name."""
    record, annotation = self.load_record(record_name)
    X, y, record_ids = self.segment_record(record, annotation, record_name)
    print(f"{record_name}: {len(X)} extracted segments.")
    return X, y, record_ids

  def process_all_records(self, limit: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Processes every `.dat` record found in `records_dir`.

    Parameters
    ----------
    limit : int, default = None
      If given, only the first `limit` records (sorted by name) are processed.

    Returns
    -------
    X_all : ndarray of shape (n_segments, window_size)
      Stacked normalized segments across every processed record.
    y_all : ndarray of shape (n_segments,)
      Encoded labels (`int` for `'binary'`, `str` for `'multiclass'`).
    record_all : ndarray of shape (n_segments,)
      Source record name per segment.
    """
    all_records = sorted([f.replace('.dat', '') for f in os.listdir(self.records_dir) if f.endswith('.dat')])
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
    record_all = np.concatenate(record_all)

    print(f"\nTotal: {len(X_all)} extracted segments from {len(all_records)} records.")
    return X_all, y_all, record_all