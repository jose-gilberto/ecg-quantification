"""Tests for `ecg_quantification.preprocessing.ECGPreprocessor`, focused
on the multiclass label path added on top of the original binary
(N-vs-rest) CBMS'26 preprocessing.

`segment_record` is exercised directly with lightweight stand-ins for
`wfdb.Record`/`wfdb.Annotation` (only the two attributes actually read,
`p_signal` and `sample`/`symbol`, are needed) so these tests don't
require real WFDB record files on disk.
"""
from types import SimpleNamespace
import numpy as np
import pytest

from ecg_quantification.preprocessing import ECGPreprocessor, MULTICLASS_SYMBOLS


def _fake_record(signal: np.ndarray):
  # p_signal is (n_samples, n_channels); duplicate the 1D signal so
  # channel=0 (the default) reads exactly `signal` back
  return SimpleNamespace(p_signal=np.column_stack([signal, signal]))


def _fake_annotation(samples, symbols):
  return SimpleNamespace(sample=np.asarray(samples), symbol=list(symbols))


@pytest.fixture
def synthetic_record():
  # a longer synthetic signal with beats every 100 samples, symbols
  # covering both MULTICLASS_SYMBOLS members and an excluded one ('/')
  rng = np.random.default_rng(0)
  n_samples = 2000
  signal = rng.normal(size=n_samples)
  beat_samples = np.arange(200, 1800, 100)
  symbols = ['N', 'V', 'N', 'A', 'N', 'L', 'N', 'R', 'N', '/', 'N', 'N',
             'V', 'N', 'A', 'N']
  assert len(beat_samples) == len(symbols)
  return _fake_record(signal), _fake_annotation(beat_samples, symbols)


def test_multiclass_keeps_only_multiclass_symbols(synthetic_record):
  record, annotation = synthetic_record
  preprocessor = ECGPreprocessor('unused/data_dir', label_mode='multiclass')

  X, y, record_ids = preprocessor.segment_record(record, annotation, 'rec001')

  assert set(np.unique(y)).issubset(set(MULTICLASS_SYMBOLS))
  # the single '/' (paced beat, excluded) annotation must be dropped
  assert len(y) == len(annotation.symbol) - 1
  assert np.all(record_ids == 'rec001')


def test_multiclass_labels_are_raw_symbol_strings(synthetic_record):
  record, annotation = synthetic_record
  preprocessor = ECGPreprocessor('unused/data_dir', label_mode='multiclass')

  _, y, _ = preprocessor.segment_record(record, annotation, 'rec001')

  assert y.dtype.kind in ('U', 'S')  # string dtype, not encoded ints
  expected = [s for s in annotation.symbol if s in MULTICLASS_SYMBOLS]
  assert list(y) == expected


def test_binary_keeps_every_beat_and_encodes_n_vs_rest(synthetic_record):
  record, annotation = synthetic_record
  preprocessor = ECGPreprocessor('unused/data_dir', label_mode='binary')

  X, y, _ = preprocessor.segment_record(record, annotation, 'rec001')

  # binary mode keeps every annotation, including the excluded-in-multiclass '/'
  assert len(y) == len(annotation.symbol)
  assert set(np.unique(y)).issubset({0, 1})
  expected = np.array([1 if s == 'N' else 0 for s in annotation.symbol])
  np.testing.assert_array_equal(y, expected)


def test_segment_shape_matches_window_size(synthetic_record):
  record, annotation = synthetic_record
  window_size = 128
  preprocessor = ECGPreprocessor('unused/data_dir', label_mode='multiclass', window_size=window_size)

  X, y, _ = preprocessor.segment_record(record, annotation, 'rec001')

  assert X.shape == (len(y), window_size)


def test_invalid_label_mode_raises():
  preprocessor = ECGPreprocessor('unused/data_dir', label_mode='multiclass')
  preprocessor.label_mode = 'not_a_real_mode'  # bypass the constructor's implicit contract

  with pytest.raises(ValueError, match="Invalid label_mode"):
    preprocessor._keep_mask(np.array(['N', 'V']))


@pytest.mark.parametrize("normalization", ['zscore', 'minmax'])
def test_normalization_output_range(normalization):
  preprocessor = ECGPreprocessor('unused/data_dir', normalization=normalization)
  segment = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

  normalized = preprocessor._normalize(segment)

  if normalization == 'zscore':
    assert normalized.mean() == pytest.approx(0.0, abs=1e-6)
  else:
    assert normalized.min() == pytest.approx(0.0, abs=1e-6)
    assert normalized.max() == pytest.approx(1.0, abs=1e-6)


def test_invalid_normalization_raises():
  preprocessor = ECGPreprocessor('unused/data_dir')
  preprocessor.normalization = 'not_a_real_normalization'

  with pytest.raises(ValueError, match="Invalid normalization"):
    preprocessor._normalize(np.array([1.0, 2.0, 3.0]))


def test_pad_length_handling_extends_with_zeros():
  preprocessor = ECGPreprocessor('unused/data_dir', window_size=10, length_mode='pad')
  short_segment = np.array([1.0, 2.0, 3.0])

  handled = preprocessor._handle_length(short_segment)

  assert handled.shape == (10,)
  np.testing.assert_array_equal(handled[3:], np.zeros(7))
  np.testing.assert_array_equal(handled[:3], short_segment)


def test_interpolate_length_handling_preserves_endpoints():
  preprocessor = ECGPreprocessor('unused/data_dir', window_size=10, length_mode='interpolate')
  short_segment = np.array([0.0, 5.0, 10.0])

  handled = preprocessor._handle_length(short_segment)

  assert handled.shape == (10,)
  assert handled[0] == pytest.approx(0.0)
  assert handled[-1] == pytest.approx(10.0)


def test_beat_near_start_of_recording_is_still_window_sized():
  # a beat annotated at sample index 5, with half_window > 5, forces the
  # `start = max(sample - half_window, 0)` clamp and a short raw segment
  rng = np.random.default_rng(0)
  signal = rng.normal(size=500)
  record = _fake_record(signal)
  annotation = _fake_annotation([5, 250, 480], ['N', 'V', 'N'])
  preprocessor = ECGPreprocessor('unused/data_dir', label_mode='multiclass', window_size=360)

  X, y, _ = preprocessor.segment_record(record, annotation, 'rec_edge')

  assert X.shape == (3, 360)
  assert not np.isnan(X).any()
