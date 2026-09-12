"""Tests for the per-record checkpoint layer added to
`scripts/prepare_splits.py` (`process_all_records_with_checkpoint`).

`scripts/` isn't an installed package, so it's imported directly by
path here rather than via a normal `import`.
"""
import importlib.util
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ecg_quantification import ExperimentCheckpoint


def _load_prepare_splits_module():
  repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  script_path = os.path.join(repo_root, 'scripts', 'prepare_splits.py')
  spec = importlib.util.spec_from_file_location('prepare_splits', script_path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


prepare_splits = _load_prepare_splits_module()


class _FakePreprocessor:
  """Stands in for `ECGPreprocessor`: only `.records_dir` (read via
  `os.listdir`) and `.process_record(name)` are used by
  `process_all_records_with_checkpoint`."""

  def __init__(self, records_dir, n_beats_per_record=5):
    self.records_dir = records_dir
    self.n_beats_per_record = n_beats_per_record
    self.process_record_calls = []

  def process_record(self, record_name):
    self.process_record_calls.append(record_name)
    n = self.n_beats_per_record
    X = np.full((n, 4), fill_value=int(record_name))
    y = np.array(['N'] * n)
    record_ids = np.full(n, record_name)
    return X, y, record_ids


@pytest.fixture
def records_dir(tmp_path):
  d = tmp_path / "records"
  d.mkdir()
  for name in ["100", "101", "102"]:
    (d / f"{name}.dat").write_bytes(b"")
    (d / f"{name}.hea").write_bytes(b"")
  return str(d)


@pytest.fixture
def checkpoint(tmp_path):
  return ExperimentCheckpoint(str(tmp_path / "checkpoint"))


def test_processes_every_record_on_first_run(records_dir, checkpoint):
  preprocessor = _FakePreprocessor(records_dir)

  X, y, record_ids = prepare_splits.process_all_records_with_checkpoint(preprocessor, checkpoint)

  assert sorted(preprocessor.process_record_calls) == ["100", "101", "102"]
  assert X.shape == (15, 4)  # 3 records x 5 beats
  assert len(y) == 15
  assert set(np.unique(record_ids)) == {"100", "101", "102"}


def test_rerun_skips_every_already_cached_record(records_dir, checkpoint):
  preprocessor_first = _FakePreprocessor(records_dir)
  prepare_splits.process_all_records_with_checkpoint(preprocessor_first, checkpoint)

  # a brand new preprocessor instance (and a fresh call-log) simulates a
  # new process picking up the same checkpoint directory after a restart
  preprocessor_second = _FakePreprocessor(records_dir)
  X, y, record_ids = prepare_splits.process_all_records_with_checkpoint(preprocessor_second, checkpoint)

  assert preprocessor_second.process_record_calls == []  # nothing reprocessed
  assert X.shape == (15, 4)
  assert set(np.unique(record_ids)) == {"100", "101", "102"}


def test_only_uncached_records_are_reprocessed_after_partial_interruption(records_dir, checkpoint):
  # simulate an interruption that only got through record "100" before dying
  preprocessor_partial = _FakePreprocessor(records_dir)
  checkpoint.save_artifact('record:100', preprocessor_partial.process_record("100"))

  preprocessor_resumed = _FakePreprocessor(records_dir)
  X, y, record_ids = prepare_splits.process_all_records_with_checkpoint(preprocessor_resumed, checkpoint)

  assert sorted(preprocessor_resumed.process_record_calls) == ["101", "102"]
  assert X.shape == (15, 4)


def test_respects_limit_records(records_dir, checkpoint):
  preprocessor = _FakePreprocessor(records_dir)

  X, y, record_ids = prepare_splits.process_all_records_with_checkpoint(preprocessor, checkpoint, limit=2)

  assert sorted(preprocessor.process_record_calls) == ["100", "101"]
  assert X.shape == (10, 4)
