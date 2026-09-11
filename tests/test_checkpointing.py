"""Tests for `ecg_quantification.ExperimentCheckpoint`.

The scenario that matters most here is genuinely restarting: a fresh
`ExperimentCheckpoint` instance pointed at the same `checkpoint_dir`
(simulating a new Python process after an interruption) must see
exactly the state a previous instance left behind.
"""
import glob
import os
import numpy as np
import pandas as pd
import pytest

from ecg_quantification import ExperimentCheckpoint


class _DummyQuantifier:
  """Minimal stand-in for a fitted quack quantifier -- just enough
  state to prove joblib round-tripping preserves it."""

  def __init__(self, tag):
    self.tag = tag
    self.classes_ = np.array(['N', 'V'])

  def predict(self, X):
    return np.array([0.5, 0.5])


@pytest.fixture
def checkpoint_dir(tmp_path):
  return str(tmp_path / "checkpoint")


def test_fresh_checkpoint_starts_empty(checkpoint_dir):
  ckpt = ExperimentCheckpoint(checkpoint_dir)

  assert ckpt.is_fitted("LR-EMQ") is False
  assert ckpt.is_evaluated("LR-EMQ", 200) is False
  assert ckpt.fitted_names == []
  assert ckpt.n_evaluated_units == 0
  assert ckpt.load_results().empty


def test_save_and_load_fitted_quantifier(checkpoint_dir):
  ckpt = ExperimentCheckpoint(checkpoint_dir)
  quantifier = _DummyQuantifier(tag="original")

  ckpt.save_fitted("LR-EMQ", quantifier)

  assert ckpt.is_fitted("LR-EMQ") is True
  loaded = ckpt.load_fitted("LR-EMQ")
  assert loaded.tag == "original"
  np.testing.assert_array_equal(loaded.classes_, quantifier.classes_)


def test_load_fitted_raises_for_unknown_name(checkpoint_dir):
  ckpt = ExperimentCheckpoint(checkpoint_dir)
  with pytest.raises(KeyError, match="LR-EMQ"):
    ckpt.load_fitted("LR-EMQ")


def test_save_fitted_overwrites_previous_checkpoint(checkpoint_dir):
  ckpt = ExperimentCheckpoint(checkpoint_dir)
  ckpt.save_fitted("LR-EMQ", _DummyQuantifier(tag="v1"))
  ckpt.save_fitted("LR-EMQ", _DummyQuantifier(tag="v2"))

  assert ckpt.load_fitted("LR-EMQ").tag == "v2"


def test_fitted_state_survives_a_new_instance_pointed_at_same_dir(checkpoint_dir):
  # simulates a process restart: a brand new ExperimentCheckpoint object,
  # not the same Python object that wrote the checkpoint
  first = ExperimentCheckpoint(checkpoint_dir)
  first.save_fitted("RF-ACC", _DummyQuantifier(tag="persisted"))

  second = ExperimentCheckpoint(checkpoint_dir)

  assert second.is_fitted("RF-ACC") is True
  assert second.load_fitted("RF-ACC").tag == "persisted"
  assert second.fitted_names == ["RF-ACC"]


def test_save_and_load_evaluation(checkpoint_dir):
  ckpt = ExperimentCheckpoint(checkpoint_dir)
  rows = [
    {"quantifier": "LR-EMQ", "bag_index": 0, "mae": 0.05},
    {"quantifier": "LR-EMQ", "bag_index": 1, "mae": 0.07},
  ]

  ckpt.save_evaluation("LR-EMQ", bag_size=200, rows=rows)

  assert ckpt.is_evaluated("LR-EMQ", 200) is True
  assert ckpt.is_evaluated("LR-EMQ", 500) is False  # different bag_size, different unit
  assert ckpt.n_evaluated_units == 1

  results = ckpt.load_results()
  assert len(results) == 2
  assert set(results["mae"]) == {0.05, 0.07}


def test_evaluation_state_survives_a_new_instance_pointed_at_same_dir(checkpoint_dir):
  first = ExperimentCheckpoint(checkpoint_dir)
  first.save_evaluation("LR-EMQ", 200, [{"mae": 0.1}])

  second = ExperimentCheckpoint(checkpoint_dir)

  assert second.is_evaluated("LR-EMQ", 200) is True
  assert len(second.load_results()) == 1


def test_load_results_concatenates_multiple_units(checkpoint_dir):
  ckpt = ExperimentCheckpoint(checkpoint_dir)
  ckpt.save_evaluation("LR-EMQ", 100, [{"mae": 0.1}, {"mae": 0.2}])
  ckpt.save_evaluation("LR-EMQ", 200, [{"mae": 0.3}])
  ckpt.save_evaluation("RF-ACC", 100, [{"mae": 0.4}])

  results = ckpt.load_results()

  assert len(results) == 4
  assert ckpt.n_evaluated_units == 3


def test_no_leftover_tmp_files_after_successful_saves(checkpoint_dir):
  ckpt = ExperimentCheckpoint(checkpoint_dir)
  ckpt.save_fitted("LR-EMQ", _DummyQuantifier(tag="v1"))
  ckpt.save_evaluation("LR-EMQ", 200, [{"mae": 0.1}])

  leftover = glob.glob(os.path.join(checkpoint_dir, "**", "*.tmp"), recursive=True)
  assert leftover == []


def test_record_run_config_persists_on_first_call(checkpoint_dir):
  ckpt = ExperimentCheckpoint(checkpoint_dir)
  config = {"random_state": 0, "bag_sizes": [100, 200, 500]}

  ckpt.record_run_config(config)

  assert os.path.exists(ckpt.config_path)


def test_record_run_config_is_a_noop_when_matching(checkpoint_dir):
  ckpt = ExperimentCheckpoint(checkpoint_dir)
  config = {"random_state": 0}

  ckpt.record_run_config(config)
  ckpt.record_run_config(config)  # must not raise


def test_record_run_config_raises_on_mismatch_after_restart(checkpoint_dir):
  first = ExperimentCheckpoint(checkpoint_dir)
  first.record_run_config({"random_state": 0})

  second = ExperimentCheckpoint(checkpoint_dir)
  with pytest.raises(ValueError, match="does not match"):
    second.record_run_config({"random_state": 1})


def test_manifest_is_human_readable_json(checkpoint_dir):
  ckpt = ExperimentCheckpoint(checkpoint_dir)
  ckpt.save_fitted("LR-EMQ", _DummyQuantifier(tag="v1"))

  with open(ckpt.manifest_path) as f:
    import json
    manifest = json.load(f)

  assert "LR-EMQ" in manifest["fitted"]
