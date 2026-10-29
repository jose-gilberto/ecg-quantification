"""Portable bundles of a finished experiment, for instance-level analysis
on another machine (which beats drive each quantifier's errors, and how
changing them changes the estimates).

`scripts/export_bundle.py` writes a bundle; `ExportedBundle` reads it:

>>> from ecg_quantification.bundle import ExportedBundle
>>> bundle = ExportedBundle("ecg_bundle")                      # doctest: +SKIP
>>> results = bundle.results()       # per-bag quantifier predictions/errors
>>> bags = bundle.bags(100)          # which test beats are in every bag
>>> beats = bundle.bag_instances(100, bag_index=7)  # one bag's beats + outputs
>>> outputs = bundle.instance_outputs("LR-CC")      # per-beat classifier outputs
>>> quantifier = bundle.load_quantifier("LR-EMQ")   # the fitted model itself

Layout of a bundle directory
----------------------------
- `bundle.json`: index of everything below (experiments, bag configs,
  quantifiers, classes, file names).
- `environment.json`: package versions and git commits it was made with.
- `data/splits.npz`: the train/test splits (optional).
- `experiments/<name>/`: `run_config.json`, `detailed_results.csv`,
  `summary_table.csv` and `checkpoint/` (the fitted quantifiers, loadable
  with `ExperimentCheckpoint`).
- `bags/<key>/bags_<size>.npz`: per bag size, `indices` (n_bags x
  bag_size) into the test split, `true_prevalences`, `classes`, and
  per-label risk `zones` -- regenerated with the run's own settings and
  checked against its recorded true prevalences.
- `instances/<experiment>/<quantifier>.npz`: per-beat outputs of the
  classifier(s) inside a quantifier, for the test (and train) split.
"""
import json
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# per-instance outputs of a fitted quantifier
# ---------------------------------------------------------------------------

def _positive_column(classifier) -> int:
  classes = np.asarray(classifier.classes_).tolist()
  return classes.index(1) if 1 in classes else len(classes) - 1


def quantifier_instance_outputs(quantifier, X) -> dict | None:
  """Per-instance outputs of the classifier(s) inside a fitted quantifier.

  Returns
  -------
  outputs : dict or None
    `kind`:
    - `"multiclass"` -- a native multiclass quantifier (CC, PCC, GAC, ...,
      possibly under `LabelEncodedQuantifier`): `proba` is its classifier's
      `predict_proba`, columns = `classes`.
    - `"one-vs-rest"` -- `OneVsRestQuantifier`: column k of `proba` is
      P(class k vs. rest) from the k-th binary member's own classifier
      (rows do not sum to 1).
    `pred` is the per-row argmax label. `None` for quantifiers without a
    classifier (HDx, ReadMe, ED work on the features directly).
  """
  from ecg_quantification.calibration._calibration import _unwrap, outputs_from_quantifier
  from ecg_quantification.quantifiers._one_vs_rest import OneVsRestQuantifier

  if isinstance(quantifier, OneVsRestQuantifier):
    columns = []
    for member in quantifier.estimators_:
      classifier = getattr(member, "classifier_", None)
      if classifier is None or not hasattr(classifier, "predict_proba"):
        return None
      classifier = _unwrap(classifier)
      columns.append(np.asarray(classifier.predict_proba(X))[:, _positive_column(classifier)])
    proba = np.column_stack(columns)
    classes = np.asarray(quantifier.classes_)
    return {"kind": "one-vs-rest", "classes": classes, "proba": proba, "pred": classes[np.argmax(proba, axis=1)]}

  inner = getattr(quantifier, "quantifier_", quantifier)
  if getattr(inner, "classifier_", None) is None:
    return None
  outputs = outputs_from_quantifier(quantifier, X)
  classes = outputs["classes"]
  return {"kind": "multiclass", "classes": classes, "proba": outputs["proba"],
          "pred": classes[np.argmax(outputs["proba"], axis=1)]}


def save_instance_outputs(path: str, outputs_by_split: dict) -> None:
  """Writes `{split: outputs}` (from `quantifier_instance_outputs`) to one npz."""
  arrays = {}
  for split, outputs in outputs_by_split.items():
    arrays[f"{split}_proba"] = np.asarray(outputs["proba"], dtype=np.float32)
    arrays[f"{split}_pred"] = np.asarray(outputs["pred"]).astype(str)
    arrays["classes"] = np.asarray(outputs["classes"]).astype(str)
    arrays["kind"] = np.asarray(outputs["kind"])
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = path + ".tmp.npz"
  np.savez_compressed(tmp, **arrays)
  os.replace(tmp, path)


def bag_config_key(n_bags: int, random_state) -> str:
  return f"capp_n{n_bags}_seed{random_state}"


# ---------------------------------------------------------------------------
# reading a bundle
# ---------------------------------------------------------------------------

class ExportedBundle:
  """Read access to a bundle written by `scripts/export_bundle.py`.

  Parameters
  ----------
  path : str
    The bundle directory (unpacked, if it was archived).
  """

  def __init__(self, path: str):
    self.path = path
    with open(os.path.join(path, "bundle.json")) as handle:
      self.index = json.load(handle)

  # -- what is inside -------------------------------------------------------

  @property
  def experiments(self) -> list[str]:
    return list(self.index["experiments"])

  def quantifiers(self, experiment: str = None) -> list[str]:
    names = []
    for name in ([experiment] if experiment else self.experiments):
      names += self.index["experiments"][name]["quantifiers"]
    return names

  def _experiment_of(self, quantifier: str, experiment: str = None) -> str:
    if experiment:
      return experiment
    owners = [e for e in self.experiments if quantifier in self.index["experiments"][e]["quantifiers"]]
    if not owners:
      raise KeyError(f"No experiment in this bundle has a quantifier named {quantifier!r}.")
    return owners[0]

  # -- data -----------------------------------------------------------------

  def splits(self) -> dict:
    """The train/test arrays (`X_train`, `y_train`, `ids_train`, `X_test`,
    `y_test`, `ids_test`), if the bundle includes them."""
    path = os.path.join(self.path, "data", "splits.npz")
    if not os.path.exists(path):
      raise FileNotFoundError("This bundle was exported without the splits (--no-include-splits).")
    with np.load(path, allow_pickle=True) as data:
      return {key: data[key] for key in data.files}

  def results(self) -> pd.DataFrame:
    """Every experiment's per-bag results (one row per quantifier x bag:
    true/predicted prevalences, MAE, ZCE), with an `experiment` column."""
    frames = []
    for name in self.experiments:
      frame = pd.read_csv(os.path.join(self.path, "experiments", name, "detailed_results.csv"))
      frame.insert(0, "experiment", name)
      frames.append(frame)
    return pd.concat(frames, ignore_index=True)

  def bags(self, bag_size: int, experiment: str = None) -> dict:
    """`indices` (n_bags x bag_size, into the test split),
    `true_prevalences` (n_bags x n_classes), `classes` and `zones_<label>`
    for the bags of `bag_size` used by `experiment` (default: the first)."""
    experiment = experiment or self.experiments[0]
    key = self.index["experiments"][experiment]["bag_config"]
    with np.load(os.path.join(self.path, "bags", key, f"bags_{bag_size}.npz")) as data:
      return {name: data[name] for name in data.files}

  def instance_outputs(self, quantifier: str, split: str = "test", experiment: str = None,
                       compute: bool = True) -> dict:
    """Per-beat outputs of the classifier(s) inside `quantifier` (see
    `quantifier_instance_outputs`). Read from the bundle when exported;
    otherwise computed from the fitted quantifier and the splits (needs
    the bundle's models and data, plus the model's own dependencies --
    TensorFlow for LITETime, PyTorch for ECGFounder) and cached in the
    bundle when it is writable."""
    experiment = self._experiment_of(quantifier, experiment)
    path = os.path.join(self.path, "instances", experiment, f"{quantifier}.npz")
    if os.path.exists(path):
      with np.load(path) as data:
        if f"{split}_proba" in data.files:
          return {"kind": str(data["kind"]), "classes": data["classes"],
                  "proba": data[f"{split}_proba"], "pred": data[f"{split}_pred"]}
    if not compute:
      raise KeyError(f"No exported {split} outputs for {quantifier}.")
    outputs = quantifier_instance_outputs(self.load_quantifier(quantifier, experiment), self.splits()[f"X_{split}"])
    if outputs is None:
      raise ValueError(f"{quantifier} has no classifier (feature-space quantifier): no per-instance outputs.")
    try:
      existing = {}
      if os.path.exists(path):
        with np.load(path) as data:
          for other in ("train", "test"):
            if f"{other}_proba" in data.files:
              existing[other] = {"kind": str(data["kind"]), "classes": data["classes"],
                                 "proba": data[f"{other}_proba"], "pred": data[f"{other}_pred"]}
      save_instance_outputs(path, {**existing, split: outputs})
    except OSError:
      pass  # read-only bundle: just return it
    return outputs

  def bag_instances(self, bag_size: int, bag_index: int, quantifier: str = None,
                    experiment: str = None) -> pd.DataFrame:
    """The beats of one bag: test index, patient, true label, and -- with
    `quantifier` -- that quantifier's per-beat classifier outputs."""
    bags = self.bags(bag_size, experiment)
    indices = bags["indices"][bag_index]
    data = np.load(os.path.join(self.path, "data", "splits.npz"), allow_pickle=True) \
      if os.path.exists(os.path.join(self.path, "data", "splits.npz")) else None
    frame = pd.DataFrame({"test_index": indices})
    if data is not None:
      frame["patient"] = data["ids_test"][indices] if "ids_test" in data.files else None
      frame["y_true"] = data["y_test"][indices]
    if quantifier:
      outputs = self.instance_outputs(quantifier, "test", experiment)
      frame["pred"] = outputs["pred"][indices]
      for column, label in enumerate(outputs["classes"]):
        frame[f"p_{label}"] = outputs["proba"][indices, column]
    return frame

  # -- models ---------------------------------------------------------------

  def checkpoint(self, experiment: str):
    from ecg_quantification._checkpointing import ExperimentCheckpoint
    directory = os.path.join(self.path, "experiments", experiment, "checkpoint")
    if not os.path.exists(os.path.join(directory, "manifest.json")):
      raise FileNotFoundError(f"This bundle was exported without {experiment}'s models (--no-include-models).")
    return ExperimentCheckpoint(directory)

  def load_quantifier(self, name: str, experiment: str = None):
    """The fitted quantifier `name` (needs its dependencies installed;
    deep models: see the bundle README)."""
    experiment = self._experiment_of(name, experiment)
    return self.checkpoint(experiment).load_fitted(name)
