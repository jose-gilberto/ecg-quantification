import os
import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any


class BaseDatasetDownloader(ABC):
  """Abstract base class for PhysioNet-style ECG dataset downloaders.

  Handles the shared plumbing (directory layout, manifest persistence,
  file-integrity checks) so concrete downloaders only need to implement
  the dataset-specific `download` logic.

  Parameters
  ----------
  dataset_name : str
    Human-readable dataset name, stored in the manifest.
  dataset_id : str
    PhysioNet project slug (e.g. `'mitdb'`), used by `wfdb.dl_database`.
  source_url : str
    Canonical PhysioNet URL for the dataset, stored in the manifest.
  version : str
    Dataset version string, stored in the manifest.
  """

  def __init__(self, dataset_name: str, dataset_id: str, source_url: str, version: str) -> None:
    super().__init__()
    self.dataset_name = dataset_name
    self.dataset_id = dataset_id
    self.source_url = source_url
    self.version = version

  @abstractmethod
  def download(self, data_dir: str, overwrite: bool = False) -> dict[str, Any]:
    """Downloads the dataset into `data_dir`, returning its manifest.

    Parameters
    ----------
    data_dir : str
      Local directory the dataset (records + manifest) will be stored in.
    overwrite : bool, default = False
      If False and a manifest already exists in `data_dir`, the existing
      manifest is returned without re-downloading.

    Returns
    -------
    manifest : dict[str, Any]
      Dataset manifest (see `_make_manifest`).
    """
    raise NotImplementedError

  def _create_dirs(self, data_dir: str) -> dict[str, str]:
    """Ensures the dataset's `records/` subdirectory exists and returns
    the standard set of paths used by every downloader."""
    records_dir = os.path.join(data_dir, 'records')
    os.makedirs(records_dir, exist_ok=True)
    manifest_path = os.path.join(data_dir, 'manifest.json')
    return {
      'records_dir': records_dir,
      'manifest_path': manifest_path,
    }

  def _save_manifest(self, manifest: dict[str, Any], manifest_path: str) -> None:
    with open(manifest_path, 'w') as file:
      json.dump(manifest, file, indent=4)

  def _make_manifest(self, records_dir: str, extra_info: dict[str, Any]) -> dict[str, Any]:
    """Builds the dataset manifest from the downloaded record files,
    merged with any dataset-specific `extra_info` (sampling rate,
    channels, description, etc.)."""
    record_files = sorted([
      file for file in os.listdir(records_dir) if file.endswith('.dat')
    ])
    record_names = [os.path.splitext(file)[0] for file in record_files]

    manifest = {
      'dataset_name': self.dataset_name,
      'dataset_id': self.dataset_id,
      'source_url': self.source_url,
      'version': self.version,
      'n_records': len(record_names),
      'record_names': record_names,
      'download_date': datetime.now(timezone.utc).isoformat(),
    }

    manifest.update(extra_info)
    return manifest

  def _check_integrity(self, records_dir: str, expected_n: int) -> bool:
    """Verifies every `.dat` record has matching `.hea`/`.atr` companion
    files and that at least `expected_n` complete record triplets exist."""
    dat_files = sorted([f for f in os.listdir(records_dir) if f.endswith('.dat')])
    hea_files = sorted([f for f in os.listdir(records_dir) if f.endswith('.hea')])
    atr_files = sorted([f for f in os.listdir(records_dir) if f.endswith('.atr')])

    n_common = min(len(dat_files), len(hea_files), len(atr_files))
    if n_common < expected_n:
      print(
        f'Integrity check failed: expected at least {expected_n} complete records, '
        f'found {n_common} (dat={len(dat_files)}, hea={len(hea_files)}, atr={len(atr_files)}).'
      )
      return False

    missing = []
    for file in dat_files:
      base = file.split('.')[0]
      for ext in ('.hea', '.atr'):
        if not os.path.exists(os.path.join(records_dir, base + ext)):
          missing.append(base + ext)

    if missing:
      print(f'Files missing: {missing}.')
      return False

    print('Integrity check passed with success.')
    return True