import os
import json
from typing import Any
import wfdb

from ecg_quantification.datasets.base import BaseDatasetDownloader


class MITBIHDownloader(BaseDatasetDownloader):
  """Downloader for the MIT-BIH Arrhythmia Database (PhysioNet `mitdb`).

  48 half-hour ambulatory ECG recordings at 360 Hz, 2 channels, with
  expert beat-level annotations (including the V/A/L/R labels used by
  `ecg_quantification`'s clinical prevalence protocol).

  Examples
  --------
  >>> from ecg_quantification.datasets import MITBIHDownloader
  >>> downloader = MITBIHDownloader()
  >>> manifest = downloader.download('data/mitbih')
  >>> manifest['n_records']
  48
  """

  def __init__(self) -> None:
    super().__init__(
      dataset_name='MIT-BIH Arrhythmia Database',
      dataset_id='mitdb',
      source_url='https://www.physionet.org/content/mitdb/1.0.0/',
      version='1.0.0',
    )

  def download(self, data_dir: str, overwrite: bool = False) -> dict[str, Any]:
    """Downloads the MIT-BIH database into `data_dir`.

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
      Dataset manifest, including `sampling_rate`, `channels`, and
      `description` alongside the shared fields from `_make_manifest`.
    """
    dirs = self._create_dirs(data_dir)
    records_dir, manifest_path = dirs['records_dir'], dirs['manifest_path']

    if not overwrite and os.path.exists(manifest_path):
      print(f'Dataset already exists in {data_dir}. Use overwrite=True to replace it.')
      with open(manifest_path, 'r') as file:
        return json.load(file)

    print(f'Downloading {self.dataset_name} from PhysioNet...')
    wfdb.dl_database(self.dataset_id, dl_dir=records_dir)

    extra_info = {
      'sampling_rate': 360,
      'channels': 2,
      'description': (
        'Ambulatory ECG with heart rate recordings '
        '(48 recordings of 30 minutes, 360Hz, 2 channels).'
      ),
    }

    manifest = self._make_manifest(records_dir, extra_info)
    self._save_manifest(manifest, manifest_path)

    print('Running integrity check...')
    self._check_integrity(records_dir, expected_n=48)

    print(f'Dataset {self.dataset_name} downloaded and stored in {records_dir}.')
    return manifest