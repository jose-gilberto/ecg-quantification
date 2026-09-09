"""FROZEN (CBMS'26 reproducibility snapshot). See legacy_cbms26/__init__.py.
Superseded by ecg_quantification.datasets.MITBIHDownloader -- do not extend this module.
"""
import wfdb
import os
import json
from datetime import datetime
from abc import ABC, abstractmethod


class BaseDatasetDownloader(ABC):

  def __init__(self, dataset_name: str, dataset_id: str, source_url: str, version: str) -> None:
    super().__init__()
    self.dataset_name = dataset_name
    self.dataset_id = dataset_id
    self.source_url = source_url
    self.version = version

  @abstractmethod
  def download(self, data_dir: str, overwrite: bool = False) -> dict[str, any]:
    raise NotImplementedError
    
  def _create_dirs(self, data_dir: str) -> dict[str, str]:
    records_dir = os.path.join(data_dir, 'records')
    os.makedirs(records_dir, exist_ok=True)
    manifest_path = os.path.join(data_dir, 'manifest.json')
    return {
      'records_dir': records_dir,
      'manifest_path': manifest_path
    }
    
  def _save_manifest(self, manifest: dict[str, any], manifest_path: str) -> None:
    with open(manifest_path, 'w') as file:
      json.dump(manifest, file, indent=4)

  def _make_manifest(self, records_dir: str, extra_info: dict[str, any]) -> dict[str, any]:
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
      'download_date': datetime.now().isoformat(),
    }

    manifest.update(extra_info)
    return manifest
  
  def _check_integrity(self, records_dir: str, expected_n: int) -> bool:
    dat_files = sorted([f for f in os.listdir(records_dir) if f.endswith('.dat')])
    hea_files = sorted([f for f in os.listdir(records_dir) if f.endswith('.hea')])
    atr_files = sorted([f for f in os.listdir(records_dir) if f.endswith('.atr')])

    n_common = min(len(dat_files), len(hea_files), len(atr_files))
    if n_common < expected_n:
      print(f'Integrity check failed.')
      return False
    
    missing = []
    for file in dat_files:
      base = file.split('.')[0]
      for ext in ['.hea', '.atr']:
        if not os.path.exists(os.path.join(records_dir, base + ext)):
          missing.append(base + ext)

    if missing:
      print(f'Files missing: {missing}.')
      return False
    
    print('Integrity check passed with success.')
    return True


class MITBIHDownloader(BaseDatasetDownloader):

  def __init__(self) -> None:
    super().__init__(
      dataset_name='MIT-BIH Arrythmia Database',
      dataset_id='mitdb',
      source_url='https://www.physionet.org/content/mitdb/1.0.0/',
      version='1.0.0'
    )

  def download(self, data_dir, overwrite = False) -> dict[str, any]:
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
        'Ambulatory ECG with heart rate recordings ',
        '(48 recordings of 30 minutes, 360Hz, 2 channels).'
      ),
    }

    manifest = self._make_manifest(records_dir, extra_info)
    self._save_manifest(manifest, manifest_path)

    print('Running integrity check...')
    self._check_integrity(records_dir, expected_n=48)

    print(f'Dataset {self.dataset_name} downloaded and stored in {records_dir}.')
    return manifest
