"""Frozen implementation used to produce the CBMS'26 conference results.

This package is a snapshot of the exact code that generated the tables
and figures in "Beyond Single-Beat Classification: Quantifying
Arrhythmia in Long-Term ECG via Prevalence Estimation" (CBMS'26). It is
kept, unmodified, purely for reproducibility of that publication (the
`notebooks/01_*` through `11_*` notebooks import from here).

Do NOT extend this package. All new development (multiclass support,
the clinically-constrained bag protocol, the Zone-Crossing Error
metric, ECGFounder/LITETime adapters, etc.) lives in
`ecg_quantification`, which supersedes every module here:

  - `legacy_cbms26.download.MITBIHDownloader`   -> `ecg_quantification.datasets.MITBIHDownloader`
  - `legacy_cbms26.preprocess.ECGPreprocessor`  -> `ecg_quantification.preprocessing.ECGPreprocessor`
  - `legacy_cbms26.bags.generate_bags`          -> `ecg_quantification.APPBagGenerator`
  - `legacy_cbms26.utils.split_by_record_id`    -> `ecg_quantification.model_selection.patient_train_test_split`

If a bug is found here, prefer fixing the equivalent code in
`ecg_quantification` and, if reproducibility of the published numbers
is not affected, leaving this package as historical record.
"""
