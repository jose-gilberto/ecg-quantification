"""Default clinical prevalence ranges for the V/A/L/R beat labels.

Bands are expressed as fractions (0-1), not percentages, and ordered
low-risk -> gray-zone -> high-risk/investigation, matching the clinical
protocol described for PVC (V), PAC (A), LBBB (L), and RBBB (R).
"""

CLINICAL_RANGES: dict[str, list[tuple[float, float]]] = {
  # PVC (Premature Ventricular Contraction)
  'V': [(0.00, 0.05), (0.05, 0.10), (0.10, 0.40)],
  # PAC (Atrial Premature Beat)
  'A': [(0.00, 0.01), (0.01, 0.02), (0.02, 0.15)],
  # LBBB: low-risk band only meaningful as an intermittent/artifact regime;
  # true continuous LBBB is qualitatively high-risk across its whole range
  'L': [(0.00, 0.01), (0.01, 1.00)],
  # RBBB: risk is contextual (co-occurring findings), not proportional to
  # burden, so a single wide band is used
  'R': [(0.01, 1.00)],
}

# human-readable zone labels, aligned index-for-index with CLINICAL_RANGES
RISK_ZONE_LABELS: dict[str, list[str]] = {
  'V': ['low_risk', 'gray_zone', 'high_risk'],
  'A': ['low_risk', 'gray_zone', 'high_risk'],
  'L': ['low_risk_intermittent', 'high_risk_continuous'],
  'R': ['contextual_risk'],
}

# default oversampling of gray/high-risk zones relative to a naive uniform
# pick across bands; adjust per study design (e.g. surveillance vs. triage)
ZONE_WEIGHTS: dict[str, list[float]] = {
  'V': [0.2, 0.3, 0.5],
  'A': [0.2, 0.3, 0.5],
  'L': [0.5, 0.5],
  'R': [1.0],
}