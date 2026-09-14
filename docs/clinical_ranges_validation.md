# CLINICAL_RANGES validation against the literature

Ranges audited: `ecg_quantification.CLINICAL_RANGES` (V/A/L/R prevalence
bands used by `ClinicalPrevalenceBagGenerator` and `ZoneCrossingError`).
This document exists to give the journal extension a citable basis for
these thresholds -- the CBMS'26 paper did not need this (binary N-vs-rest
only), so it was never written up before.

## V (PVC) -- well supported, matches the code as-is

Current: `[(0.00, 0.05), (0.05, 0.10), (0.10, 0.40)]`

- <5% burden is consistently treated as low risk; >5% commonly prompts
  structural workup (echocardiography). Source: Mayo Clinic Proceedings,
  "Management of Premature Ventricular Complexes in the Outpatient
  Setting" (2023) -- https://www.mayoclinicproceedings.org/article/S0025-6196%252823%252900032-0/fulltext
- 10% is the most consistently reported cutoff for PVC-induced
  cardiomyopathy (PIC) across studies, with reported thresholds ranging
  from 4% to 24%. Source: "Burden of Premature Ventricular Complexes and
  Risk of Cardiomyopathy: A Cross-Sectional Study" (JACC EP, 2025) --
  https://www.jacc.org/doi/10.1016/j.jacep.2025.01.004
- A separate review reaches the same "10% or higher" framing for referral
  to EP/cardiology. Source: "Clinical and Translational Insights on PVCs
  and PVC-Induced Cardiomyopathy" (PMC, 2022) --
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9192164/
- The 40% upper bound is a plausible real-world ceiling, not a clinical
  threshold: one cross-sectional cohort observed PVC burdens up to 43.4%.
  Same source as above (JACC EP cross-sectional study).

**Conclusion**: no change needed. This is the best-supported band of the
four.

## A (PAC) -- right direction, unit mismatch worth addressing

Current: `[(0.00, 0.01), (0.01, 0.02), (0.02, 0.15)]` (fractions of total
beats)

- The clinical literature on PAC burden almost universally uses
  **absolute counts**, not a fraction of total beats. The standard
  "excessive supraventricular ectopic activity" (ESVEA) definition
  endorsed by the European Heart Rhythm Association (EHRA) is >=720
  PACs/day, or any run of >=20 PACs; an earlier, still-cited definition
  uses >30 PACs/hour. Sources: "Excessive supraventricular ectopic
  activity and risk of incident atrial fibrillation" (2021) --
  https://pubmed.ncbi.nlm.nih.gov/34337573/ ; "Excessive Supraventricular
  Ectopic Activity and Increased Risk of Atrial Fibrillation and Stroke"
  (Circulation, 2010) -- https://www.ahajournals.org/doi/10.1161/circulationaha.109.874982
- Converting the EHRA threshold to a fraction of total beats, assuming a
  representative resting heart rate of ~70 bpm (~100,000 beats/day):
  720 / 100,000 ~= 0.72%. This lands close to, but *below*, the current
  1% low/gray boundary.
- No percentage-based literature threshold was found for a "high risk"
  PAC burden analogous to PVC's 10% cutoff; PAC risk literature reports
  hazard ratios against ESVEA presence/absence or PAC/24h counts (e.g.
  57 PAC/24h as an optimal cutoff for AF recurrence prediction in one
  ablation-outcome study --
  https://www.explorationpub.com/Journals/ec/Article/101281), not
  against a percentage-of-total-beats framing at all.

**Recommendation**: state explicitly in the paper's methods that these
bounds are a percentage-of-total-beats approximation of the
count-based ESVEA literature, under an assumed average heart rate.
Consider tightening the low/gray boundary from 0.01 to ~0.0072 to match
the EHRA definition more precisely -- left as a judgment call, since the
existing 0.01/0.02 bounds are still defensible as a round-number
approximation.

## L (LBBB) -- correct framing, threshold is a modeling choice

Current: `[(0.00, 0.01), (0.01, 1.00)]`

- LBBB's clinical significance in the literature is about **presence**,
  not dose: Framingham data show markedly elevated 10-year cardiovascular
  mortality associated with LBBB onset; more recent work reports LBBB as
  an independent predictor of sudden cardiac death, heart failure
  mortality, and MI. Source: "Left Bundle Branch Block: Current and
  Future Perspectives" (Circ Arrhythm Electrophysiol, 2019) --
  https://www.ahajournals.org/doi/10.1161/CIRCEP.119.008239
- No study defines a burden percentage separating "intermittent" from
  "sustained" LBBB; intermittent/rate-related LBBB is discussed
  qualitatively (e.g. exercise-induced LBBB in ~0.38% of patients
  undergoing exercise testing -- a different population/context, not a
  burden threshold). Source: "Unmasking Coronary Artery Disease With
  Intermittent Left Bundle Branch Block" (PMC, 2024) --
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10942114/

**Conclusion**: the code's own comment ("low-risk band only meaningful
as an intermittent/artifact regime; true continuous LBBB is
qualitatively high-risk across its whole range") is an honest, defensible
design choice, not a literature-derived number. State it as such in the
paper rather than implying the 1% cutoff itself is clinically validated.

## R (RBBB) -- single band is the right call, best-supported qualitative claim

Current: `[(0.01, 1.00)]`

- Consistently described as contextual, not burden-dependent: isolated
  RBBB in an asymptomatic person is benign and does not significantly
  affect life expectancy; RBBB co-occurring with structural heart
  disease, CAD, or heart failure carries materially higher risk. Sources:
  "Right Bundle Branch Block and Its Impact on Life Expectancy" --
  https://www.rupahealth.com/post/right-bundle-branch-block-and-its-impact-on-life-expectancy
  ; "Prognostic Significance of Right Bundle Branch Block for Patients
  with Acute Myocardial Infarction" (meta-analysis) --
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4811299/

**Conclusion**: no change needed. A quantification framework fundamentally
cannot capture "context" (comorbidities, structural disease) from beat
burden alone, so collapsing R to a single non-informative band --
which `ZoneCrossingError` already excludes from scoring (bands < 2) --
is the honest choice rather than fabricating a burden threshold the
literature doesn't support.

## Summary table

| Label | Bounds well-supported by literature? | Action |
|---|---|---|
| V | Yes, directly (10% cutoff, <5%/>5% zones) | None |
| A | Direction yes, exact bounds are a unit-converted approximation | Document the beats-vs-count conversion; optionally tighten 0.01 -> ~0.0072 |
| L | Framing yes (presence-based risk), exact cutoff is a design choice | Document as a modeling choice, not a cited threshold |
| R | Yes, qualitatively (contextual, not burden-dependent) | None |