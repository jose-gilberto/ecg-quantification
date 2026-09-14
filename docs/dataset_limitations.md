# MIT-BIH multiclass label distribution: dataset limitations

Empirically measured from `prepare_splits.py --label-mode multiclass`
(random_state=0, test_size=0.3, StratifiedGroupKFold-based patient
split). Values below are specific to this split; see
`scripts/validate_multiclass_balance.py` to reproduce for any other
split.

## Beat count distribution (full dataset, 100,062 beats / 48 patients)

| Label | Count | % |
|---|---|---|
| N | 75,052 | 75.01% |
| L | 8,075 | 8.07% |
| R | 7,259 | 7.25% |
| V | 7,130 | 7.13% |
| A | 2,546 | 2.54% |

## Patient diversity per label (this is the number that matters for
## external validity, not raw beat count)

| Label | Distinct patients (train) | Distinct patients (test) | Total distinct patients (dataset) |
|---|---|---|---|
| N | 27 | 13 | 40 |
| V | 25 | 12 | ~30 (some overlap possible across splits, counted separately per split) |
| A | 18 | 9 | 27 |
| R | 4 | 2 | 6 |
| L | 3 | 1 | **4** |

## Implication

`L` and `R` beat annotations in MIT-BIH come from a small, fixed subset
of patients (4 and 6 respectively, out of 48). No patient-wise
train/test split -- regardless of random_state -- can give these
classes the same patient diversity as N/V/A, because the patients
carrying these labels simply don't exist in larger numbers in this
database. This is a structural property of MIT-BIH, not an artifact of
our splitting code (`guaranteed_patient_train_test_split` already
retries seeds specifically to guarantee both-split presence, which
succeeded here on the first attempt -- presence isn't the issue,
diversity is).

**Consequence for result interpretation**: prevalence estimates for `L`
and `R` (and any Zone-Crossing Error computed against their clinical
risk bands in `CLINICAL_RANGES`) are more likely to reflect
patient-specific ECG morphology learned by the base classifier than a
generalizable population-level arrhythmia pattern. Report these two
classes' results with this caveat attached, and consider it when
weighing the "quantification paradox" claim if it's driven
disproportionately by L/R performance.

**Mitigation considered**: patient-wise k-fold evaluation (aggregating
every patient into a test fold at least once across the full run)
narrows this problem somewhat but cannot eliminate it -- the ceiling on
patient diversity for L/R is set by the dataset itself, not the
evaluation protocol.