# Vasopressor-response prediction from documented administration (MOVER)

Analysis code accompanying the manuscript:

> **Incremental Value of Pre-Dose Treatment History and Model Complexity for
> Predicting Short-Term Arterial Pressure After Documented Intraoperative
> Vasopressor Bolus-Class Records.**
> Ge Gao, Department of Anesthesiology, The First Affiliated Hospital,
> Zhejiang University School of Medicine. Submitted to *Scientific Reports*.

## What the study does

We predict the change in mean arterial pressure (MAP) five minutes after a
**documented** intravenous vasopressor administration, for phenylephrine,
ephedrine and vasopressin, with norepinephrine reported as an exploratory
cohort. Exposure is taken only from the medication administration record (MAR) —
never inferred from the pressure change being predicted — which removes the
circular exposure definition that arises when treatment events are reconstructed
from the outcome.

The design varies two factors orthogonally on identical patient partitions:

1. **Information** — a base set of routinely available pre-dose variables versus
   the base set plus a block of pre-dose treatment history.
2. **Model class** — Ridge regression versus tuned XGBoost, given identical
   information.

A calibrated probability model for reaching MAP ≥ 65 mmHg at five minutes is
reported alongside the regression, because that is the decision the regression
implicitly implies.

## Summary of results

- Both increments are small: the treatment-history effect is at most a few
  tenths of a millimetre of mercury and the model-class effect at most about a
  fifth, against an outcome standard deviation above 20 mmHg and roughly 60% unexplained
  variance.
- The best candidate model improves on a baseline-MAP-only reference by about
  0.95–1.04 mmHg, an interval spanning the investigator-set descriptive benchmark
  of 1 mmHg. The benchmark is not a validated minimum clinically important difference.
- The model-class comparison depends on the comparator: XGBoost beats a Ridge model
  fitted to squared error, but a linear median regression achieves lower mean absolute
  error than the evaluated XGBoost configuration by 0.14–0.34 mmHg. XGBoost was trained
  on squared error and the median regression on absolute error, so that comparison
  reflects both model class and training objective and does not separate them. What it
  does show is that greater model complexity did not guarantee lower error.
- The calibrated classifier improves on a probability model built from pre-dose MAP
  alone (AUROC +0.057 to +0.062 whole cohort, +0.065 to +0.083 below 65 mmHg), so it
  carries information beyond the current pressure; absolute discrimination below
  65 mmHg is moderate.
- Excluding administrations followed by further vasopressor treatment within the
  outcome window preserved the qualitative conclusions, although individual effect
  estimates changed.
- Evaluation is internal to a single development source; no external validation was done.

## Repository layout

```
host_extraction/     runs on the data host; no modelling
  src/mover_io.py        loading and windowing (forward-only, read-only)
  src/extract_events.py  exposure classification, history features, event table
  src/inspect_columns.py column-name discovery
  src/make_fixture.py    synthetic smoke-test data
  src/package_return.py  bundles outputs for return
  config/                column map and MAR action taxonomy
  run.sh                 --inspect | --test | --extract | --package

analysis/            runs locally from the extracted event table
  01_fit_regression_models.py    Ridge / XGBoost, five patient-grouped partitions
  02_fit_probability_models.py   penalised logistic and calibrated XGBoost classifiers
  03_descriptive_reports.py      cohort flow, feature availability, follow-up census
  04_score_per_partition.py      per-partition scoring of every reported metric
  05_build_results_index.py      single results index keyed by estimand
  06_audit_manuscript.py         consistency audit of manuscript against the index
  07_figure1_cohort_flow.py      Figure 1
  08_figures.py                  Figures 2-4 and Figure S1
  09_build_supplement.py         Supplementary Information LaTeX
  10_fair_linear_comparators.py  tuned Ridge, median regression, baseline-MAP-only classifier
  11_no_further_vasopressor_sensitivity.py  sensitivity cohort with no further treatment

results/             aggregate result tables (no row-level data)
```

## Running the analysis

Nothing in this repository fits a model until `data/event_master.parquet` exists.
That table is produced by `host_extraction/` on the MOVER data host and is **not
distributed**: it is row-level perioperative data under the data-use agreement.

```bash
# on the data host, with config/mover_columns.yaml pointing at the MOVER tables
bash host_extraction/run.sh --inspect
bash host_extraction/run.sh --test       # synthetic smoke test
bash host_extraction/run.sh --extract
bash host_extraction/run.sh --package

# locally, once data/event_master.parquet is in place
python analysis/01_fit_regression_models.py
python analysis/02_fit_probability_models.py
python analysis/03_descriptive_reports.py
python analysis/04_score_per_partition.py
python analysis/05_build_results_index.py
python analysis/06_audit_manuscript.py
python analysis/07_figure1_cohort_flow.py
python analysis/08_figures.py
python analysis/10_fair_linear_comparators.py
python analysis/11_no_further_vasopressor_sensitivity.py
python analysis/09_build_supplement.py
```

## Environment

Python 3.10; XGBoost 3.1.3; scikit-learn 1.7.2; NumPy 1.26.4; pandas 2.3.3;
SciPy; Matplotlib. The host extraction ran under Python 3.12 (NumPy 2.5.4,
pandas 2.2.3) and fits no models.

## Statistics

Every performance measure is scored on each of five patient-grouped partitions
separately and reported as the across-partition mean with a patient-cluster
bootstrap confidence interval whose resampling target is that same mean. The
exception is the binned reliability and regression-decile curve, which uses the
five-partition mean prediction and is labelled a stability summary. Bootstrap
resample counts are 2,000 for whole-cohort quantities, 1,000 for calibration
parameters, subgroup and strata, and 500 per reliability bin. Seed 20261027.

The paired model comparisons are also reported against linear comparators matched to the
evaluation loss (tuned Ridge; median regression) and against a baseline-MAP-only
probability model, in `10_fair_linear_comparators.py`, and repeated on a cohort with no
further vasopressor treatment in the outcome window in
`11_no_further_vasopressor_sensitivity.py`.

`analysis/06_audit_manuscript.py` checks the reported numbers for internal
consistency against `results/results_index.csv`. It is a consistency check of the
reported values, not an independent re-extraction of source data.

## Data and ethics

MOVER is available from the University of California, Irvine, under a data-use
agreement (https://mover.ics.uci.edu). No patient data are redistributed here.
The source database was approved by the Institutional Review Boards of the UCI
Medical Center and the main UCI campus, with the requirement for individual
consent waived on the basis of HIPAA-compliant de-identification; this study
accessed only the de-identified release under its data-use agreement.

## Licence

MIT (see LICENSE).
