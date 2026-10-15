# Vasopressor-response prediction from documented administration (VitalDB + MOVER)

Code accompanying the manuscript:

> **Prediction of Observed Arterial-Pressure Response Following Documented
> Intraoperative Vasopressor Administration: A Two-Cohort Study (VitalDB and MOVER).**
> Ge Gao. Submitted to *Heliyon*, 2026.

## What this study does

We predict the observed mean-arterial-pressure (MAP) response in the five minutes
after a **documented** intraoperative vasopressor dose. Exposure is taken only from
device/record evidence — VitalDB infusion-pump transitions and the MOVER medication
administration record (MAR). It is **never** inferred from the pressure change being
predicted. This structurally removes the circular design that inflates apparent
performance when "treatment events" are reverse-engineered from the outcome.

The study is framed as **prediction / association of an observed response**, not as a
causal estimate of a drug's effect.

### Headline findings
- A **baseline-MAP-only reference** is a strong competitor and sometimes the best model;
  much of the measured "response" is regression to the mean.
- 80% prediction intervals were **under-covered** — individual-level uncertainty is not
  yet trustworthy.
- Frozen VitalDB (Korea) models needed recalibration on MOVER (US) and never beat
  cohort-native models — a genuine OR→MAR domain shift.
- Feature-group SHAP: baseline-MAP context drives ~67–70% of predictions, **dose only ~3–6%**,
  which bounds the impact of MAR dose-unit heterogeneity.
- **Epinephrine** was excluded from the primary analysis after an IV-route sensitivity
  check (only ~37% of its documented starts were IV; the rest are local-anaesthetic adjuvant).

## Repository layout

```
vitaldb_pipeline/     VitalDB development pipeline (v3_step0..step9)
                      census -> event extraction -> modeling -> interpretation
                      -> figures -> model export -> IV sensitivity -> SHAP
mover_package/        Self-contained package shipped to the MOVER server:
  src/                  dose audit, event extraction, modeling + external validation
  config/               column mappings + MAR action vocabulary (YAML)
  models/vitaldb/       VitalDB-trained model bundles (frozen, for external validation);
                        store population-median imputation defaults only — no patient rows
  make_fixture.py       synthetic smoke fixture generator
mover_run_record/     Snapshot of the scripts the MOVER-side operator actually ran
                      (rewritten to the real EPIC schema), plus REPRODUCE.md
LICENSE               MIT
```

## Data availability

This repository contains **code only**. No patient-level or per-event data from VitalDB
or MOVER is included or redistributable; both are governed by their respective data-use
agreements.

- **VitalDB** — https://vitaldb.net (open; registration/DUA)
- **MOVER** — https://mover.ics.uci.edu (UC Irvine; DUA required)

The frozen model bundles in `mover_package/models/vitaldb/` store only trained
parameters and population-median imputation defaults; they contain no individual records.

## Reproducing

Paths are read from environment variables (defaults are placeholders):

```bash
export VITALDB_ROOT=/path/to/vitaldb/physionet.org
export TOPIC2_WORK=/path/to/working_dir      # optional; defaults to repo parent
python vitaldb_pipeline/v3_step0_documented_census.py
# ... steps run in numeric order
```

The MOVER arm is run on a MOVER-licensed host with the self-contained package; see
`mover_package/README.md` and `mover_package/TRANSFER_AND_RETURN.md`. The exact scripts
used for the reported MOVER run, and how they map to the real EPIC schema, are in
`mover_run_record/REPRODUCE.md`.

## Citation

Please cite the manuscript above once published. Code released under the MIT License.

