#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate the Supplementary Information LaTeX from the result tables.

Every number is read from results/; nothing is typed by hand. Tables S1-S16 and Figure S1.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
V16 = RES
V21 = RES
V14 = RES
OUT = ROOT / "manuscript" / "supplement"
OUT.mkdir(parents=True, exist_ok=True)

DRUG_ORDER = ["phenylephrine", "ephedrine", "vasopressin", "norepinephrine"]
DRUG_ORDER_PRIMARY = ["phenylephrine", "ephedrine", "vasopressin"]
DRUG_LABEL = {"phenylephrine": "Phenylephrine", "ephedrine": "Ephedrine",
              "vasopressin": "Vasopressin", "norepinephrine": "Norepinephrine"}

S9_RESTRICTED_BLOCK = r"""\section*{Supplementary Table S9. Restricted versus unrestricted all-administration cohort (previous version)}
\addcontentsline{toc}{section}{Table S9. Restricted vs unrestricted cohort (previous version)}
This table is retained from the earlier all-administration version of this work for
transparency and is \emph{not} the primary analysis here. The restricted cohort excludes every
administration followed by a \emph{different} vasopressor within the 10-minute outcome window (a
selection on post-dose information); the unrestricted cohort retains them. Each cohort is fitted
and evaluated on its own patient-grouped folds, so the estimates are descriptive comparisons.
MAE in mmHg; $\Delta$MAE $=$ tuned XGBoost $-$ baseline-MAP reference with patient-cluster
bootstrap 95\% CI.

\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{longtable}{llrrrrl}
\toprule Drug & Cohort & $n$ & Baseline-MAP & Ridge & XGBoost & $\Delta$MAE vs baseline (95\% CI) \\ \midrule \endhead
Phenylephrine & Restricted (earlier primary) & 25,937 & 11.36 & 10.57 & 10.38 & $-$0.98 ($-$1.08 to $-$0.89) \\
 & Unrestricted & 32,827 & 11.77 & 11.01 & 10.82 & $-$0.95 ($-$1.04 to $-$0.87) \\
Ephedrine & Restricted (earlier primary) & 7,242 & 12.26 & 11.78 & 11.52 & $-$0.74 ($-$0.89 to $-$0.60) \\
 & Unrestricted & 11,587 & 12.97 & 12.35 & 12.08 & $-$0.88 ($-$1.04 to $-$0.73) \\
Vasopressin & Restricted (earlier primary) & 3,959 & 12.48 & 11.58 & 11.53 & $-$0.94 ($-$1.15 to $-$0.73) \\
 & Unrestricted & 6,893 & 12.78 & 11.94 & 11.90 & $-$0.88 ($-$1.03 to $-$0.71) \\
Norepinephrine & Restricted (earlier primary) & 1,282 & 11.11 & 10.72 & 10.96 & $-$0.15 ($-$0.50 to $+$0.17) \\
 & Unrestricted & 2,017 & 12.03 & 11.38 & 11.23 & $-$0.79 ($-$1.49 to $-$0.28) \\
\bottomrule \end{longtable}
\normalsize
\textbf{Reading.} Removing the selection changed little for the three primary drugs. For
norepinephrine the increment moved from null in the restricted cohort to distinguishable from zero
in the unrestricted cohort, but the events the restriction removed are harder for every model and
the comparator degrades on them, so part of the widened gap is a weaker baseline rather than a more
accurate model. Norepinephrine is exploratory in this version (main text).

\clearpage"""


def r16(name):
    return pd.read_csv(V16 / name)


def r14(name):
    return pd.read_csv(V14 / name)


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def fnum(x, d=2):
    if pd.isna(x):
        return "---"
    return f"{float(x):.{d}f}"


def signed(x, d=3):
    if pd.isna(x):
        return "---"
    v = float(x)
    if abs(v) < 0.5 * 10 ** (-d):      # avoids "+$-0.000"
        return f"{0:.{d}f}"
    return (f"$+${v:.{d}f}" if v > 0 else f"$-${abs(v):.{d}f}")


def signed_ci(x, lo, hi, d=3):
    return f"{signed(x, d)} ({signed(lo, d)} to {signed(hi, d)})"


P = []


def W(s):
    P.append(s)


# ---------- preamble ----------
W(r"""\documentclass[11pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{booktabs,longtable,array,graphicx,amsmath,amssymb,caption}
\usepackage[table]{xcolor}
\usepackage{url}
\captionsetup{labelfont=bf,font=small}
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{5pt}
\title{Supplementary Information\\[2pt]
\large Incremental Value of Pre-Dose Treatment History and Model Complexity for
Predicting Short-Term Arterial Pressure After Documented Intraoperative Vasopressor
Bolus-Class Records}
\author{Ge Gao\\ Department of Anesthesiology, The First Affiliated Hospital, Zhejiang University School of Medicine, Hangzhou, China}
\date{}
\begin{document}
\maketitle
\vspace{-2.2em}
\tableofcontents
\bigskip

This document contains the supplementary tables and figure referenced in the main text.
All numbers are generated directly from the analysis result files; the three primary drugs
(phenylephrine, ephedrine, vasopressin) are reported together and norepinephrine is shown
as an exploratory cohort where applicable. Every performance measure uses the per-partition
scoring convention described in the main text: each quantity is computed on each of the five
partitions separately and averaged, and its interval is a patient-level cluster bootstrap of
the across-partition mean. The only exception is the binned reliability and regression-decile
summaries, which use the five-partition mean prediction per event as a stability summary.
Bootstrap resample counts are fixed per analysis rather than per document: 2{,}000 resamples
for the paired comparisons and the difference-in-differences (Table~S3), the whole-cohort
$R^2$, regression-calibration slope and large-error proportions (Table~S4), and the
whole-cohort classifier AUROC, AUPRC and Brier score (Table~S5); 1{,}000 resamples for the
whole-cohort classifier calibration intercept and slope, every below-65 subgroup quantity
(Table~S6), the pre-dose MAP strata (Table~S4b) and the follow-up-treatment strata (Table~S7);
and 500 resamples for each per-bin point of the reliability curves.
\clearpage
""")

# ---------- S1: exposure-class / cohort accounting (Count / Unit / Distinct patients) ----------
flow = r14("cohort_flow_full.csv").set_index("step")

# per-row display: (label, count, unit, distinct-patients-or-blank, note)
UNIT = {
    "A_raw_documented_MAR_rows_all_vasopressors": "MAR rows",
    "B_raw_arterial_MAP_rows": "MAP samples",
    "C_cases_with_documented_candidates": "cases",
    "D_raw_study_drug_administrations_all_classes": "administrations",
    "E_MAP_linked_administrations_incl_epinephrine": "administrations",
    "F_MAP_linked_study_drugs": "administrations",
    "G_class_iv_bolus": "administrations",
    "G_class_infusion_start_or_bag_change": "administrations",
    "G_class_infusion_resume": "administrations",
    "G_class_non_iv": "administrations",
    "G_class_uncertain_unit": "administrations",
    "G_class_rate_documented_as_given": "administrations",
    "H_bolus_all_study_drugs": "administrations",
    "I_bolus_three_primary": "administrations",
    "J_bolus_norepinephrine_exploratory": "administrations",
}
NOTE_OVERRIDE = {
    "H_bolus_all_study_drugs": "IV-bolus class, all four study drugs",
    "I_bolus_three_primary": "primary analysis cohort (three primary drugs)",
}

W(r"\section*{Supplementary Table S1. Cohort accounting from the database to the analysis cohort}")
W(r"\addcontentsline{toc}{section}{Table S1. Cohort accounting}")
W(r"""The accounting runs from the documented medication administration record and the raw
arterial-pressure stream to the arterial-MAP-linked administrations of the four study drugs,
which are then split into mutually exclusive exposure classes by a rule fixed before analysis.
Each row states its own counting \emph{unit}: audit-log stages are counted in MAR rows,
arterial-MAP samples or cases, whereas the event-level stages are counted in administrations.
Distinct-patient counts are given only where the source artefact records them; a blank means the
unit of that row is not an administration and no distinct-patient count applies. Patients may
contribute to more than one drug. Upstream pre-linkage exclusions cannot be decomposed into
mutually exclusive reasons from the returned tables and are stated as such. The intravenous-bolus
class of the three primary drugs (47{,}051 administrations) is the primary analysis cohort; the
47{,}808 intravenous-bolus row is the four-drug bolus set.
""")
W(r"\footnotesize")
W(r"\begin{longtable}{p{7.4cm}rlr}")
W(r"\toprule Step & Count & Unit & Distinct patients \\ \midrule \endhead")
for step, r in flow.iterrows():
    unit = UNIT.get(step, "")
    if unit == "cases":
        # a case count is itself the count; it is not a distinct-patient tally
        cnt = "" if pd.isna(r["n_patients"]) else f"{int(r['n_patients']):,}"
        npt = ""
    else:
        cnt = "" if pd.isna(r["n_events"]) else f"{int(r['n_events']):,}"
        npt = "" if pd.isna(r["n_patients"]) else f"{int(r['n_patients']):,}"
    W(f"{esc(step)} & {cnt} & {unit} & {npt} \\\\")
W(r"\bottomrule \end{longtable}")
W(r"""\footnotesize\emph{Notes.} Audit-log rows (documented MAR rows, arterial-MAP samples,
cases with a documented candidate) are host-side tallies in their stated units and are not the
unit of any model. The intravenous-bolus, infusion-start/bag-change, infusion-resume and non-IV
classes are mutually exclusive partitions of the 53{,}324 MAP-linked four-drug administrations.
Infusion-start and bag-change records cannot be separated in this release (no order identifier).
\normalsize""")
W(r"\clearpage")

# ---------- S2: availability + missingness of history features ----------
miss = r14("missingness.csv")
W(r"\section*{Supplementary Table S2. Availability and missingness of base and treatment-history features}")
W(r"\addcontentsline{toc}{section}{Table S2. Feature availability and missingness}")
W(r"""For each drug and feature we report the per-event missing rate. ``Structural absence''
marks history features whose missing value is a true ``no prior dose'' state (encoded by a
sentinel), distinct from a missing record. The proportion of administrations \emph{with} at
least one prior vasopressor administration within 30 minutes is shown once per drug (computed as
$100-$ the no-prior proportion recorded in the source file).
""")
for drug in DRUG_ORDER:
    sub = miss[miss.drug == drug]
    if sub.empty:
        continue
    npv = sub["no_prior_vaso_30m_pct"].dropna()
    if len(npv):
        prior_pct = 100.0 - float(npv.iloc[0])
        npv_txt = f" Prior vasopressor within 30 min: {fnum(prior_pct,1)}\\%."
    else:
        npv_txt = ""
    W(rf"\textbf{{{DRUG_LABEL[drug]}}} ($n={int(sub['n'].iloc[0]):,}$).{npv_txt}")
    W(r"\begin{longtable}{p{5.8cm}lrc}")
    W(r"\toprule Feature & Block & Missing (\%) & Structural \\ \midrule \endhead")
    for _, r in sub.iterrows():
        struct = "yes" if bool(r["structural_absence"]) else "---"
        W(f"{esc(r['feature'])} & {esc(r['block'])} & {fnum(r['missing_pct'],2)} & {struct} \\\\")
    W(r"\bottomrule \end{longtable}")
W(r"\clearpage")

# ---------- S3: paired comparisons incl DiD with CI ----------
dm = r14("delta_mae.csv")
did = r16("did.csv")
W(r"\section*{Supplementary Table S3. Paired $\Delta$MAE comparisons with across-partition variability}")
W(r"\addcontentsline{toc}{section}{Table S3. Paired comparisons and difference-in-differences}")
W(r"""$\Delta$MAE (mmHg) $=$ MAE of model $A$ minus MAE of model $B$, computed on identical
out-of-fold events within each of five patient partitions and averaged; negative favours $A$.
The cluster-bootstrap 95\% CI is shown alongside the standard deviation and range across the five
partitions. The difference-in-differences row (history effect under XGBoost minus history effect
under Ridge) is estimated with a single paired bootstrap that resamples both history terms
together; because each history effect is a per-event mean under the same scoring convention, the
difference-in-differences equals the difference of the two history-effect rows above it.
""")
CMP_LABEL = {
    "history_effect_on_ridge": "History effect (Ridge$+$H $-$ Ridge)",
    "history_effect_on_xgb": "History effect (XGB$+$H $-$ XGB)",
    "algorithm_effect_base": "Model class (XGB $-$ Ridge, base)",
    "algorithm_effect_with_hist": "Model class (XGB $-$ Ridge, $+$history)",
    "best_candidate_vs_baseline": "Preselected XGBoost $-$ baseline-MAP",
}
W(r"\footnotesize")
W(r"\begin{longtable}{p{5.6cm}rccc}")
W(r"\toprule Comparison & $n$ & $\Delta$MAE (95\% CI) & SD & Range \\ \midrule \endhead")
for drug in DRUG_ORDER:
    sub = dm[dm.drug == drug]
    if sub.empty:
        continue
    role = sub["role"].iloc[0]
    W(rf"\multicolumn{{5}}{{l}}{{\textbf{{{DRUG_LABEL[drug]}}} ({esc(role)})}} \\")
    for key, lab in CMP_LABEL.items():
        r = sub[sub.comparison == key]
        if r.empty:
            continue
        r = r.iloc[0]
        W(f"{lab} & {int(r['n']):,} & {signed_ci(r['dmae'], r['ci_lo'], r['ci_hi'])} & "
          f"{fnum(r['sd_across_partitions'],3)} & "
          f"{signed(r['min_partition'])} to {signed(r['max_partition'])} \\\\")
    dr = did[did.drug == drug]
    if not dr.empty:
        dr = dr.iloc[0]
        W(rf"\quad Difference-in-differences (history$_{{\mathrm{{XGB}}}}$ $-$ history$_{{\mathrm{{Ridge}}}}$) & "
          rf"{int(dr['n']):,} & {signed_ci(dr['did'], dr['ci_lo'], dr['ci_hi'])} & & \\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\clearpage")

# ---------- S4: explanatory performance + baseline R2 + regression calibration + strata + deciles ----------
mq = r16("regression_quality.csv")
br = r16("baseline_r2.csv")
strata = r16("map_strata.csv")
cal = r16("regression_calibration.csv")
W(r"\section*{Supplementary Table S4. Explanatory performance, regression calibration and error by pre-dose MAP stratum}")
W(r"\addcontentsline{toc}{section}{Table S4. Explanatory performance, calibration and MAP strata}")
W(r"""Preselected XGBoost reference model (XGBoost with treatment history), out-of-fold. Panel (a) gives
$R^2$ and the regression calibration slope, each with a patient-cluster bootstrap 95\% CI, the
baseline-MAP-only reference $R^2$ and the increment over it, and the proportion of predictions in
error by more than 5, 10 and 15 mmHg. Panel (b) gives mean absolute error by pre-dose MAP stratum
with the observed mean 5-minute change. Panel (c) gives observed-versus-predicted means by decile
of the five-partition mean predicted $\Delta$MAP (a stability summary).
""")
W(r"\textbf{(a) Explanatory performance and regression calibration.}")
W(r"\footnotesize")
W(r"\begin{longtable}{lrrccrr}")
W(r"\toprule Drug & MAE & MAE/SD & $R^2$ (95\% CI) & Calib.\ slope (95\% CI) & $R^2$ base & $\Delta R^2$ \\ \midrule \endhead")
m = mq.merge(br[["drug", "r2_baseline_only_mean", "r2_increment"]], on="drug", how="left")
for drug in DRUG_ORDER:
    r = m[m.drug == drug]
    if r.empty:
        continue
    r = r.iloc[0]
    W(f"{DRUG_LABEL[drug]} & {fnum(r['mae_mean'],2)} & {fnum(r['mae_over_sd_mean'],3)} & "
      f"{fnum(r['r2_mean'],3)} ({fnum(r['r2_boot_lo'],3)}--{fnum(r['r2_boot_hi'],3)}) & "
      f"{fnum(r['calib_slope_mean'],3)} ({fnum(r['calib_slope_boot_lo'],3)}--{fnum(r['calib_slope_boot_hi'],3)}) & "
      f"{fnum(r['r2_baseline_only_mean'],3)} & {signed(r['r2_increment'],3)} \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\medskip")
W(r"\textit{Proportion of predictions in error by more than 5, 10, 15 mmHg (95\% CI):}")
W(r"\begin{longtable}{lccc}")
W(r"\toprule Drug & $>$5 mmHg & $>$10 mmHg & $>$15 mmHg \\ \midrule \endhead")
for drug in DRUG_ORDER:
    r = mq[mq.drug == drug]
    if r.empty:
        continue
    r = r.iloc[0]
    W(f"{DRUG_LABEL[drug]} & {fnum(r['pct_err_gt_5_mean'],1)} ({fnum(r['pct_err_gt_5_lo'],1)}--{fnum(r['pct_err_gt_5_hi'],1)}) & "
      f"{fnum(r['pct_err_gt_10_mean'],1)} ({fnum(r['pct_err_gt_10_lo'],1)}--{fnum(r['pct_err_gt_10_hi'],1)}) & "
      f"{fnum(r['pct_err_gt_15_mean'],1)} ({fnum(r['pct_err_gt_15_lo'],1)}--{fnum(r['pct_err_gt_15_hi'],1)}) \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\medskip")
W(r"\textbf{(b) Mean absolute error by pre-dose MAP stratum.}")
STRAT_LABEL = {"lt65": "$<$65 mmHg", "65to75": "65--75 mmHg", "gt75": "$>$75 mmHg"}
W(r"\footnotesize")
W(r"\begin{longtable}{llrrr}")
W(r"\toprule Drug & Stratum & $n$ & MAE (95\% CI) & Obs.\ $\Delta$MAP \\ \midrule \endhead")
for drug in DRUG_ORDER:
    sub = strata[strata.drug == drug]
    for key in ["lt65", "65to75", "gt75"]:
        r = sub[sub.stratum == key]
        if r.empty:
            continue
        r = r.iloc[0]
        W(f"{DRUG_LABEL[drug]} & {STRAT_LABEL[key]} & {int(r['n']):,} & "
          f"{fnum(r['mae'],2)} ({fnum(r['lo'],2)}--{fnum(r['hi'],2)}) & {signed(r['observed_delta_mean'],2)} \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\medskip")
W(r"\textbf{(c) Observed versus predicted $\Delta$MAP by decile of predicted risk.}")
W(r"\footnotesize")
W(r"\begin{longtable}{llrrr}")
W(r"\toprule Drug & Decile & $n$ & Predicted mean & Observed mean \\ \midrule \endhead")
for drug in DRUG_ORDER:
    sub = cal[cal.drug == drug].sort_values("bin")
    for _, r in sub.iterrows():
        W(f"{DRUG_LABEL[drug]} & {int(r['bin'])+1} & {int(r['n']):,} & "
          f"{signed(r['pred_mean'],2)} & {signed(r['obs_mean'],2)} \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\clearpage")

# ---------- S5: penalised logistic classifier with ITS OWN Brier skill ----------
cm = r16("classifier_metrics.csv")
pb = r16("prob_baseline.csv")
W(r"\section*{Supplementary Table S5. Penalised logistic-regression classifier and Brier skill}")
W(r"\addcontentsline{toc}{section}{Table S5. Logistic-regression classifier and Brier skill}")
W(r"""Out-of-fold discrimination and calibration for the penalised logistic-regression classifier
(the calibrated XGBoost classifier is in main-text Table~4). All quantities use the per-partition
convention: each metric is scored on every partition and averaged. AUROC, AUPRC and Brier score are
shown with patient-cluster bootstrap 95\% CIs; calibration intercept and slope are on the log-odds
scale. The final column is the Brier skill of \emph{this logistic model}, defined as the
proportional reduction in whole-cohort Brier score relative to an out-of-fold prevalence reference.
""")
W(r"\footnotesize")
W(r"\begin{longtable}{lrccccr}")
W(r"\toprule Drug & Base rate & AUROC (95\% CI) & AUPRC & Brier (95\% CI) & Calib.\ int./slope & Brier skill \\ \midrule \endhead")
for drug in DRUG_ORDER:
    r = cm[(cm.drug == drug) & (cm.model == "logit")]
    if r.empty:
        continue
    r = r.iloc[0]
    sk = pb[(pb.drug == drug) & (pb.model == "logit")]
    skv = float(sk["brier_skill_vs_oof_prevalence"].iloc[0]) * 100.0 if not sk.empty else float("nan")
    W(f"{DRUG_LABEL[drug]} & {fnum(r['prevalence'],3)} & "
      f"{fnum(r['auroc_mean'],3)} ({fnum(r['auroc_boot_lo'],3)}--{fnum(r['auroc_boot_hi'],3)}) & "
      f"{fnum(r['auprc_mean'],3)} & "
      f"{fnum(r['brier_mean'],3)} ({fnum(r['brier_boot_lo'],3)}--{fnum(r['brier_boot_hi'],3)}) & "
      f"{signed(r['calib_intercept_mean'],2)}/{fnum(r['calib_slope_mean'],3)} & "
      f"{fnum(skv,1)}\\% \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\medskip Each Brier skill is the logistic model's own proportional reduction in whole-cohort "
  r"Brier score against its out-of-fold prevalence reference; the corresponding XGBoost skills are "
  r"in the main-text classifier table.")
W(r"\clearpage")

# ---------- S6: below-65 subgroup ----------
sg = r16("classifier_subgroup.csv")
W(r"\section*{Supplementary Table S6. Below-65 mmHg subgroup: discrimination and calibration of the probability model}")
W(r"\addcontentsline{toc}{section}{Table S6. Below-65 subgroup classifier performance}")
W(r"""Calibrated XGBoost classifier for reaching MAP $\geq$ 65 mmHg at 5 minutes, restricted to
administrations starting below 65 mmHg (the subgroup in which the decision matters most). AUROC,
Brier score, calibration intercept and calibration slope are each reported with a patient-cluster
bootstrap 95\% CI. Discrimination is lower than in the whole cohort while calibration remains
reasonable; norepinephrine is exploratory and its subgroup is small.""")
W(r"\scriptsize")
W(r"\setlength{\tabcolsep}{3pt}")
W(r"\begin{longtable}{lrrcccc}")
W(r"\toprule Drug & $n$ & Attain.\ rate & AUROC (95\% CI) & Brier (95\% CI) & Calib.\ int.\ (95\% CI) & Calib.\ slope (95\% CI) \\ \midrule \endhead")
for drug in DRUG_ORDER:
    r = sg[(sg.drug == drug) & (sg.model == "xgb") & (sg.subgroup == "baseline_map_lt65")]
    if r.empty:
        continue
    r = r.iloc[0]
    W(f"{DRUG_LABEL[drug]} & {int(r['n']):,} & {fnum(r['observed_attainment'],3)} & "
      f"{fnum(r['auroc'],3)} ({fnum(r['auroc_lo'],3)}--{fnum(r['auroc_hi'],3)}) & "
      f"{fnum(r['brier'],3)} ({fnum(r['brier_lo'],3)}--{fnum(r['brier_hi'],3)}) & "
      f"{signed(r['calib_intercept'],2)} ({signed(r['calib_intercept_lo'],2)} to {signed(r['calib_intercept_hi'],2)}) & "
      f"{fnum(r['calib_slope'],3)} ({fnum(r['calib_slope_lo'],3)}--{fnum(r['calib_slope_hi'],3)}) \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\setlength{\tabcolsep}{5pt}")
W(r"\clearpage")

# ---------- S7: outcome-window intervention + timestamp + running-rate ----------
fu = r16("followup_strata.csv")
union = r14("followup_union.csv")
rate = r14("running_rate_desc.csv")
W(r"\section*{Supplementary Table S7. Outcome-window intervention, timestamp audit and running-rate reconstruction}")
W(r"\addcontentsline{toc}{section}{Table S7. Outcome-window, timestamp and running-rate audit}")
W(r"""Panel (a): out-of-fold MAE of the preselected XGBoost reference model split by the further treatment
recorded in the 5-minute outcome window, with the observed mean $\Delta$MAP and baseline MAP.
Three strata are shown: any additional vasopressor (including a same-drug re-dose), a
different-drug vasopressor, and a same-drug re-dose. The first stratum is the union of the other
two, so the strata overlap and are not a partition; read each row against its own complement.
Because the further treatment responds to an inadequate pressure, the split is descriptive, not a
decomposition of error. Panel (b): combined proportion followed within
5 minutes by a further vasopressor or a same-drug re-dose. Panel (c): the MOVER release carries the
medication action time to the minute and contains no chart-entry or modification timestamp, so
charting lag cannot be measured; the running same-drug infusion rate before a bolus could be
reconstructed for only a minority of records and is labelled the most recent recorded rate.""")
W(r"\textbf{(a) Error by post-dose intervention.}")
W(r"\footnotesize")
W(r"\begin{longtable}{p{4.6cm}lrrrr}")
W(r"\toprule Drug / intervention & Present & $n$ & MAE & Obs.\ $\Delta$MAP & Baseline MAP \\ \midrule \endhead")
prim = fu[fu.role == "primary"]
for drug in DRUG_ORDER_PRIMARY:
    sub = prim[prim.drug == drug]
    for interv in ["any additional vasopressor within 5 min (incl. same-drug re-dose)",
                  "different-drug vasopressor within 5 min",
                  "same-drug re-dose within 5 min"]:
        for present in [0, 1]:
            r = sub[(sub.intervention == interv) & (sub.present == present)]
            if r.empty:
                continue
            r = r.iloc[0]
            yn = "yes" if present == 1 else "no"
            W(f"{DRUG_LABEL[drug]}: {interv} & {yn} & {int(r['n']):,} & "
              f"{fnum(r['mae'],2)} & {signed(r['observed_delta_mean'],2)} & {fnum(r['baseline_map_mean'],1)} \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\medskip\textbf{(b) Combined proportion with further treatment in the outcome window.}")
W(r"\begin{longtable}{lrrrr}")
W(r"\toprule Drug & $n$ & Other vasopressor (\%) & Same-drug re-dose (\%) & Union (\%) \\ \midrule \endhead")
for drug in DRUG_ORDER_PRIMARY:
    r = union[union.drug == drug]
    if r.empty:
        continue
    r = r.iloc[0]
    W(f"{DRUG_LABEL[drug]} & {int(r['n']):,} & {fnum(r['other_vaso_5m_pct'],2)} & "
      f"{fnum(r['same_redose_5m_pct'],2)} & {fnum(r['union_5m_pct'],2)} \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\medskip\textbf{(c) Timestamp audit and running-rate reconstruction.}")
W(r"\begin{longtable}{lrrrrr}")
W(r"\toprule Drug & $n$ & With recent rate (\%) & Rate age median (min) & IQR (min) & Distinct units \\ \midrule \endhead")
for drug in DRUG_ORDER_PRIMARY:
    r = rate[rate.drug == drug]
    if r.empty:
        continue
    r = r.iloc[0]
    if r["pct_with_recent_rate"] == 0:
        W(f"{DRUG_LABEL[drug]} & {int(r['n']):,} & {fnum(r['pct_with_recent_rate'],2)} & --- & --- & {int(r['n_distinct_units'])} \\\\")
    else:
        W(f"{DRUG_LABEL[drug]} & {int(r['n']):,} & {fnum(r['pct_with_recent_rate'],2)} & "
          f"{fnum(r['rate_age_median_min'],1)} & {fnum(r['rate_age_p25_min'],1)}--{fnum(r['rate_age_p75_min'],1)} & "
          f"{int(r['n_distinct_units'])} \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\medskip")
W(r"""\textit{Outcome-window availability.} Eligibility required an observable outcome in the
$+4$ to $+6$ minute window; administrations without it are excluded. The returned tables do not
isolate the exact count excluded for an unobservable outcome, so an exact outcome-missing
proportion cannot be reported (main-text Limitations); the cohort accounting in Table~S1 gives the
counts entering and leaving each stage.""")
W(r"\clearpage")

# ---------- S8: pre-specified operating points ----------
op = r16("operating_points.csv")
SCOPE_LABEL = {"overall": "Whole cohort", "baseline_lt65": "Below 65 mmHg"}
W(r"\section*{Supplementary Table S8. Pre-specified descriptive operating points of the probability model}")
W(r"\addcontentsline{toc}{section}{Table S8. Pre-specified operating points}")
W(r"""Sensitivity, specificity and predictive values of the calibrated XGBoost classifier at a grid
of \emph{pre-specified} probability cut-offs, for the whole cohort and for the below-65 mmHg
subgroup. No operating threshold was derived from the evaluation data; these cut-offs were fixed in
advance and are tabulated only to describe the trade-off. ``Attain'' is the positive class
(MAP $\geq$ 65 mmHg at 5 minutes).""")
W(r"\footnotesize")
for drug in DRUG_ORDER:
    sub = op[(op.drug == drug) & (op.model == "xgb")]
    if sub.empty:
        continue
    W(rf"\textbf{{{DRUG_LABEL[drug]}}}")
    W(r"\begin{longtable}{llrrrrr}")
    W(r"\toprule Scope & Cut-off & $n$ & Sensitivity & Specificity & PPV & NPV \\ \midrule \endhead")
    for scope in ["overall", "baseline_lt65"]:
        ss = sub[sub.scope == scope].sort_values("threshold")
        for _, r in ss.iterrows():
            ppv = "---" if pd.isna(r["ppv"]) else fnum(r["ppv"], 3)
            W(f"{SCOPE_LABEL[scope]} & {fnum(r['threshold'],1)} & {int(r['n']):,} & "
              f"{fnum(r['sensitivity'],3)} & {fnum(r['specificity'],3)} & {ppv} & {fnum(r['npv'],3)} \\\\")
    W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\clearpage")

# ---------- S9: restricted vs unrestricted (carried verbatim from v15 S10) ----------
W(S9_RESTRICTED_BLOCK)

# ---------- S16: route audit ----------
route = pd.read_csv(RES / "route_audit.csv")
W(r"\section*{Supplementary Table S16. Route audit: re-scoring on the explicitly intravenous subset}")
W(r"\addcontentsline{toc}{section}{Table S16. Route audit}")
W(r"""The exposure keyword rule excluded recorded non-intravenous routes but did not include
``intramuscular'', and seven analysed administrations carried that route. This table gives the
per-drug, per-model count of records by recorded route and the mean absolute error computed on
the full analysed set and on the explicitly intravenous subset. These are re-scores of the saved
out-of-fold predictions on a restricted evaluation set; no model was retrained, so the training
cohort of each model still contains the seven records. The maximum change in any reported error
is 0.003~mmHg.""")
W(r"\footnotesize")
W(r"\begin{longtable}{llrrrrr}")
W(r"\toprule Drug & Model & $n$ & IV & IM & MAE, all & MAE, IV only \\ \midrule \endhead")
MLAB2 = {"xgb_sq": "XGBoost, squared loss", "xgb_mae": "XGBoost, absolute loss",
         "median": "Median regression", "ridge_tuned": "Tuned Ridge", "baseline_map": "Baseline-MAP"}
for _, r_ in route.iterrows():
    W(f"{DRUG_LABEL[r_['drug']]} & {MLAB2[r_['model']]} & {int(r_['n_all']):,} & "
      f"{int(r_['n_intravenous']):,} & {int(r_['n_intramuscular'])} & {r_['mae_all']:.3f} & "
      f"{r_['mae_intravenous_only']:.3f} \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\medskip Missing route: none. Phenylephrine has no non-intravenous record.")
W(r"\clearpage")



# ---------- S14: implementation checks ----------
W(r"\section*{Supplementary Table S14. Implementation checks on the analysis code}")
W(r"\addcontentsline{toc}{section}{Table S14. Implementation checks}")
W(r"""Each property below was verified against the saved fold assignments, the saved out-of-fold
prediction arrays and the feature lists actually passed to the models, rather than by
re-reading the manuscript. Every check passed.""")
W(r"\footnotesize")
W(r"\begin{longtable}{p{7.6cm}p{7.0cm}}")
W(r"\toprule Check & Result \\ \midrule \endhead")
CHECKS = [
 ("Patient isolation, per drug and per repeat",
  "No patient appears on both sides of a fold, and the five folds partition the patients exactly once: 0 overlap violations in 5 repeats x 5 folds for each of the four drugs"),
 ("One prediction per event per repeat",
  "No missing values in any out-of-fold array; array width equals the analysed cohort size for every drug and feature set"),
 ("Repeats are distinct partitions",
  "Each drug's five out-of-fold vectors differ from one another, so five repeats are five partitions rather than one partition scored five times"),
 ("No post-dose information in the feature lists",
  "None of the 27 base and history features begins fu\\_ (the 12 follow-up columns in the table are unused); the outcome column is not a feature"),
 ("Exposure-class consistency",
  "The analysed subset for each drug contains only records classified as intravenous bolus; no other class enters the analysis"),
 ("Outcome-observability denominator",
  "Analysable events as a fraction of documented intravenous-bolus medication-record rows: phenylephrine 29,114/214,385 (13.6\\%), ephedrine 11,585/75,945 (15.2\\%), vasopressin 6,352/39,819 (16.0\\%). Upper bound only: the census counts every bolus row on a case with a documented candidate, without requiring an eligible surgery or arterial-pressure window"),
]
for a, b in CHECKS:
    W(f"{a} & {b} \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\medskip\emph{What these checks do not establish.} They confirm that the code implements the described "
  r"partition, prediction and feature rules. They do not verify that a medication-record entry labelled as a "
  r"bolus corresponds to a clinical intravenous push, and they cannot detect an error reproduced identically "
  r"in both the analysis and the result index.")
W(r"\clearpage")

# ---------- S15: XGBoost with an absolute-error objective ----------
mae_cmp = pd.read_csv(V21 / "mae_xgb_comparison.csv")
mae_d = pd.read_csv(V21 / "mae_xgb_delta.csv")
W(r"\section*{Supplementary Table S15. XGBoost trained on absolute error}")
W(r"\addcontentsline{toc}{section}{Table S15. XGBoost with an absolute-error objective}")
W(r"""The main-text XGBoost model was trained on squared error while the median regression
minimised absolute error, so the comparison between them reflected both model class and
training objective. Here XGBoost is refitted with the absolute-error objective
(\texttt{reg:absoluteerror}) on the identical folds and events, with the same
12-configuration randomised search scored on mean absolute error over the same inner
patient-grouped three-fold splits. Panel (a) gives mean absolute error for the three
configurations; panel (b) gives the paired differences. Negative favours the first-named
model.""")
W(r"\textbf{(a) Mean absolute error.}")
W(r"\footnotesize")
W(r"\begin{longtable}{llrrrr}")
W(r"\toprule Drug & Features & XGBoost (squared error) & XGBoost (absolute error) & Median reg. & Tuned Ridge \\ \midrule \endhead")
for drug in DRUG_ORDER:
    for feat in ["base", "base+hist"]:
        m = mae_cmp[(mae_cmp.drug == drug) & (mae_cmp.features == feat)]
        sq = r14("model_comparison.csv")
        sqv = sq[(sq.drug == drug) & (sq.features == feat) & (sq.model == "xgb")]
        med = r16("model_comparison_linear.csv")
        medv = med[(med.drug == drug) & (med.features == feat) & (med.model == "median")]
        rid = med[(med.drug == drug) & (med.features == feat) & (med.model == "ridge_tuned")]
        W(f"{DRUG_LABEL[drug]} & {esc(feat)} & {fnum(sqv.mae_mean.iloc[0],2) if len(sqv) else '---'} & "
          f"{fnum(m.mae_mean.iloc[0],2) if len(m) else '---'} & "
          f"{fnum(medv.mae_mean.iloc[0],2) if len(medv) else '---'} & "
          f"{fnum(rid.mae_mean.iloc[0],2) if len(rid) else '---'} \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\medskip")
W(r"\textbf{(b) Paired differences in mean absolute error.}")
MAE_LABEL = {
 "mae_xgb_vs_squared_xgb_base": "XGBoost (absolute) $-$ XGBoost (squared), base",
 "mae_xgb_vs_squared_xgb_hist": "XGBoost (absolute) $-$ XGBoost (squared), $+$history",
 "mae_xgb_vs_median_base": "XGBoost (absolute) $-$ median regression, base",
 "mae_xgb_vs_median_hist": "XGBoost (absolute) $-$ median regression, $+$history",
 "mae_xgb_vs_ridge_tuned_base": "XGBoost (absolute) $-$ tuned Ridge, base",
 "mae_xgb_hist_vs_baseline": "XGBoost (absolute, $+$history) $-$ baseline-MAP",
}
W(r"\footnotesize")
W(r"\begin{longtable}{p{7.0cm}rccc}")
W(r"\toprule Comparison & $n$ & $\Delta$MAE (95\% CI) & SD & Range \\ \midrule \endhead")
for drug in DRUG_ORDER:
    sub = mae_d[mae_d.drug == drug]
    if sub.empty:
        continue
    W(rf"\multicolumn{{5}}{{l}}{{\textbf{{{DRUG_LABEL[drug]}}}}} \\")
    for key, lab in MAE_LABEL.items():
        q = sub[sub.comparison == key]
        if q.empty:
            continue
        q = q.iloc[0]
        W(f"{lab} & {int(q['n']):,} & {signed_ci(q['dmae'], q['ci_lo'], q['ci_hi'])} & "
          f"{fnum(q['sd_across_partitions'],3)} & {signed(q['min_partition'])} to {signed(q['max_partition'])} \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\clearpage")



# ---------- S10: substantive changes across versions ----------
W(r"\section*{Supplementary Table S10. Substantive changes across versions of this analysis}")
W(r"\addcontentsline{toc}{section}{Table S10. Version history}")
W(r"""The analysis was revised repeatedly in the light of earlier analyses of the same database and of
earlier analyses of the same database. Only versions in which the cohort definition or a reading of the results changed
substantively are listed; purely editorial revisions are omitted. None of the versions was publicly
registered, so the sequence is a record of what was done, not a pre-specification.""")
VERSIONS = [
 ("Earlier version A", "All documented administrations of five agents in a two-database design (VitalDB and MOVER); epinephrine later excluded after an intravenous-route check",
  "A baseline-MAP-only reference was competitive with or better than every model"),
 ("Earlier version B", "MOVER only; administrations not followed by a different vasopressor within 10 min (a selection on post-dose care)",
  "Same qualitative reading; the norepinephrine increment was null"),
 ("Earlier version C", "All arterial-MAP-linked administrations with no post-dose selection (53,324 events, four study drugs)",
  "The norepinephrine increment favoured the model but only in the above-75 mmHg stratum, and was judged unstable"),
 ("Earlier version D", "Intravenous-bolus class only (47,808 events, four study drugs); the orthogonal design over treatment history and model class was introduced",
  "Both increments small; the norepinephrine bolus cohort (757 events) was too small for a stable estimate"),
 ("Current version", "Intravenous-bolus class of the three primary drugs (47,051 events); norepinephrine exploratory; per-partition scoring; calibrated probability model; tuned Ridge and median-regression comparators; baseline-MAP-only probability model; no-further-vasopressor sensitivity cohort",
  "As reported in the main text"),
]
W(r"\footnotesize")
W(r"\begin{longtable}{p{2.6cm}p{7.4cm}p{5.4cm}}")
W(r"\toprule Version & Cohort definition & Change in reading \\ \midrule \endhead")
for v, c, r_ in VERSIONS:
    W(f"{esc(v)} & {esc(c)} & {esc(r_)} \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\clearpage")

# ---------- S11: fair linear comparators (tuned Ridge, median regression) ----------
v19d = r16("delta_mae.csv")
v19m = r16("model_comparison_linear.csv")
alph = r16("ridge_alphas.csv")
W(r"\section*{Supplementary Table S11. Fair linear comparators: tuned Ridge and median regression}")
W(r"\addcontentsline{toc}{section}{Table S11. Tuned Ridge and median regression}")
W(r"""The main-text Ridge comparison uses a fixed penalty (alpha $=1$) while XGBoost receives an inner
cross-validated random search. To test whether that asymmetry favours the gradient-boosted model we
refitted, on the identical folds and events, (i) Ridge with alpha selected by inner patient-grouped
3-fold cross-validation scored on MAE over a grid of 11 values from 0.01 to 1000, and (ii) a median
($\tau=0.5$) quantile regression, a linear model whose objective is the evaluation metric itself.
XGBoost was fitted with the squared-error objective (the library default), so the XGBoost-versus-median
comparison differs in both model class and training objective and does not separate the two.
All quantities are per-partition means with patient-cluster bootstrap 95\% CIs. Panel (a) gives mean
absolute error; panel (b) gives the paired differences; negative favours the first-named model.""")
W(r"\textbf{(a) Mean absolute error.}")
W(r"\footnotesize")
W(r"\begin{longtable}{llrrrr}")
W(r"\toprule Drug & Features & Ridge ($\alpha=1$) & Ridge (tuned) & Median reg. & XGBoost \\ \midrule \endhead")
v14m = r14("model_comparison.csv")
for drug in DRUG_ORDER:
    for feat in ["base", "base+hist"]:
        row = {}
        for model in ["ridge", "ridge_tuned", "median", "xgb"]:
            if model == "ridge":
                q = v14m[(v14m.drug == drug) & (v14m.features == feat) & (v14m.model == "ridge")]
            else:
                q = v19m[(v19m.drug == drug) & (v19m.features == feat) & (v19m.model == model)]
            row[model] = fnum(q.mae_mean.iloc[0], 2) if len(q) else "---"
        W(f"{DRUG_LABEL[drug]} & {esc(feat)} & {row['ridge']} & {row['ridge_tuned']} & {row['median']} & {row['xgb']} \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
al = alph[alph.features == "base"]
W(r"\medskip")
W(r"\textbf{Selected Ridge penalties.} Reported as the median and the range across the 25 fold fits, "
  r"per drug and feature set; a single value means the inner search chose the same penalty in every fit.")
W(r"\footnotesize")
W(r"\begin{longtable}{llrrr}")
W(r"\toprule Drug & Feature set & Median $\alpha$ & Min & Max \\ \midrule \endhead")
for drug in DRUG_ORDER:
    for feat in ["base", "base+hist"]:
        q = alph[(alph.drug == drug) & (alph.features == feat)]["alpha"]
        if q.empty:
            continue
        W(f"{DRUG_LABEL[drug]} & {esc(feat)} & {q.median():.3g} & {q.min():.3g} & {q.max():.3g} \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\medskip")
W(r"\textbf{(b) Paired differences against each linear comparator.}")
V19_LABEL = {
 "tuned_vs_fixed_ridge_base": "Tuned Ridge $-$ Ridge ($\\alpha=1$), base",
 "tuned_vs_fixed_ridge_hist": "Tuned Ridge $-$ Ridge ($\\alpha=1$), $+$history",
 "history_effect_on_ridge_tuned": "History effect on tuned Ridge",
 "history_effect_on_median": "History effect on median regression",
 "algorithm_effect_base_vs_ridge_tuned": "XGBoost $-$ tuned Ridge, base",
 "algorithm_effect_hist_vs_ridge_tuned": "XGBoost $-$ tuned Ridge, $+$history",
 "algorithm_effect_base_vs_median": "XGBoost $-$ median regression, base",
 "algorithm_effect_hist_vs_median": "XGBoost $-$ median regression, $+$history",
 "median_vs_ridge_tuned_base": "Median regression $-$ tuned Ridge, base",
 "ridge_tuned_hist_vs_baseline": "Tuned Ridge with history $-$ baseline-MAP",
 "median_hist_vs_baseline": "Median regression with history $-$ baseline-MAP",
}
W(r"\footnotesize")
W(r"\begin{longtable}{p{6.4cm}rccc}")
W(r"\toprule Comparison & $n$ & $\Delta$MAE (95\% CI) & SD & Range \\ \midrule \endhead")
for drug in DRUG_ORDER:
    sub = v19d[v19d.drug == drug]
    if sub.empty:
        continue
    W(rf"\multicolumn{{5}}{{l}}{{\textbf{{{DRUG_LABEL[drug]}}} ({esc(sub['role'].iloc[0])})}} \\")
    for key, lab in V19_LABEL.items():
        r_ = sub[sub.comparison == key]
        if r_.empty:
            continue
        r_ = r_.iloc[0]
        W(f"{lab} & {int(r_['n']):,} & {signed_ci(r_['dmae'], r_['ci_lo'], r_['ci_hi'])} & "
          f"{fnum(r_['sd_across_partitions'],3)} & {signed(r_['min_partition'])} to {signed(r_['max_partition'])} \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\clearpage")

# ---------- S12: baseline-MAP-only probability model ----------
vc = r16("classifier_vs_maponly.csv")
vd = r16("classifier_paired_diff.csv")
W(r"\section*{Supplementary Table S12. Probability model versus a baseline-MAP-only probability model}")
W(r"\addcontentsline{toc}{section}{Table S12. Baseline-MAP-only probability model}")
W(r"""The main-text Brier skill compares each classifier with an out-of-fold prevalence null, which is
a weak reference: a clinician asking whether to give a vasopressor already knows the current pressure.
We therefore fitted a probability model on pre-dose MAP alone (logistic regression on a single
variable, median-imputed within each training fold) on the identical folds, and computed the
increment of the full classifiers over it. Panel (a) gives discrimination, Brier score and the Brier
skill relative to the MAP-only model; panel (b) gives the paired difference (full model minus MAP-only
model) with a patient-cluster bootstrap 95\% CI that resamples the same patients for both models.""")
W(r"\textbf{(a) Performance of the MAP-only reference and the full classifiers.}")
W(r"\footnotesize")
W(r"\begin{longtable}{llrrcccc}")
W(r"\toprule Drug & Scope & $n$ & Model & AUROC (95\% CI) & Brier & Calib.\ slope & Skill vs MAP-only \\ \midrule \endhead")
MLAB = {"logit_map": "MAP only", "logit": "Logistic", "xgb": "XGBoost"}
for drug in DRUG_ORDER:
    for scope in ["whole", "below65"]:
        for model in ["logit_map", "logit", "xgb"]:
            r_ = vc[(vc.drug == drug) & (vc.scope == scope) & (vc.model == model)]
            if r_.empty:
                continue
            r_ = r_.iloc[0]
            sk = "---" if pd.isna(r_["brier_skill_vs_maponly"]) else f"{fnum(r_['brier_skill_vs_maponly']*100,1)}\\%"
            W(f"{DRUG_LABEL[drug]} & {esc(scope)} & {int(r_['n']):,} & {MLAB[model]} & "
              f"{fnum(r_['auroc_mean'],3)} ({fnum(r_['auroc_lo'],3)}--{fnum(r_['auroc_hi'],3)}) & "
              f"{fnum(r_['brier_mean'],3)} & {fnum(r_['calib_slope_mean'],3)} & {sk} \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\medskip")
W(r"\textbf{(b) Paired difference from the MAP-only model (positive favours the full model for AUROC, negative for Brier).}")
W(r"\footnotesize")
W(r"\begin{longtable}{lllrc}")
W(r"\toprule Drug & Scope & Model & Metric & Difference (95\% CI) \\ \midrule \endhead")
for drug in DRUG_ORDER:
    for scope in ["whole", "below65"]:
        for model in ["logit", "xgb"]:
            for mname in ["auroc", "brier"]:
                r_ = vd[(vd.drug == drug) & (vd.scope == scope) & (vd.model == model) & (vd.metric == mname)]
                if r_.empty:
                    continue
                r_ = r_.iloc[0]
                W(f"{DRUG_LABEL[drug]} & {esc(scope)} & {MLAB[model]} & {mname.upper() if mname=='auroc' else 'Brier'} & "
                  f"{signed(r_['diff_vs_maponly'],3)} ({signed(r_['ci_lo'],3)} to {signed(r_['ci_hi'],3)}) \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\clearpage")

# ---------- S13: no-further-vasopressor sensitivity cohort ----------
scq = r16("sens_quality.csv")
scd = r16("sens_delta_mae.csv")
scc = r16("sens_classifier.csv")
scoh = r16("sens_cohort.csv")
W(r"\section*{Supplementary Table S13. Sensitivity analysis excluding administrations followed by further vasopressor treatment}")
W(r"\addcontentsline{toc}{section}{Table S13. No-further-vasopressor sensitivity cohort}")
W(r"""In the primary analysis the outcome is the observed pressure change, which a clinician's further
treatment can itself alter. We therefore re-fitted every model on the subset of administrations with
no further vasopressor record (a different drug or a same-drug re-dose) at or before $+6$ minutes,
the end of the $+4$ to $+6$ minute outcome window, and separately on the excluded complement. An
administration followed only at $+7$ to $+10$ minutes is retained. This subset is created by a
selection on post-dose care, so it is a sensitivity analysis and not an unbiased estimate of the
drug's physiological effect. All quantities are per-partition means with patient-cluster bootstrap
95\% CIs; folds were redrawn within each cohort by the same patient-grouped rule.""")
W(r"\textbf{(a) Cohort sizes and explained variance.}")
W(r"\footnotesize")
W(r"\begin{longtable}{llrrrrr}")
W(r"\toprule Cohort & Drug & $n$ & Patients & Obs.\ $\Delta$MAP & $R^2$ (95\% CI) & $R^2$ baseline-MAP \\ \midrule \endhead")
for drug in DRUG_ORDER_PRIMARY:
    for c in ["sens", "excluded"]:
        r_ = scq[(scq.drug == drug) & (scq.cohort == c)]
        if r_.empty:
            continue
        r_ = r_.iloc[0]
        lab = "No further vasopressor" if c == "sens" else "Excluded complement"
        W(f"{lab} & {DRUG_LABEL[drug]} & {int(r_['n']):,} & {int(r_['n_patients']):,} & "
          f"{signed(r_['observed_delta_mean'],2)} & {fnum(r_['r2_mean'],3)} ({fnum(r_['r2_lo'],3)}--{fnum(r_['r2_hi'],3)}) & "
          f"{fnum(r_['r2_baseline_only'],3)} \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\medskip")
W(r"\textbf{(b) Paired differences in the sensitivity cohort and the excluded complement.}")
SENS_LABEL = {
 "history_effect_on_ridge_tuned": "History effect on tuned Ridge",
 "history_effect_on_xgb": "History effect on XGBoost",
 "algorithm_effect_base_vs_ridge_tuned": "XGBoost $-$ tuned Ridge, base",
 "algorithm_effect_hist_vs_ridge_tuned": "XGBoost $-$ tuned Ridge, $+$history",
 "best_candidate_vs_baseline": "Preselected XGBoost $-$ baseline-MAP",
}
W(r"\footnotesize")
W(r"\begin{longtable}{p{6.2cm}lrr}")
W(r"\toprule Comparison & Cohort & $n$ & $\Delta$MAE (95\% CI) \\ \midrule \endhead")
for drug in DRUG_ORDER_PRIMARY:
    W(rf"\multicolumn{{4}}{{l}}{{\textbf{{{DRUG_LABEL[drug]}}}}} \\")
    for key, lab in SENS_LABEL.items():
        for c in ["sens", "excluded"]:
            r_ = scd[(scd.drug == drug) & (scd.cohort == c) & (scd.comparison == key)]
            if r_.empty:
                continue
            r_ = r_.iloc[0]
            cl = "No further" if c == "sens" else "Excluded"
            W(f"{lab} & {cl} & {int(r_['n']):,} & {signed_ci(r_['dmae'], r_['ci_lo'], r_['ci_hi'])} \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\medskip")
W(r"\textbf{(c) Probability model in the sensitivity cohort.}")
W(r"\footnotesize")
W(r"\begin{longtable}{llrrccc}")
W(r"\toprule Drug & Scope & $n$ & Model & AUROC (95\% CI) & Brier & Skill vs MAP-only \\ \midrule \endhead")
for drug in DRUG_ORDER_PRIMARY:
    for scope in ["whole", "below65"]:
        for model in ["logit_map", "xgb"]:
            r_ = scc[(scc.drug == drug) & (scc.cohort == "sens") & (scc.scope == scope) & (scc.model == model)]
            if r_.empty:
                continue
            r_ = r_.iloc[0]
            sk = "---" if pd.isna(r_["brier_skill_vs_maponly"]) else f"{fnum(r_['brier_skill_vs_maponly']*100,1)}\\%"
            W(f"{DRUG_LABEL[drug]} & {esc(scope)} & {int(r_['n']):,} & {MLAB[model]} & "
              f"{fnum(r_['auroc_mean'],3)} ({fnum(r_['auroc_lo'],3)}--{fnum(r_['auroc_hi'],3)}) & "
              f"{fnum(r_['brier_mean'],3)} & {sk} \\\\")
W(r"\bottomrule \end{longtable}")
W(r"\normalsize")
W(r"\clearpage")



# ---------- Figure S1 ----------
W(r"\section*{Supplementary Figure S1. Prediction error and observed change by post-dose intervention}")
W(r"\addcontentsline{toc}{section}{Figure S1. Error by post-dose intervention}")
W(r"\begin{figure}[htbp]\centering\includegraphics[width=0.92\textwidth]{figures/figS_intervention.png}")
W(r"""\caption{Supplementary Figure S1. Out-of-fold mean absolute error (bars from zero; $n$ per
bar) of the preselected XGBoost reference model for administrations with and without further treatment in
the 5-minute outcome window, per primary drug, under three overlapping stratifications: any
additional vasopressor including a same-drug re-dose (left), a different-drug vasopressor
(centre) and a same-drug re-dose (right). The left stratum is the union of the other two.
Error is higher when the outcome window contains further treatment; because that treatment
responds to an inadequate pressure, the association has a circular component and is descriptive
only. Observed mean $\Delta$MAP for each stratum is given in Table~S7.}""")
W(r"\label{fig:s1}\end{figure}")
W(r"\end{document}")

# ========================= WRITE =========================
OUT = ROOT / "manuscript" / "supplement"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "supplementary_information.tex").write_text("\n".join(P) + "\n", encoding="utf-8")
print(f"WROTE {OUT/'supplementary_information.tex'} ({len(P)} lines)")
