#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: v3_step8_figures.py | Topic: 2 (v3) | Purpose:
Publication figures from the REAL MOVER results (returned) + VitalDB development
cohort. All inputs are the returned outputs/ CSVs/parquets (read-only) plus the
local IV-sensitivity CSV. 300 DPI, colourblind-friendly.

Figures:
  Fig1  CONSORT-style flow (gate counts from m0 + event summary)
  Fig2  per-drug ΔMAP@5min distribution (violin) with n annotations
  Fig3  per-drug MAE across 3 models (XGBoost/Ridge/baseline-MAP) with 95% CI
  Fig4  calibration slope + PI80 coverage per drug (XGBoost)
  Fig5  VitalDB->MOVER external validation: MAE + calib slope before/after recalib
  Fig6  IV-only vs all-route MAE (sensitivity) highlighting epinephrine
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 300, 'savefig.dpi': 300})
CB = {'xgboost': '#0072B2', 'ridge': '#E69F00', 'baseline_map_only': '#009E73'}

WORK = Path(os.environ.get("TOPIC2_WORK", Path(__file__).resolve().parents[1]))
RET = WORK / "mover_returned" / "outputs"
OUT = WORK / "outputs_v3"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DRUG_ORDER = ['phenylephrine', 'ephedrine', 'vasopressin', 'norepinephrine', 'epinephrine']


def _lab(d): return d.capitalize()


def fig1_consort():
    gate = json.loads((RET / "m0_dose_audit.json").read_text())
    es = pd.read_csv(RET / "m1_event_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 8)); ax.axis('off')
    boxes = [
        (f"MOVER vasopressor MAR rows\nn = {gate['n_vaso_mar_rows']:,}", 0.9),
        (f"START administrations in valid surgery windows\nn = {gate['n_start_admin']:,}", 0.74),
        ("Gate G1 timestamp 100% | G2 dose+unit 99.7%\nG3 actions fully mapped | G4 ≥ 1000  → GO", 0.58),
        (f"START with arterial-MAP linkage (eligible)\nn = {gate['G4_START_events']:,}", 0.42),
        ("Per-drug analysis cohorts (START, modelled)", 0.26),
    ]
    for txt, y in boxes:
        ax.annotate(txt, xy=(0.5, y), ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.5', fc='#EAF2FB', ec='#0072B2'))
        if y > 0.26:
            ax.annotate('', xy=(0.5, y - 0.065), xytext=(0.5, y - 0.012),
                        arrowprops=dict(arrowstyle='-|>', color='#555'))
    tbl = "\n".join(f"{_lab(r.drug):<16} n={int(r.n_start):>6,}  patients={int(r.n_patients):>5,}"
                    for r in es.sort_values('n_start', ascending=False).itertuples())
    ax.annotate(tbl, xy=(0.5, 0.12), ha='center', va='center', family='monospace', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#999'))
    ax.set_title("Figure 1. Documented vasopressor administration cohort (MOVER)", fontsize=11)
    fig.savefig(FIG / "fig1_consort.png", bbox_inches='tight'); plt.close(fig)


def fig2_delta_dist():
    drugs = [d for d in DRUG_ORDER if (RET / f"oof_mover_{d}.parquet").exists()]
    data, labs = [], []
    for d in drugs:
        df = pd.read_parquet(RET / f"oof_mover_{d}.parquet")
        data.append(df['delta_map_5min'].values); labs.append(f"{_lab(d)}\nn={len(df):,}")
    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for b in parts['bodies']:
        b.set_facecolor('#0072B2'); b.set_alpha(0.5)
    ax.axhline(0, color='#888', lw=0.8, ls='--')
    ax.set_xticks(range(1, len(labs) + 1)); ax.set_xticklabels(labs)
    ax.set_ylabel("ΔMAP at 5 min (mmHg)")
    ax.set_title("Figure 2. Observed arterial-pressure response after documented administration")
    ax.set_ylim(-40, 60)
    fig.savefig(FIG / "fig2_delta_distribution.png", bbox_inches='tight'); plt.close(fig)


def fig3_mae():
    nat = pd.read_csv(RET / "m2_mover_native.csv")
    nat = nat[nat.model != 'INSUFFICIENT_N']
    drugs = [d for d in DRUG_ORDER if d in nat.drug.values]
    models = ['xgboost', 'ridge', 'baseline_map_only']
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.26; x = np.arange(len(drugs))
    for i, mdl in enumerate(models):
        sub = nat[nat.model == mdl].set_index('drug').reindex(drugs)
        mae = sub['MAE'].values
        lo = mae - sub['MAE_lo'].values; hi = sub['MAE_hi'].values - mae
        ax.bar(x + (i - 1) * w, mae, w, yerr=[lo, hi], capsize=3,
               label=mdl.replace('_', ' '), color=CB[mdl])
    ax.set_xticks(x); ax.set_xticklabels([_lab(d) for d in drugs])
    ax.set_ylabel("MAE (mmHg), patient-grouped OOF")
    ax.set_title("Figure 3. Per-drug prediction error with 95% CI (MOVER-native)")
    ax.legend(frameon=False)
    fig.savefig(FIG / "fig3_mae_by_drug.png", bbox_inches='tight'); plt.close(fig)


def fig4_calib_pi():
    nat = pd.read_csv(RET / "m2_mover_native.csv")
    xgb = nat[nat.model == 'xgboost'].set_index('drug')
    drugs = [d for d in DRUG_ORDER if d in xgb.index]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.bar([_lab(d) for d in drugs], [xgb.loc[d, 'calib_slope'] for d in drugs], color='#0072B2')
    a1.axhline(1.0, color='#D55E00', ls='--', label='ideal = 1.0')
    a1.set_ylabel("Calibration slope"); a1.set_title("(a) Calibration slope (XGBoost)")
    a1.legend(frameon=False); a1.tick_params(axis='x', rotation=20)
    cov = [xgb.loc[d, 'PI80_coverage'] for d in drugs]
    a2.bar([_lab(d) for d in drugs], cov, color='#009E73')
    a2.axhline(0.80, color='#D55E00', ls='--', label='nominal = 0.80')
    a2.set_ylabel("Empirical coverage"); a2.set_title("(b) 80% prediction-interval coverage")
    a2.set_ylim(0, 1); a2.legend(frameon=False); a2.tick_params(axis='x', rotation=20)
    fig.suptitle("Figure 4. Calibration and interval coverage (MOVER-native XGBoost)")
    fig.savefig(FIG / "fig4_calibration_pi.png", bbox_inches='tight'); plt.close(fig)


def fig5_external():
    ext = pd.read_csv(RET / "m2_external_validation.csv")
    drugs = sorted(ext.drug.unique())
    stages = ['before_recalib', 'linear_recalib_group_oof', 'isotonic_recalib_group_oof']
    slab = ['Before', 'Linear recalib', 'Isotonic recalib']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    w = 0.25; x = np.arange(len(drugs))
    for i, st in enumerate(stages):
        sub = ext[ext.stage == st].set_index('drug').reindex(drugs)
        mae = sub['MAE'].values
        lo = mae - sub['MAE_lo'].values; hi = sub['MAE_hi'].values - mae
        a1.bar(x + (i - 1) * w, mae, w, yerr=[lo, hi], capsize=3, label=slab[i])
        a2.bar(x + (i - 1) * w, sub['calib_slope'].values, w, label=slab[i])
    a1.set_xticks(x); a1.set_xticklabels([_lab(d) for d in drugs]); a1.set_ylabel("MAE (mmHg)")
    a1.set_title("(a) VitalDB→MOVER MAE"); a1.legend(frameon=False, fontsize=8)
    a2.set_xticks(x); a2.set_xticklabels([_lab(d) for d in drugs]); a2.set_ylabel("Calibration slope")
    a2.axhline(1.0, color='#D55E00', ls='--'); a2.set_title("(b) Calibration slope")
    a2.legend(frameon=False, fontsize=8)
    fig.suptitle("Figure 5. External transportability of frozen VitalDB models (retrospective)")
    fig.savefig(FIG / "fig5_external_recalib.png", bbox_inches='tight'); plt.close(fig)


def fig6_iv_sensitivity():
    allr = pd.read_csv(RET / "m2_mover_native.csv")
    ivo = pd.read_csv(OUT / "metrics" / "m2_mover_native_IVonly.csv")
    xa = allr[allr.model == 'xgboost'].set_index('drug')
    xi = ivo[ivo.model == 'xgboost'].set_index('drug')
    drugs = [d for d in DRUG_ORDER if d in xa.index and d in xi.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.38; x = np.arange(len(drugs))
    ax.bar(x - w / 2, [xa.loc[d, 'MAE'] for d in drugs], w, label='All routes', color='#999')
    ax.bar(x + w / 2, [xi.loc[d, 'MAE'] for d in drugs], w, label='IntraVENOUS only', color='#0072B2')
    for i, d in enumerate(drugs):
        n_all = int(xa.loc[d, 'n']); n_iv = int(xi.loc[d, 'n'])
        ax.annotate(f"IV {100*n_iv/n_all:.0f}%", (i, max(xa.loc[d, 'MAE'], xi.loc[d, 'MAE']) + 0.3),
                    ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([_lab(d) for d in drugs])
    ax.set_ylabel("MAE (mmHg), XGBoost")
    ax.set_title("Figure 6. Route-restriction sensitivity (epinephrine is majority non-IV)")
    ax.legend(frameon=False)
    fig.savefig(FIG / "fig6_iv_sensitivity.png", bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    fig1_consort(); fig2_delta_dist(); fig3_mae(); fig4_calib_pi(); fig5_external(); fig6_iv_sensitivity()
    print("figures written to", FIG)
    for p in sorted(FIG.glob("fig*.png")):
        print(" ", p.name)
