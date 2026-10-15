#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: v3_step5_figures.py | Topic: 2 (v3) | Purpose:
Publication figures for the documented-administration design.
  Fig1 CONSORT-style flow (documented events only, exclusions counted)
  Fig2 ΔMAP@5min distribution per drug
  Fig3 predicted vs observed (OOF) per drug w/ calibration line + 80% PI band
  Fig4 performance bar: XGB vs Ridge vs baseline-MAP-only per drug
"""
import os, json, logging, argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = Path(os.environ.get("TOPIC2_WORK", Path(__file__).resolve().parents[1]))
RAW_ROOTS = [Path(os.environ.get("VITALDB_ROOT", "/path/to/vitaldb/physionet.org")),
             Path(os.environ.get("INSPIRE_ROOT", "/path/to/inspire/physionet.org")),
             Path(os.environ.get("MIMIC4_ROOT", "/path/to/mimic/physionet.org"))]


def safe_path(p):
    r = str(Path(p).resolve())
    for root in RAW_ROOTS:
        assert not r.startswith(str(root.resolve())), f"FATAL write to raw: {root}"
    return Path(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default=str(WORK / "outputs_v3/metrics/vitaldb_documented_events.parquet"))
    ap.add_argument('--cohort', default='vitaldb')
    ap.add_argument('--outdir', default=str(WORK / "outputs_v3"))
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(out / "logs" / f"{ts}_v3_step5_fig_{args.cohort}.log"),
                  logging.StreamHandler()])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

    ev = pd.read_parquet(args.events)
    starts = ev[ev['event_type'] == 'start'].copy()
    drugs = sorted(starts['drug'].unique())
    PAL = {'phenylephrine': '#4C78A8', 'norepinephrine': '#E45756',
           'ephedrine': '#54A24B', 'epinephrine': '#F58518', 'vasopressin': '#B279A2'}

    # Fig2 distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    for d in drugs:
        v = starts[starts['drug'] == d]['delta_map_5min'].clip(-40, 80)
        ax.hist(v, bins=30, alpha=0.55, density=True, label=f'{d} (n={len(v)})',
                color=PAL.get(d, '#777'))
    ax.axvline(0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('ΔMAP at 5 min (mmHg)'); ax.set_ylabel('density')
    ax.set_title(f'{args.cohort}: documented vasopressor MAP response', fontweight='bold')
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(safe_path(out / "figures" / f"v3_fig2_dmap_dist_{args.cohort}.png"),
                                    dpi=300, bbox_inches='tight'); plt.close()

    # Fig3 pred vs observed per drug
    present = [d for d in drugs if (out / "metrics" / f"oof_predictions_{args.cohort}_{d}.parquet").exists()]
    if present:
        fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4.5), squeeze=False)
        for j, d in enumerate(present):
            od = pd.read_parquet(out / "metrics" / f"oof_predictions_{args.cohort}_{d}.parquet")
            od = od[~od['pred_xgb'].isna()]
            ax = axes[0][j]
            ax.scatter(od['pred_xgb'], od['delta_map_5min'], s=14, alpha=0.5, color=PAL.get(d, '#777'))
            lim = [-40, 80]
            ax.plot(lim, lim, 'k--', lw=0.8, label='ideal')
            if od['pred_xgb'].notna().sum() > 2:
                b, a = np.polyfit(od['pred_xgb'], od['delta_map_5min'], 1)
                xs = np.array(lim); ax.plot(xs, a + b * xs, color='#333', lw=1.3,
                                            label=f'calib slope={b:.2f}')
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel('predicted ΔMAP (mmHg)'); ax.set_ylabel('observed ΔMAP (mmHg)')
            ax.set_title(f'{d} (n={len(od)})', fontweight='bold'); ax.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(safe_path(out / "figures" / f"v3_fig3_pred_obs_{args.cohort}.png"),
                                        dpi=300, bbox_inches='tight'); plt.close()

    # Fig4 performance bars
    perfp = out / "metrics" / f"v3_model_performance_{args.cohort}.csv"
    if perfp.exists():
        perf = pd.read_csv(perfp)
        perf = perf[perf['model'].isin(['xgboost', 'ridge', 'baseline_map_only'])]
        if len(perf):
            fig, ax = plt.subplots(figsize=(8, 4))
            models = ['baseline_map_only', 'ridge', 'xgboost']
            width = 0.25
            dl = sorted(perf['drug'].unique())
            x = np.arange(len(dl))
            for k, mdl in enumerate(models):
                vals = [perf[(perf.drug == d) & (perf.model == mdl)]['MAE'].values[:1] for d in dl]
                vals = [v[0] if len(v) else np.nan for v in vals]
                ax.bar(x + (k - 1) * width, vals, width, label=mdl)
            ax.set_xticks(x); ax.set_xticklabels(dl)
            ax.set_ylabel('MAE (mmHg)'); ax.set_title(f'{args.cohort}: MAE by model & drug', fontweight='bold')
            ax.legend(fontsize=8)
            plt.tight_layout(); plt.savefig(safe_path(out / "figures" / f"v3_fig4_mae_{args.cohort}.png"),
                                            dpi=300, bbox_inches='tight'); plt.close()

    # Fig1 CONSORT text panel
    fig, ax = plt.subplots(figsize=(7, 5)); ax.axis('off')
    n_start = len(starts)
    n_total = len(ev)
    lines = [
        f"{args.cohort.upper()} — documented vasopressor administrations",
        "",
        f"Pump/MAR-documented events detected: {n_total}",
        f"  START events: {int((ev['event_type']=='start').sum())}",
        f"  rate_increase (sensitivity): {int((ev['event_type']=='rate_increase').sum())}",
        f"  censored (same-drug re-dose): {int(ev['censored_repeat'].fillna(False).sum())}",
        "",
        "Primary analysis set (START, uncensored):",
        f"  n = {n_start}  across {starts['caseid'].nunique()} cases / "
        f"{starts.get('subjectid', starts['caseid']).nunique()} patients",
        "",
        "By drug:",
    ] + [f"  {d}: {int((starts['drug']==d).sum())}" for d in drugs]
    ax.text(0.03, 0.97, "\n".join(lines), va='top', ha='left', family='monospace', fontsize=10)
    plt.tight_layout(); plt.savefig(safe_path(out / "figures" / f"v3_fig1_consort_{args.cohort}.png"),
                                    dpi=300, bbox_inches='tight'); plt.close()
    logging.info("=== Step 5 figures complete ===")


if __name__ == '__main__':
    main()
