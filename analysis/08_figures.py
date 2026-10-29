#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main-text figures 2-4 and the supplementary follow-up figure, from the per-partition
result tables (no retraining).

Figure 2: pre-specified paired dMAE comparisons with cluster-bootstrap CIs.
Figure 3: R^2, MAE/SD and large-error proportions of the primary candidate model, with CIs.
Figure 4: calibrated probability model — binned reliability (whole cohort, below-65) and
          AUROC with CIs.
Figure S1: MAE by follow-up treatment in the outcome window (three overlapping strata).
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results'
A = R
V = R
FIG = R / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
D = [("phenylephrine", "Phenylephrine"), ("ephedrine", "Ephedrine"), ("vasopressin", "Vasopressin")]
plt.rcParams.update({"font.size": 9})


def hide(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ---------------- Figure 2: three pre-specified paired comparisons ----------------
d = pd.read_csv(A / 'delta_mae.csv')
C = [("history_effect_on_xgb", "Treatment history\n(XGB+H - XGB)"),
     ("algorithm_effect_base", "Algorithm\n(XGB - Ridge, base)"),
     ("best_candidate_vs_baseline", "Candidate vs baseline-MAP\n(XGB+H - baseline)")]
sel = d[d.comparison.isin([c for c, _ in C]) & d.drug.isin([dd for dd, _ in D])]
xmin = min(sel.ci_lo.min(), -1.1); xmax = max(sel.ci_hi.max(), 0.3)
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), sharey=True, sharex=True)
for ax, (drug, lab) in zip(axes, D):
    s = d[d.drug == drug].set_index("comparison")
    ys = np.arange(len(C))[::-1]
    for y_, (c, _) in zip(ys, C):
        r = s.loc[c]
        col = ["#2C6E9B", "#E0A33C", "#3FA796"][list(ys).index(y_)]
        ax.errorbar(r.dmae, y_, xerr=[[r.dmae - r.ci_lo], [r.ci_hi - r.dmae]],
                    fmt="o", color=col, capsize=3, lw=1.3)
    ax.axvline(0, color="#555", lw=0.9)
    ax.axvline(-1, color="#bbb", lw=0.9, ls="--")
    ax.set_xlim(xmin, xmax)
    ax.set_title(f"{lab}\n({int(s.n.iloc[0]):,} administrations)", fontsize=9)
    ax.set_xlabel("Paired dMAE (mmHg), 95% CI")
    hide(ax)
axes[0].set_yticks(ys); axes[0].set_yticklabels([l for _, l in C], fontsize=8)
fig.suptitle("Negative favours the first-named model. Dashed line: dMAE = -1 mmHg, "
             "an investigator-set descriptive benchmark (not a validated MCID).",
             fontsize=8, y=-0.02)
fig.tight_layout(); fig.savefig(FIG / "fig2_delta_mae.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 3: explanatory performance, SEPARATE panels --------------
q = pd.read_csv(V / 'regression_quality.csv').set_index('drug')
q = q.loc[[dd for dd, _ in D]]
x = np.arange(3)
fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.2))
# (a) R^2 with per-partition-mean cluster-bootstrap CI
axes[0].bar(x, q.r2_mean, 0.55, color="#2C6E9B",
            yerr=[q.r2_mean - q.r2_boot_lo, q.r2_boot_hi - q.r2_mean], capsize=3)
axes[0].set_xticks(x); axes[0].set_xticklabels([l for _, l in D]); axes[0].set_ylim(0, 0.6)
axes[0].set_ylabel("Out-of-fold $R^2$")
axes[0].set_title("(a) Variance explained", fontsize=9)
# (b) MAE / SD
axes[1].bar(x, q.mae_over_sd_mean, 0.55, color="#888888")
axes[1].axhline(1.0, color="#bbb", lw=0.9, ls="--")
axes[1].set_xticks(x); axes[1].set_xticklabels([l for _, l in D]); axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel("MAE / outcome SD")
axes[1].set_title("(b) Error relative to outcome spread", fontsize=9)
# (c) large-error distribution with bootstrap 95% CI
groups = [('pct_err_gt_5', '> 5 mmHg', "#CCCCCC"),
          ('pct_err_gt_10', '> 10 mmHg', "#2C6E9B"),
          ('pct_err_gt_15', '> 15 mmHg', "#12324A")]
for k, (base, lab, col) in enumerate(groups):
    mean = q[f'{base}_mean']; lo = q[f'{base}_lo']; hi = q[f'{base}_hi']
    axes[2].bar(x + (k - 1) * 0.26, mean, 0.26, color=col, label=lab,
                yerr=[mean - lo, hi - mean], capsize=2,
                error_kw={"elinewidth": 0.8})
axes[2].set_xticks(x); axes[2].set_xticklabels([l for _, l in D])
axes[2].set_ylabel("% of predictions"); axes[2].set_ylim(0, 100)
axes[2].set_title("(c) Absolute-error distribution", fontsize=9)
axes[2].legend(frameon=False, fontsize=7.5, title="error exceeds", title_fontsize=7.5,
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=1.2)
for ax in axes:
    hide(ax)
fig.tight_layout(); fig.savefig(FIG / "fig3_model_quality.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 4: Route A probability model -----------------------------
m = pd.read_csv(V / 'classifier_metrics.csv')
cur = pd.read_csv(V / 'calibration_curve.csv')
mx = m[(m.model == 'xgb') & (m.drug.isin([dd for dd, _ in D]))].set_index('drug')
cols = {"phenylephrine": "#2C6E9B", "ephedrine": "#E0A33C", "vasopressin": "#3FA796"}
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
# (a) binned reliability, whole cohort, with per-bin CI
axes[0].plot([0, 1], [0, 1], color="#bbb", lw=0.9, ls="--")
for drug, lab in D:
    c = cur[(cur.drug == drug) & (cur.scope == 'overall')].sort_values('pred_mean')
    axes[0].errorbar(c.pred_mean, c.obs_rate,
                     yerr=[c.obs_rate - c.obs_lo, c.obs_hi - c.obs_rate],
                     fmt="o", color=cols[drug], ms=3, lw=0, elinewidth=1.0,
                     capsize=2, label=lab)
axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
axes[0].set_xlabel("Mean predicted probability per decile")
axes[0].set_ylabel("Observed attainment (95% CI)")
axes[0].set_title("(a) Binned calibration, whole cohort", fontsize=9)
axes[0].legend(frameon=False, fontsize=8)
# (b) binned reliability, below-65 subgroup
axes[1].plot([0, 1], [0, 1], color="#bbb", lw=0.9, ls="--")
for drug, lab in D:
    c = cur[(cur.drug == drug) & (cur.scope == 'baseline_lt65')].sort_values('pred_mean')
    if len(c):
        axes[1].errorbar(c.pred_mean, c.obs_rate,
                         yerr=[c.obs_rate - c.obs_lo, c.obs_hi - c.obs_rate],
                         fmt="o", color=cols[drug], ms=3, lw=0, elinewidth=1.0,
                         capsize=2, label=lab)
axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
axes[1].set_xlabel("Mean predicted probability per bin")
axes[1].set_ylabel("Observed attainment (95% CI)")
axes[1].set_title("(b) Binned calibration, baseline MAP < 65", fontsize=9)
axes[1].legend(frameon=False, fontsize=8)
# (c) AUROC with CI, whole cohort vs below-65
sub = pd.read_csv(V / 'classifier_subgroup.csv')
sub = sub[sub.model == 'xgb'].set_index('drug')
y = np.arange(3)[::-1]
for yi, (drug, lab) in zip(y, D):
    r = mx.loc[drug]
    axes[2].errorbar(r.auroc_mean, yi + 0.14,
                     xerr=[[r.auroc_mean - r.auroc_boot_lo], [r.auroc_boot_hi - r.auroc_mean]],
                     fmt="o", color="#2C6E9B", capsize=3, lw=1.3,
                     label="whole cohort" if yi == 2 else None)
    s = sub.loc[drug]
    axes[2].errorbar(s.auroc, yi - 0.14,
                     xerr=[[s.auroc - s.auroc_lo], [s.auroc_hi - s.auroc]],
                     fmt="s", color="#D55E00", capsize=3, lw=1.3,
                     label="baseline MAP < 65" if yi == 2 else None)
axes[2].axvline(0.5, color="#bbb", lw=0.9, ls="--")
axes[2].set_yticks(y); axes[2].set_yticklabels([l for _, l in D], fontsize=8)
axes[2].set_xlabel("AUROC (95% CI)"); axes[2].set_xlim(0.45, 0.85)
axes[2].set_title("(c) Discrimination", fontsize=9)
axes[2].legend(frameon=False, fontsize=7.5)
for ax in axes:
    hide(ax)
fig.tight_layout(); fig.savefig(FIG / "fig4_classifier.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------- Supplementary figure: intervention (y-axis from 0) --------------
fu_path = V / 'followup_strata.csv'
if fu_path.exists():
    fu = pd.read_csv(fu_path)
    fu = fu[fu.role == 'primary']
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.3))
    labels = ["any additional vasopressor within 5 min (incl. same-drug re-dose)",
              "different-drug vasopressor within 5 min", "same-drug re-dose within 5 min"]
    for ax, lab in zip(axes, labels):
        for k, pr in enumerate((0, 1)):
            rows = [fu[(fu.drug == dd) & (fu.intervention == lab) & (fu.present == pr)] for dd, _ in D]
            v = [float(r.mae.iloc[0]) if len(r) else np.nan for r in rows]
            nn = [int(r.n.iloc[0]) if len(r) else 0 for r in rows]
            ax.bar(x + (k - 0.5) * 0.38, v, 0.38, color=["#3FA796", "#D55E00"][k],
                   label=["not followed", "followed"][k])
            for xi, vi, ni in zip(x + (k - 0.5) * 0.38, v, nn):
                if not np.isnan(vi):
                    ax.text(xi, vi + 0.1, f"n={ni:,}", ha="center", fontsize=6.5, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in D]); ax.set_ylim(0, 17)
        ax.set_title((lab[0].upper() + lab[1:]).replace(" (incl.", "\n(incl."), fontsize=8.5)
        hide(ax)
    axes[0].set_ylabel("MAE (mmHg), primary candidate model")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "figS_intervention.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

print("figures written:", sorted(p.name for p in FIG.iterdir()))
