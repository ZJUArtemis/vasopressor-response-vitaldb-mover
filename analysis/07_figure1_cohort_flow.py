#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 1: cohort flow for the documented-exposure cohort.

The flow closes at the arterial-MAP-linked exposure-classification stage (the four
mutually exclusive classes sum to the MAP-linked study-drug set). Upstream exclusions
that cannot be decomposed into mutually exclusive reasons from the returned tables are
labelled as such. All counts are read from results/cohort_flow_full.csv.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results'
FIG = R / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
f = pd.read_csv(R / 'cohort_flow_full.csv').set_index('step')


def n(step):
    return int(f.loc[step, 'n_events'])


def p(step):
    v = f.loc[step, 'n_patients']
    return None if pd.isna(v) else int(v)


plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(7.8, 9.0))
ax.axis("off")


def box(x, y, t, fc="#EEF4FB", ec="#1F5F99", fs=8.5):
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=.45", fc=fc, ec=ec, lw=1.3))


def arr(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3))


# --- top: database sources ---
box(.46, .965,
    "MOVER perioperative database\n"
    f"documented vasopressor MAR rows n = {n('A_raw_documented_MAR_rows_all_vasopressors'):,}; "
    f"arterial-MAP samples n = {n('B_raw_arterial_MAP_rows'):,}\n"
    f"cases with >=1 documented candidate administration n = {p('C_cases_with_documented_candidates'):,}",
    fs=8)
arr(.46, .925, .46, .885)

# exclusion note (kept clear of the central column, lower-right)
ax.text(.93, .80,
        "Excluded before linkage\n(counts not decomposable\ninto mutually exclusive\nreasons from returned\ntables):\n"
        "* epinephrine (not a\n   study drug)\n"
        "* no eligible pre/post\n   arterial-MAP window",
        ha="center", va="center", fontsize=6.6, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=.3", fc="#F7F7F7", ec="#999", lw=0.9))
ax.annotate("", xy=(.78, .80), xytext=(.46, .855), xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.0, ls="--"))

box(.46, .80,
    "Study-drug administrations with an eligible\n"
    "pre- and post-dose arterial-MAP window\n"
    f"n = {n('F_MAP_linked_study_drugs'):,} administrations, "
    f"{p('F_MAP_linked_study_drugs'):,} patients (four study drugs)",
    fs=8.5)
arr(.46, .76, .46, .72)

box(.46, .685,
    "Exposure class assigned from the MAR action string,\n"
    "dose unit and route (rule fixed before analysis)",
    fs=8.5)

# --- four mutually-exclusive sibling classes ---
ybar = .635
ax.plot([.11, .88], [ybar, ybar], color="#555", lw=1.3, transform=ax.transAxes)
arr(.46, .648, .46, ybar)
xs = [.14, .38, .62, .86]
for x_ in xs:
    ax.annotate("", xy=(x_, .585), xytext=(x_, ybar), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3))

classes = [
    (xs[0], f"Intravenous bolus\nn = {n('H_bolus_all_study_drugs'):,}\n"
            f"({p('H_bolus_all_study_drugs'):,} patients)\nfour study drugs",
     "#DCEBD8", "#2E7D32"),
    (xs[1], f"Infusion start /\nbag change\nn = {n('G_class_infusion_start_or_bag_change'):,}\n"
            "(not separable:\nno order ID)", "#F2F2F2", "#999"),
    (xs[2], f"Infusion resume\nn = {n('G_class_infusion_resume'):,}\n"
            "(descriptive only)", "#F2F2F2", "#999"),
    (xs[3], f"Non-IV route\nn = {n('G_class_non_iv'):,}\n(excluded)", "#F2F2F2", "#999"),
]
for x_, t, fc, ec in classes:
    ax.text(x_, .520, t, ha="center", va="center", fontsize=7.2, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=.35", fc=fc, ec=ec, lw=1.1))

ax.text(.46, .425,
        "Mutually exclusive classes sum to the MAP-linked study-drug set "
        f"({n('H_bolus_all_study_drugs'):,} + {n('G_class_infusion_start_or_bag_change'):,} + "
        f"{n('G_class_infusion_resume'):,} + {n('G_class_non_iv'):,} = {n('F_MAP_linked_study_drugs'):,}).",
        ha="center", va="center", fontsize=7, style="italic",
        transform=ax.transAxes, color="#444")

# --- primary cohort split into analysis arms (single continuous elbow) ---
# vertical drop from the bolus box, a horizontal distributor that SPANS both arm x's,
# and a downward arrow into each arm. The distributor runs from the primary-arm x to
# the exploratory-arm x and the drop lands on it, so there is no gap.
x_arm1, x_arm2 = .26, .66
y_drop_top, y_bar2 = .478, .355
arr(xs[0], y_drop_top, xs[0], y_bar2 + .003)                      # bolus -> distributor
ax.plot([x_arm1, x_arm2], [y_bar2, y_bar2], color="#555", lw=1.3, transform=ax.transAxes)
ax.plot([xs[0], x_arm1], [y_bar2, y_bar2], color="#555", lw=1.3, transform=ax.transAxes)
for x_ in (x_arm1, x_arm2):
    ax.annotate("", xy=(x_, .305), xytext=(x_, y_bar2), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3))
box(x_arm1, .255,
    f"Three primary drugs\nn = {n('I_bolus_three_primary'):,} "
    f"({p('I_bolus_three_primary'):,} patients)\n"
    "phenylephrine 29,114\nephedrine 11,585\nvasopressin 6,352\nPRIMARY ANALYSIS COHORT",
    fc="#DCEBD8", ec="#2E7D32", fs=7.6)
box(x_arm2, .255,
    f"Norepinephrine (exploratory)\nn = {n('J_bolus_norepinephrine_exploratory'):,} "
    f"({p('J_bolus_norepinephrine_exploratory'):,} patients)\n"
    "bolus class is 38% of its\nadministrations; too small\nfor confirmatory inference",
    fc="#FFF4E0", ec="#B26A00", fs=7.6)

ax.set_title("Documented-exposure cohort (MOVER)", fontsize=11)
fig.savefig(FIG / "fig1_flow.png", dpi=300, bbox_inches="tight")
print("wrote", FIG / "fig1_flow.png")
