#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Consistency audit of the manuscript against the results index.

Checks explicit (drug, quantity, comparison, sign) assertions against results_index.csv
and fails loudly on any mismatch; verifies that the reported difference-in-differences
equals the difference of the two history effects (within rounding); then scans the
manuscript body for numerals not traceable to the index (reported, not fatal).

This is an internal-consistency check of the reported values, not an independent
re-extraction of the source data. Exit 0 only if there are zero failures.
"""
import re
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent


def _norm(v):
    if v is None:
        return ''
    s = str(v)
    return '' if s.lower() == 'nan' else s


def load_registry(path):
    df = pd.read_csv(path)
    idx = {}
    for _, r in df.iterrows():
        key = (_norm(r['drug']), _norm(r['estimand']), _norm(r.get('features', '')),
               _norm(r.get('model', '')), _norm(r.get('comparator', '')))
        idx[key] = r
    return df, idx


def val(idx, drug, estimand, features='', model='', comparator='', col='estimate'):
    key = (drug, estimand, features, model, comparator)
    if key not in idx:
        return None
    return idx[key][col]


# (label, drug, estimand, features, model, comparator, expected_sign)
# sign: '-' CI below 0, '+' CI above 0 / value exists positive, '0' CI spans 0.
CHECKS = [
    ("history_effect_on_xgb lowers MAE (phe)", "phenylephrine", "dMAE", "xgb", "", "history_effect_on_xgb", "-"),
    ("history_effect_on_xgb lowers MAE (eph)", "ephedrine", "dMAE", "xgb", "", "history_effect_on_xgb", "-"),
    ("history_effect_on_xgb lowers MAE (vas)", "vasopressin", "dMAE", "xgb", "", "history_effect_on_xgb", "-"),
    ("algorithm_effect_base (phe) favours XGB", "phenylephrine", "dMAE", "xgb_vs_ridge_base", "", "algorithm_effect_base", "-"),
    ("algorithm_effect_base (eph) favours XGB", "ephedrine", "dMAE", "xgb_vs_ridge_base", "", "algorithm_effect_base", "-"),
    ("algorithm_effect_base (vas) spans zero", "vasopressin", "dMAE", "xgb_vs_ridge_base", "", "algorithm_effect_base", "0"),
    ("best_vs_baseline (phe)", "phenylephrine", "dMAE", "xgb_hist", "", "best_candidate_vs_baseline", "-"),
    ("best_vs_baseline (eph)", "ephedrine", "dMAE", "xgb_hist", "", "best_candidate_vs_baseline", "-"),
    ("best_vs_baseline (vas)", "vasopressin", "dMAE", "xgb_hist", "", "best_candidate_vs_baseline", "-"),
    ("did_history spans zero (phe)", "phenylephrine", "did_history", "xgb_vs_ridge_hist", "", "history_did", "0"),
    ("did_history spans zero (eph)", "ephedrine", "did_history", "xgb_vs_ridge_hist", "", "history_did", "0"),
    ("did_history spans zero (vas)", "vasopressin", "did_history", "xgb_vs_ridge_hist", "", "history_did", "0"),
    ("whole AUROC exists (phe)", "phenylephrine", "auroc", "whole", "clf_xgb", "", "+"),
    ("whole AUROC exists (eph)", "ephedrine", "auroc", "whole", "clf_xgb", "", "+"),
    ("whole AUROC exists (vas)", "vasopressin", "auroc", "whole", "clf_xgb", "", "+"),
    ("below65 AUROC exists (phe)", "phenylephrine", "auroc", "below65", "clf_xgb", "", "+"),
    ("below65 AUROC exists (eph)", "ephedrine", "auroc", "below65", "clf_xgb", "", "+"),
    ("below65 AUROC exists (vas)", "vasopressin", "auroc", "below65", "clf_xgb", "", "+"),
]


def check_signs(idx):
    fails = []
    for label, drug, estimand, feat, model, comp, sign in CHECKS:
        est = val(idx, drug, estimand, feat, model, comp, 'estimate')
        lo = val(idx, drug, estimand, feat, model, comp, 'ci_lo')
        hi = val(idx, drug, estimand, feat, model, comp, 'ci_hi')
        if est is None:
            fails.append(f"MISSING registry row: {label} [{drug}/{estimand}/{feat}/{model}/{comp}]")
            continue
        est = float(est)
        try:
            lo = float(lo); hi = float(hi)
        except (TypeError, ValueError):
            lo = hi = None
        if sign == '-' and not (est < 0 and (hi is None or hi < 0)):
            fails.append(f"SIGN: {label}: expected CI below 0, got est={est} CI=({lo},{hi})")
        elif sign == '+' and not (est > 0 and (lo is None or lo > 0)):
            fails.append(f"SIGN: {label}: expected positive, got est={est} CI=({lo},{hi})")
        elif sign == '0' and not (lo is not None and lo < 0 < hi):
            fails.append(f"SIGN: {label}: expected CI to span 0, got est={est} CI=({lo},{hi})")
    return fails


def check_did_consistency(did_path, tol=5e-4):
    """Acceptance criterion: DiD == (history effect under XGB) - (history effect under
    Ridge) within rounding, for every primary drug. Both history effects are computed
    under the same per-repeat convention and stored in did.csv."""
    fails = []
    if not Path(did_path).exists():
        return [f"MISSING {did_path} for DiD-consistency check"]
    d = pd.read_csv(did_path)
    for _, r in d.iterrows():
        if r['role'] != 'primary':
            continue
        implied = float(r['hist_xgb']) - float(r['hist_ridge'])
        if abs(implied - float(r['did'])) > tol:
            fails.append(f"DiD-CONSISTENCY ({r['drug']}): did={r['did']} but "
                         f"hist_xgb-hist_ridge={implied:.4f}")
    return fails


def _isnum(v):
    try:
        float(v); return True
    except (TypeError, ValueError):
        return False


def orphan_numerals(tex_path, registry_df):
    text = Path(tex_path).read_text()
    body = re.sub(r'%.*', '', text)
    toks = set(re.findall(r'(?<![\w.])\d+\.\d+|(?<![\w.])\d{1,3}(?:,\d{3})+|(?<![\w.])\d+', body))
    known = set()
    for col in ['estimate', 'ci_lo', 'ci_hi', 'n', 'n_patients']:
        if col in registry_df.columns:
            for v in registry_df[col].dropna():
                known.add(str(v))
                if _isnum(v):
                    known.add(f"{abs(float(v)):.3f}".rstrip('0').rstrip('.'))
                    known.add(f"{abs(float(v)):.2f}")
                    known.add(f"{abs(float(v)):.1f}")
    WHITELIST = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '15', '30', '65', '75',
                 '2024', '2026', '0.5', '0.6', '0.7', '0.8', '0.9', '0.05', '95', '45',
                 '21', '39', '40', '60', '20', '16', '0.80', '0.78',
                 '5331', '3822', '1461', '6451', '47051', '47808', '53324',
                 '29114', '11585', '6352', '757', '218', '10219', '4282', '2623',
                 '11359', '7536', '4430', '2873', '2206', '1523'}

    def strip(x):
        return x.replace(',', '')
    known_stripped = {strip(k) for k in known | WHITELIST}
    return sorted(t for t in toks if strip(t) not in known_stripped)


FORBIDDEN = [
    ("loss-vs-class overclaim",
     "the choice of loss mattered more than the choice of hypothesis class"),
    ("blanket-unchanged claim", "Every conclusion is unchanged in direction and magnitude"),
    ("blanket-unchanged claim (abstract)", "The findings were unchanged after excluding"),
    ("loss conflation", "fitted to the same loss"),
    ("comparison typo", "cheaper than both Ridge variants"),
    ("stale cross-reference", "Supplementary Tables~S1 and S8"),
    ("unqualified best-model designation", "The primary candidate model is XGBoost with treatment history."),
]
REQUIRED = [
    ("XGBoost objective stated", "reg:squarederror"),
    ("qualitative-preserved wording", "qualitative conclusions were preserved"),
    ("preselected-model designation", "preselected XGBoost reference model"),
    ("corrected cross-reference", "Supplementary Tables~S1 and S7"),
]


def check_text(tex_path):
    fails = []
    text = Path(tex_path).read_text() if Path(tex_path).exists() else ""
    if not text:
        return [f"manuscript not found at {tex_path}"]
    for label, phrase in FORBIDDEN:
        if phrase in text:
            fails.append(f"FORBIDDEN [{label}]: found {phrase!r}")
    for label, phrase in REQUIRED:
        if phrase not in text:
            fails.append(f"REQUIRED MISSING [{label}]: {phrase!r}")
    return fails

def main():
    import argparse
    ap = argparse.ArgumentParser()
    res = HERE.parent / "results"
    ap.add_argument("--registry", default=str(res / "results_index.csv"))
    ap.add_argument("--did", default=str(res / "did.csv"))
    ap.add_argument("--tex", default=str(
        HERE.parent / "manuscript" / "manuscript.tex"))
    args = ap.parse_args()
    reg_path, tex_path = Path(args.registry), Path(args.tex)
    if not reg_path.exists():
        print(f"FATAL: registry not found: {reg_path}"); sys.exit(2)
    df, idx = load_registry(reg_path)
    fails = check_signs(idx)
    did_fails = check_did_consistency(args.did)
    print("=" * 60)
    print("SEMANTIC SIGN/EXISTENCE CHECKS")
    print("=" * 60)
    if fails:
        for f in fails:
            print("  FAIL:", f)
    else:
        print(f"  all {len(CHECKS)} load-bearing claims consistent with registry")
    print("\nDiD-CONSISTENCY CHECK (did == hist_xgb - hist_ridge):")
    if did_fails:
        for f in did_fails:
            print("  FAIL:", f)
    else:
        print("  DiD equals the difference of same-convention history effects (all primary)")
    if tex_path.exists():
        orphans = orphan_numerals(tex_path, df)
        print("\nORPHAN NUMERALS (for manual check, not fatal):")
        print("  " + (", ".join(orphans) if orphans else "(none)"))
    else:
        print(f"\n(tex not found at {tex_path}; skipped orphan scan)")
    text_fails = check_text(str(tex_path)) if tex_path.exists() else []
    print("\nTEXT-CONSISTENCY CHECKS (forbidden/required phrasings):")
    if text_fails:
        for f in text_fails:
            print("  FAIL:", f)
    else:
        print(f"  all {len(FORBIDDEN)+len(REQUIRED)} phrasing checks pass")
    total = fails + did_fails + text_fails
    if total:
        print(f"\nAUDIT FAILED: {len(total)} consistency error(s)"); sys.exit(1)
    print("\nAUDIT PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
