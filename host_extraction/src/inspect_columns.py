#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inspect_columns.py | MOVER host.

Reads only the header (and a 5-row sample) of the three MOVER source files and
prints the column names, so the operator can fill in the two optional timestamp
columns in config/mover_columns.yaml without guessing.

Reads forward-only. Writes nothing.
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
CFG = PKG / "config" / "mover_columns.yaml"


def show(label, path, sample=True):
    p = Path(path)
    if not p.exists():
        print(f"\n### {label}: FILE NOT FOUND -> {p}")
        return
    try:
        head = pd.read_csv(p, nrows=0)
        print(f"\n### {label}  ({p})\n{len(head.columns)} columns:")
        print(", ".join(head.columns))
        if sample:
            print("\nfirst row values (for type sanity):")
            r = pd.read_csv(p, nrows=1)
            for c in r.columns:
                print(f"   {c:32s} = {str(r.iloc[0][c])[:60]}")
    except Exception as e:
        print(f"\n### {label}: could not read ({e})")


def main():
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CFG
    cfg = yaml.safe_load(open(cfg_path))
    print("=" * 78)
    print("column inspection — copy exact names into config/mover_columns.yaml")
    print("=" * 78)
    show("MAR / medications (med_file)", cfg["med_file"])
    print("\n" + "-" * 78)
    print("LOOK FOR: an administration timestamp (med_time_col), and any entry /")
    print("modification / verify timestamps. Put the latter two into")
    print("  med_entry_time_col:")
    print("  med_mod_time_col:")
    print("If there are no such columns, leave both blank — the timestamp audit")
    print("will be reported as 'column_absent' rather than silently skipped.")
    print("-" * 78)
    show("Observations (arterial MAP)", cfg["obs_file"], sample=False)
    show("Surgery / patient information", cfg["surgery_table"], sample=False)


if __name__ == "__main__":
    main()
