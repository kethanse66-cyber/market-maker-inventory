"""Day 28: CHECK EVERY KEY NUMBER IN THE PAPER AGAINST THE RESULT FILES.

Run:  python E:\\p3\\day28_verify_paper.py
Prints each number as written in the paper next to the value in the result file. OK = equal after rounding to the
digits the paper shows. Anything else is printed as CHECK.
"""
import os
import sys

import pandas as pd

RES = os.environ.get("P3_RESULTS", r"E:\p3\results")


def load(name):
    f = os.path.join(RES, name)
    if not os.path.exists(f):
        return None
    return pd.read_csv(f)


cache = {}


def get(name, col, **flt):
    if name not in cache:
        cache[name] = load(name)
    d = cache[name]
    if d is None:
        return None, f"{name} missing"
    for k, v in flt.items():
        if k not in d.columns:
            return None, f"column {k} not in {name}"
        if callable(v):
            d = d[d[k].astype(str).map(v)]
        else:
            d = d[d[k].astype(str) == str(v)]
    if d.empty:
        return None, f"no row {flt} in {name}"
    return float(d[col].iat[0]), ""


CHECKS = [
    # (where in the paper, paper value, decimals, file, column, filters)
    ("T1 side response",                 -0.0620, 4, "part1_results.csv", "value", dict(test="T1 pooled", estimate="side")),
    ("T1 price response, ticks",         -0.2445, 4, "part1_results.csv", "value", dict(test="T1 pooled", estimate="price_ticks")),
    ("T1 size response",                 +0.1466, 4, "part1_results.csv", "value", dict(test="T1 pooled", estimate="size")),
    ("T1(c) makers significant",           192,   0, "part1_results.csv", "makers", dict(test="T1 verdict")),
    ("T1(c) side-sig, price < 1 tick",     134,   0, "part1_results.csv", "quotes", dict(test="T1 verdict")),
    ("T2 price x vol",                   +0.054,  3, "part1_results.csv", "value", dict(test="T2", estimate="price q x vol")),
    ("T3 slope |q| < 1",                 -0.096,  3, "part1_results.csv", "value", dict(test="T3", estimate="slope |q| < 1")),
    ("T4 Spearman",                      +0.476,  3, "part1_checks.csv", "value", dict(test="T4", version=lambda s: s.startswith("finite"))),
    ("T4 makers",                          202,   0, "part1_checks.csv", "makers", dict(test="T4", version=lambda s: s.startswith("finite"))),
    ("No fill 60 s: side",               -0.0855, 4, "part1_checks.csv", "value", dict(test="check", version="no fill 60s", estimate="side")),
    ("Return controls: side",            -0.0608, 4, "part1_checks.csv", "value", dict(test="check", version="return ctrl", estimate="side")),
    ("No fill 60 s + returns: side",     -0.0846, 4, "part1_checks.csv", "value", dict(test="check", version="both (60s+ret)", estimate="side")),
    ("Natives only: side",               -0.0731, 4, "part1_checks.csv", "value", dict(test="robust natives", version="NATIVE", estimate="side")),
    ("Rank q: side",                     -0.0710, 4, "part1_qscale.csv", "value", dict(version=lambda s: s.startswith("rank"), estimate="side")),
    ("Without cap-heavy makers: side",   -0.0698, 4, "part1_qscale.csv", "value", dict(version=lambda s: s.startswith("main q, without"), estimate="side")),
    ("Replication T1 side",              -0.1074, 4, "part1_results_replication.csv", "value", dict(test="T1 pooled", estimate="side")),
    ("Replication T1 price, ticks",      -0.3714, 4, "part1_results_replication.csv", "value", dict(test="T1 pooled", estimate="price_ticks")),
    ("Tick test tau (SOL+XRP)",          +0.0037, 4, "main_result.csv", "tau", dict(estimate=lambda s: s.startswith("SOL+XRP"))),
    ("Tick test randomization p",         0.416,  3, "main_result.csv", "p_randomization", dict(estimate=lambda s: s.startswith("SOL+XRP"))),
    ("Tick test baseline side",          -0.055,  3, "main_result.csv", "b_baseline", dict(estimate=lambda s: s.startswith("SOL+XRP"))),
    ("H5 k_plus",                        +0.165,  3, "part1_asymmetry.csv", "k_plus", dict(outcome="aggr", sample="all coins pooled")),
    ("H5 k_minus",                       +0.381,  3, "part1_asymmetry.csv", "k_minus", dict(outcome="aggr", sample="all coins pooled")),
    ("H5 difference",                    -0.216,  3, "part1_asymmetry.csv", "difference", dict(outcome="aggr", sample="all coins pooled")),
    ("H6 q x T",                         +0.0005, 4, "part1_toxicity.csv", "value", dict(model="side", H="10", term="q x T")),
    ("H6 q at T = 0",                    -0.0564, 4, "part1_toxicity.csv", "value", dict(model="side", H="10", term="q")),
    ("H7a gap per sd, bp",               -0.70,   2, "part1_pressure.csv", "value", dict(test="H7a")),
    ("H7b 5-min reversal, bp",           +0.031,  3, "part1_pressure.csv", "value", dict(test="H7b")),
    ("H7b 60-min reversal, bp",          +0.253,  3, "part1_pressure.csv", "value", dict(test="H7b h=60")),
    ("H7 HL own return, bp",             +0.51,   2, "part1_pressure.csv", "value", dict(test="HL own return")),
    ("Replication H7a, bp",              -1.26,   2, "part1_pressure_replication.csv", "value", dict(test="H7a")),
    ("Replication H7b, bp",              +0.089,  3, "part1_pressure_replication.csv", "value", dict(test="H7b")),
    ("Replication H7b 60 min, bp",       +0.497,  3, "part1_pressure_replication.csv", "value", dict(test="H7b h=60")),
]

bad = 0
print(f"{'paper number':<36} {'paper':>10} {'file':>12}  result")
for lab, paper, dec, fname, col, flt in CHECKS:
    v, err = get(fname, col, **flt)
    if v is None:
        print(f"{lab:<36} {paper:>10} {'-':>12}  CHECK ({err})")
        bad += 1
        continue
    ok = round(v, dec) == round(paper, dec)
    bad += not ok
    print(f"{lab:<36} {paper:>10} {v:>12.{max(dec, 1) + 1}f}  {'OK' if ok else 'CHECK'}")
print(f"\n{len(CHECKS) - bad} of {len(CHECKS)} numbers match." + ("" if not bad else " Send me the CHECK lines."))
# derived numbers in the text
v60, _ = get("part1_pressure.csv", "value", test="H7b h=60")
va, _ = get("part1_pressure.csv", "value", test="H7a")
if v60 is not None and va:
    print(f"Derived: share of the gap closed in 60 min = {v60 / abs(va):.0%} (paper: 36%)")
v60r, _ = get("part1_pressure_replication.csv", "value", test="H7b h=60")
var, _ = get("part1_pressure_replication.csv", "value", test="H7a")
if v60r is not None and var:
    print(f"Derived: replication share closed in 60 min = {v60r / abs(var):.0%} (paper: 39%)")
