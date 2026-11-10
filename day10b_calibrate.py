"""Day 10b: re-label hedgers with p-values CALIBRATED on the placebo (no new data, ~10 seconds).

Run:  python E:\\p3\\day10b_calibrate.py > E:\\p3\\day10b_output.txt
Reads/updates: E:\\p3\\data\\natives.csv  (adds p_cal_*, sig_cal_*, label_cal; keeps the old columns)

Why: in Day 10 the placebo (flow paired with Binance from OTHER days, so no real link exists) gave
p < 5% far more often than 5% for some coins (SOL 1 s 19%, HYPE 10 s 19%), so the plain Granger p-values
are too small there and some HEDGER labels are false alarms.
Fix (a permutation test): for each coin and speed, a unit's calibrated p = the share of placebo tests
that came out at least as extreme as its real test:  p_cal = (1 + #placebo p <= p) / (1 + #placebo).
By construction p_cal is right-sized whatever went wrong in the plain test.
Two labels, same rules otherwise (decisions 18 and 19):
  label_cal      PRIMARY - each test at the 5% level (decision 18 as written), p_cal <= 0.05. About 5%
                 of accounts per test can be false hedgers; the two speeds overlap, so less than 10% in all.
  label_cal_fdr  STRICTER - Benjamini-Hochberg FDR at 5% within coin x level x speed. With a few hundred
                 placebo tests the smallest possible p_cal is ~0.002, so FDR over 100-250 accounts can
                 rarely pass: it is shown as the conservative bound, not used for the split.
The placebo tests come from accounts; sleeve and firm units use the same coin x speed placebo set.
"""
import os

import numpy as np
import pandas as pd

F = os.environ.get("P3_NATIVES", r"E:\p3\data\natives.csv")
ALPHA, HL_MAX = 0.05, 60.0


def bh(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.zeros(len(p), bool)
    if ok.sum():
        q = p[ok]
        o = np.argsort(q)
        passed = q[o] <= ALPHA * np.arange(1, len(q) + 1) / len(q)
        k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
        s = np.zeros(len(q), bool)
        s[o[:k]] = True
        out[ok] = s
    return out


res = pd.read_csv(F)
for sp in ("1s", "10s"):
    res[f"p_cal_{sp}"] = np.nan
    pc = [c for c in res.columns if c.startswith(f"placebo_p_{sp}_")]
    for coin, idx in res.groupby("coin").groups.items():
        pl = res.loc[(res["coin"] == coin) & (res["level"] == "account"), pc].to_numpy().ravel()
        pl = np.sort(pl[np.isfinite(pl)])
        if not len(pl):
            continue
        p = res.loc[idx, f"p_{sp}"].to_numpy()
        res.loc[idx, f"p_cal_{sp}"] = np.where(np.isfinite(p),
                                                (1 + np.searchsorted(pl, p, side="right")) / (1 + len(pl)), np.nan)
    res[f"sig_cal_{sp}"] = res[f"p_cal_{sp}"] <= ALPHA
    res[f"sig_fdr_{sp}"] = False
    for _, idx in res.groupby(["coin", "level"]).groups.items():
        res.loc[idx, f"sig_fdr_{sp}"] = bh(res.loc[idx, f"p_cal_{sp}"].to_numpy())

both = res["p_cal_1s"].notna() & res["p_cal_10s"].notna()
hedger = res["sig_cal_1s"] | res["sig_cal_10s"]
fast = res["half_life_min"] < HL_MAX
inc = ~both | res["half_life_min"].isna()
res["label_cal"] = np.select([hedger, inc, fast], ["HEDGER", "INCOMPLETE", "NATIVE"], "SLOW")
res["label_cal_fdr"] = np.select([res["sig_fdr_1s"] | res["sig_fdr_10s"], inc, fast],
                                 ["HEDGER", "INCOMPLETE", "NATIVE"], "SLOW")
res.to_csv(F, index=False)

print("HEDGER LABELS WITH PLACEBO-CALIBRATED p-VALUES (old -> new), accounts")
print("   coin   placebo p<5% (1s/10s)    NATIVE         HEDGER         SLOW           INCOMPLETE")
for coin, d in res[res["level"] == "account"].groupby("coin", sort=False):
    pr = []
    for sp in ("1s", "10s"):
        pc = [c for c in d.columns if c.startswith(f"placebo_p_{sp}_")]
        v = d[pc].to_numpy().ravel()
        v = v[np.isfinite(v)]
        pr.append(f"{(v < ALPHA).mean():.0%}")
    cells = []
    for lab in ("NATIVE", "HEDGER", "SLOW", "INCOMPLETE"):
        a, b = int((d["label"] == lab).sum()), int((d["label_cal"] == lab).sum())
        sh = d.loc[d["label_cal"] == lab, "fills"].sum() / d["fills"].sum()
        cells.append(f"{a:>3} -> {b:<3} ({sh:.0%})")
    print(f"   {coin:<5}  {pr[0]:>4} / {pr[1]:<4}            " + "  ".join(cells))
print("\n   (share in brackets = share of the coin's fills under the new label)")
print("\n   stricter FDR version, accounts: " + " | ".join(
    f"{c}: native {int((g['label_cal_fdr'] == 'NATIVE').sum())}, hedger {int((g['label_cal_fdr'] == 'HEDGER').sum())}"
    for c, g in res[res["level"] == "account"].groupby("coin", sort=False)))
print("   real tests calibrated p <= 5% vs the 5% expected by chance, accounts: " + " | ".join(
    f"{c} 1s {(g['p_cal_1s'] <= ALPHA).mean():.0%}, 10s {(g['p_cal_10s'] <= ALPHA).mean():.0%}"
    for c, g in res[res["level"] == "account"].groupby("coin", sort=False)))
for lvl in ("sleeve", "firm"):
    d = res[res["level"] == lvl]
    print(f"   {lvl}: " + " | ".join(f"{c}: native {int((g['label_cal'] == 'NATIVE').sum())}, hedger "
                                     f"{int((g['label_cal'] == 'HEDGER').sum())}" for c, g in d.groupby("coin", sort=False)))
print(f"\nUpdated {F} (label_cal is the label used from now on; the old label is kept for comparison)")
