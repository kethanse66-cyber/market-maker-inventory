"""Recalibrate the hedger p-values on the placebo and update data/natives.csv."""
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
