"""Day 27 (Nov 14): FIGURE 6 - makers' inventory and the Hyperliquid-minus-Binance price gap (prereg-v7 results).

Run (after day26, both runs):  python -W ignore E:\\p3\\day27_fig_pressure.py
Reads   E:\\p3\\results\\part1_pressure_minutes.parquet, part1_pressure.csv (+ _replication if present)
Saves   E:\\p3\\figures\\fig6_price_pressure.png / .pdf

Panel A: the gap B (bp, minus its coin-day mean) by 0.5-sd bins of maker inventory z (also minus its coin-day mean, as in the regression); minutes pooled within a coin,
         every coin weighted equally; 95% band from resampling days; dashed line = the H7a regression slope.
Panel B: how much of the gap has closed h minutes later: b(h) / |a| for h = 1, 5, 15, 60 (H7b at each horizon
         divided by the H7a level), 5 main coins and the replication coins.
"""
import os
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("STOP: matplotlib missing - run:  pip install matplotlib")

RES = os.environ.get("P3_RESULTS", r"E:\p3\results")
FIG = os.environ.get("P3_FIGURES", r"E:\p3\figures")
os.makedirs(FIG, exist_ok=True)
mf = os.path.join(RES, "part1_pressure_minutes.parquet")
cf = os.path.join(RES, "part1_pressure.csv")
for f in (mf, cf):
    if not os.path.exists(f):
        sys.exit(f"STOP: {f} missing - run day26_price_pressure.py first")

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C1, C2 = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False})

# ---- panel A data
A = pd.read_parquet(mf, columns=["coin", "day", "z", "B"]).dropna()
A["Bd"] = A["B"] - A.groupby(["coin", "day"])["B"].transform("mean")
edges = np.arange(-3, 3.01, 0.5)
A["zd"] = A["z"] - A.groupby(["coin", "day"])["z"].transform("mean")    # x within coin-day too, as the regression
A["bin"] = np.clip(np.searchsorted(edges, A["zd"], side="right") - 1, 0, len(edges) - 2)
# minutes pooled (as in the regression), every coin equal; 95% band by resampling days (500 draws, seed fixed)
S = A.groupby(["day", "coin", "bin"])["Bd"].agg(["sum", "count"]).reset_index()
mid = (edges[:-1] + edges[1:]) / 2
nb = len(mid)


def curve(Sx):
    cb = Sx.groupby(["coin", "bin"])[["sum", "count"]].sum()
    cm = (cb["sum"] / cb["count"]).where(cb["count"] >= 200)            # a coin-bin needs 200+ minutes
    return cm.groupby("bin").mean().reindex(range(nb))                  # coins equal


mean = curve(S)
days = S["day"].unique()
rng = np.random.default_rng(20261114)
idx = {d: g for d, g in S.groupby("day")}
boot = []
for _ in range(500):
    pick = rng.choice(days, len(days), replace=True)
    boot.append(curve(pd.concat([idx[d] for d in pick], ignore_index=True)).to_numpy())
boot = np.array(boot)
lo_b, hi_b = np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)
ok = mean.notna().to_numpy()
r0 = pd.read_csv(cf)
a_main = float(r0.loc[r0["test"] == "H7a", "value"].iat[0]) if (r0["test"] == "H7a").any() else np.nan

# ---- panel B data
def path(csv):
    if not os.path.exists(csv):
        return None
    r = pd.read_csv(csv)
    a = r.loc[r["test"] == "H7a", "value"]
    if a.empty:
        return None
    a = abs(float(a.iat[0]))
    pts = []
    for h, t in ((1, "H7b h=1"), (5, "H7b"), (15, "H7b h=15"), (60, "H7b h=60")):
        s = r[r["test"] == t]
        if len(s):
            pts.append((h, float(s["value"].iat[0]) / a, 1.96 * float(s["se"].iat[0]) / a))
    return pd.DataFrame(pts, columns=["h", "share", "ci"]), a, int(r.loc[r["test"] == "H7a", "coins"].iat[0])


main = path(cf)
rep = path(os.path.join(RES, "part1_pressure_replication.csv"))

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
a0 = ax[0]
a0.fill_between(mid[ok], lo_b[ok], hi_b[ok], color=C1, alpha=0.18, linewidth=0)
a0.plot(mid[ok], mean[ok], color=C1, linewidth=2, marker="o", markersize=5, label="average gap (minutes pooled)")
if np.isfinite(a_main):
    xs = np.array([mid[ok].min(), mid[ok].max()])
    a0.plot(xs, a_main * xs, color=INK2, linewidth=1.2, linestyle="--", label=f"regression slope {a_main:+.2f} bp per sd")
    a0.legend(fontsize=8, loc="lower left")
a0.axhline(0, color=INK2, linewidth=0.8)
a0.axvline(0, color=INK2, linewidth=0.8)
a0.set_title("A. Makers long -> Hyperliquid trades below Binance", loc="left")
a0.set_xlabel("makers' combined inventory z, vs its coin-day average (sd; + = more long)")
a0.set_ylabel("Hyperliquid minus Binance, bp\n(vs the coin-day average)")

a1 = ax[1]
for res, col, lab in ((main, C1, "5 main coins"), (rep, C2, "replication coins")):
    if res is None:
        continue
    d, a, nc = res
    a1.errorbar(d["h"], d["share"] * 100, yerr=d["ci"] * 100, color=col, marker="o", markersize=6,
                linewidth=2, elinewidth=1.2, capsize=0, label=f"{lab} ({nc}; gap {a:.2f} bp per sd)")
    a1.text(d["h"].iat[-1] * 1.08, d["share"].iat[-1] * 100, f"{d['share'].iat[-1]:.0%}", color=INK,
            fontsize=9, va="center")
a1.axhline(0, color=INK2, linewidth=0.8)
a1.set_xscale("log")
a1.set_xticks([1, 5, 15, 60])
a1.set_xticklabels(["1", "5", "15", "60"])
a1.set_xlim(0.8, 90)
a1.set_title("B. The gap closes over the next hour", loc="left")
a1.set_xlabel("minutes ahead (log scale)")
a1.set_ylabel("share of the gap closed, %")
a1.legend(fontsize=8, loc="upper left")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"fig6_price_pressure.{ext}"), dpi=200)
print(f"Saved fig6_price_pressure.png / .pdf in {FIG}")
if main is not None:
    print("   5 main coins, share of the gap closed: " + ", ".join(f"{int(h)} min {s:.0%}" for h, s, _ in main[0].itertuples(index=False)))
if rep is not None:
    print("   replication coins:                     " + ", ".join(f"{int(h)} min {s:.0%}" for h, s, _ in rep[0].itertuples(index=False)))
