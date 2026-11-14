"""Day 19: REPLICATION TEST on the 13 delivery-3 coins (pre-registration v1 section 8, addendum v2 section 4).

Run (after day18 finished):  python E:\\p3\\day19_replication_test.py      (about 1-3 hours)
Output: E:\\p3\\results\\day19_replication.txt  and  E:\\p3\\results\\placebo_result_replication.csv

For each coin: crossing episodes at its boundary (a real 10x tick change) and at two NON-round control levels,
0.95x and 1.05x of the boundary. Each control lies entirely below or entirely above the boundary, so the
tick is the same on both sides of it; they are not powers of ten. Then the frozen four-outcome test (side, price,
size, at-best) at every level, 500 circular shifts each. A coin-boundary with < 3 episodes is descriptive only.
Summary: precision-weighted average of the boundary estimates over coins (each coin's estimate weighted by
1 / sd^2 of its own shifts), and the same for the control levels - addendum v2 predicts price at the
boundaries like SOL/XRP/PENGU, and ~0 at the controls.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

P3 = r"E:\p3"
RES = os.path.join(P3, "results")
SKIP = {"CHIP", "XRP", "PENGU", "STRK", "FARTCOIN"}
rq = pd.read_csv(os.path.join(P3, "data", "request_coin_days.csv"))
low = {c.lower(): c for c in rq.columns}
B = rq.groupby(rq[low["coin"]].str.upper())[low["boundary"]].first().astype(float)
coins = [c for c in B.index if c not in SKIP and os.path.exists(os.path.join(RES, f"day18_DONE_{c}.txt"))]
missing = [c for c in B.index if c not in SKIP and c not in coins]
print(f"coins built: {len(coins)} | not built yet (skipped): {missing}")


def lv(x):
    return float(f"{x:.6g}")


# 1. episodes at boundary + controls (re-run of the frozen day16d for each coin)
for c in coins:
    b = B[c]
    arg = f"{c}={lv(b)},{lv(0.95 * b)},{lv(1.05 * b)}"
    with open(os.path.join(RES, f"day19_episodes_{c}.txt"), "w", encoding="utf-8") as out:
        r = subprocess.run([sys.executable, os.path.join(P3, "day16d_episodes_d3.py"), arg], stdout=out,
                           stderr=subprocess.STDOUT)
    print(f"{c}: episodes {'OK' if r.returncode == 0 else 'FAILED'} ({arg})", flush=True)
    if r.returncode != 0:
        sys.exit(f"STOP: building episodes failed for {c} - see E:\\p3\\results\\day19_episodes_{c}.txt")

# 2. the frozen four-outcome test at every level
env = dict(os.environ, P3_EPISODES=os.path.join(P3, "data", "episodes_d3.csv"), P3_ALL_DAYS="1",
           P3_ONLY="NONE", P3_SUFFIX="_replication",
           P3_EXTRA_REAL=" ".join(f"{c}:{lv(B[c])}" for c in coins),
           P3_EXTRA=" ".join(f"{c}:{lv(0.95 * B[c])} {c}:{lv(1.05 * B[c])}" for c in coins))
env["P3_ONLY"] = ""                    # keep every extra run, drop the SOL/XRP levels below
env["P3_LEVELS"] = " ".join(sorted({str(lv(x)) for c in coins for x in (B[c], 0.95 * B[c], 1.05 * B[c])}))
env["P3_COINS_ONLY"] = " ".join(coins)
out_txt = os.path.join(RES, "day19_replication.txt")
res_csv = os.path.join(RES, "placebo_result_replication.csv")
if os.path.exists(res_csv):
    os.remove(res_csv)                 # never summarise an old file by mistake
with open(out_txt, "w", encoding="utf-8") as out:
    r = subprocess.run([sys.executable, os.path.join(P3, "day15_placebos.py")], stdout=out, env=env)
if r.returncode != 0 or not os.path.exists(res_csv):
    sys.exit(f"STOP: the test run failed - see {out_txt}")

# 3. summary across coins
R = pd.read_csv(os.path.join(RES, "placebo_result_replication.csv"))
if "outcome" not in R:
    R = pd.DataFrame(columns=["kind", "episodes", "draw_sd", "level", "outcome", "estimate", "p_randomization"])
R = R[R["outcome"].notna()]
lines = ["\nREPLICATION SUMMARY (precision-weighted over coins; coins with >= 3 episodes)",
         "   DESCRIPTIVE: the z treats coins as independent, which they are not fully (shared market days);",
         "   the per-coin randomization p-values are the tests."]
for kind, lab in (("REAL", "boundaries (tick changes 10x)"), ("placebo", "non-round controls (0.95x, 1.05x)")):
    g = R[(R["kind"] == kind) & (R["episodes"] >= 3) & (R["draw_sd"] > 0)]
    lines.append(f"\n   {lab}: {g['level'].nunique()} levels")
    for o, h in g.groupby("outcome"):
        w = 1 / h["draw_sd"] ** 2
        est, se = float((w * h["estimate"]).sum() / w.sum()), float(1 / np.sqrt(w.sum()))
        pos = int((h["estimate"] > 0).sum())
        lines.append(f"      {o:<13} weighted {est:+.4f} (se {se:.4f}, z {est / se:+.2f}) | positive at {pos} of {len(h)}"
                     f" | p < 0.05 at {int((h['p_randomization'] < 0.05).sum())}")
txt = "\n".join(lines)
print(txt)
with open(out_txt, "a", encoding="utf-8") as f:
    f.write(txt + "\n")
