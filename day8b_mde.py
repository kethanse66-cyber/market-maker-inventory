"""Day 8b: read the power check in bp (not as a share of b, which is ~0 in June) + June tick regime.

Run:  python E:\\p3\\day8b_mde.py
"""
import glob

import duckdb
import numpy as np
import pandas as pd

res = pd.read_csv(r"E:\p3\data\power_check.csv")
c_ri = np.quantile(np.abs(res["tau_hat"]), 0.95)
c_cal = np.quantile(np.abs(res["tau_hat"] / res["tau_se"]), 0.95)


def mde(rule):
    for tau in np.linspace(0, 0.05, 50001):
        th = res["tau_hat"] + tau
        p = (np.abs(th) > c_ri).mean() if rule == "ri" else (np.abs(th / res["tau_se"]) > c_cal).mean()
        if p >= 0.80:
            return tau
    return np.inf


print("SMALLEST TAU DETECTED 80% OF THE TIME (bp of price per 1 std of inventory)")
print(f"   randomization-style: {mde('ri'):.4f} bp | calibrated t: {mde('cal'):.4f} bp")
print("   for scale: one FINE tick just below the boundary = 0.1 bp (SOL 0.001 at $100, XRP 0.00001 at $1)")

con = duckdb.connect()
print("\nJUNE TICK REGIME (main-sample quotes)")
for coin, lv in (("SOL", 100.0), ("XRP", 1.0)):
    f = sorted(glob.glob(rf"E:\p3\data\quotes\{coin}\2026-06-*.parquet"))
    r = con.execute(f"""SELECT median(mid), min(mid), max(mid), median((best_ask - best_bid) / tick),
                               avg(CASE WHEN (best_ask - best_bid) / tick < 1.5 THEN 1 ELSE 0 END),
                               stddev_samp(rel_mid_bp)
                        FROM read_parquet({f}, union_by_name=true)
                        WHERE near_best AND NOT in_gap AND NOT crosses_book AND mid IS NOT NULL""").fetchone()
    print(f"   {coin}: mid median {r[0]:.4f} (range {r[1]:.4f}-{r[2]:.4f}; boundary {lv}) | spread median "
          f"{r[3]:.1f} ticks, 1-tick spread {r[4]:.0%} of quotes | std of shading {r[5]:.3f} bp")
