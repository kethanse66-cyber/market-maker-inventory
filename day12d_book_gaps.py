"""Day 12d: check the downloaded order-book prices for the 18 new coins: missing files and gaps. ~2 minutes.

Run:  python E:\\p3\\day12d_book_gaps.py > E:\\p3\\day12d_output.txt

Gap = more than 60 s without a snapshot. Every gap goes to E:\\p3\\data\\book_gaps_new.csv
(coin, day, gap_start, gap_end, seconds) so later scripts drop those minutes, like delivery-3 gaps.
Also checks the Binance and day-before fill files exist, and that bid < ask.
"""
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, r"E:\p3")
from day12c_extras import ROOT, prev_days, read_coin_days  # noqa: E402

GAP_S = 60
cd = read_coin_days()
pv = prev_days(cd)
gaps, summ = [], []
for coin in sorted({c for c, _ in cd}):
    days = sorted(d for c, d in cd if c == coin)
    s = {"coin": coin, "days": len(days), "missing": 0, "empty_files": 0, "snapshots": 0, "cadence_s": [],
         "gap_min": 0.0, "days_gap_over_1h": 0, "crossed": 0, "binance_missing": 0}
    for d in days:
        f = os.path.join(ROOT, "book", coin, f"{coin}_{d}.parquet")
        if coin == "CHIP":
            f2 = os.path.join(ROOT, "book_l2cp", "CHIP", f"CHIP_{d}.parquet")
            f = f2 if os.path.exists(f2) else f
        if not os.path.exists(os.path.join(ROOT, "binance", f"{coin}USDT", f"{coin}USDT_{d}.parquet")):
            s["binance_missing"] += 1
        if not os.path.exists(f):
            s["missing"] += 1
            continue
        b = pq.read_table(f, columns=["time", "bid_px", "ask_px"]).to_pandas()
        if not len(b):
            s["empty_files"] += 1
            gaps.append((coin, d, f"{d} 00:00:00+00:00", f"{d} 23:59:59+00:00", 86400.0))
            s["gap_min"] += 1440
            s["days_gap_over_1h"] += 1
            continue
        t = pd.to_datetime(b["time"], utc=True).sort_values().reset_index(drop=True)
        s["snapshots"] += len(t)
        s["crossed"] += int((b["bid_px"] >= b["ask_px"]).sum())
        day0, day1 = pd.Timestamp(d, tz="UTC"), pd.Timestamp(d, tz="UTC") + pd.Timedelta(days=1)
        edges = pd.concat([pd.Series([day0]), t, pd.Series([day1])], ignore_index=True)
        dt = edges.diff().dt.total_seconds()
        s["cadence_s"].append(t.diff().dt.total_seconds().median())
        day_gap = 0.0
        for i in dt.index[dt > GAP_S]:
            gaps.append((coin, d, edges[i - 1], edges[i], dt[i]))
            day_gap += dt[i]
        s["gap_min"] += day_gap / 60
        s["days_gap_over_1h"] += day_gap > 3600
    s["fills_prev_day"] = sum(os.path.exists(os.path.join(ROOT, coin, "trades", f"{coin}_{d}.parquet"))
                              for c, d in pv if c == coin)
    s["runs"] = sum(1 for c, _ in pv if c == coin)
    cad = pd.Series(s.pop("cadence_s"))
    s["cadence_s"] = round(cad.median(), 2) if len(cad) else None
    summ.append(s)

g = pd.DataFrame(gaps, columns=["coin", "day", "gap_start", "gap_end", "seconds"])
g.to_csv(r"E:\p3\data\book_gaps_new.csv", index=False)
S = pd.DataFrame(summ)
S["gap_pct"] = (S["gap_min"] / (S["days"] * 1440) * 100).round(2)
S["gap_min"] = S["gap_min"].round(0)
pd.set_option("display.width", 200)
print("ORDER-BOOK PRICES FOR THE NEW COINS (gap = no snapshot for > 60 s, day edges included)\n")
print(S[["coin", "days", "missing", "empty_files", "snapshots", "cadence_s", "gap_min", "gap_pct", "days_gap_over_1h",
         "crossed", "binance_missing", "runs", "fills_prev_day"]].to_string(index=False))
print(f"\nTOTAL: {S.days.sum()} coin-days | missing files {S.missing.sum()} | empty {S.empty_files.sum()} | "
      f"gap time {S.gap_min.sum() / 60:.1f} h of {S.days.sum() * 24} h ({S.gap_min.sum() / (S.days.sum() * 1440) * 100:.2f}%)")
print(f"day-before fill files: {S.fills_prev_day.sum()} of {S.runs.sum()}")
big = g[g.seconds > 3600].sort_values("seconds", ascending=False)
print(f"\nGaps longer than 1 hour: {len(big)}")
print(big.head(30).to_string(index=False) if len(big) else "   none")
print(f"\nSaved {len(g)} gaps to E:\\p3\\data\\book_gaps_new.csv")
