"""Day 4b: find silent stretches (holes) in the order data. About 5-10 minutes.

Run:  python E:\\p3\\day4b_gaps.py > E:\\p3\\day4b_output.txt

Makers send orders many times a second, so the order stream should never go quiet for long.
For every day we list each silence longer than 60 seconds (SOL and XRP side by side).
A silence in BOTH coins at the same time = a hole in the data, not a quiet market.
"""
import os

import duckdb
import pandas as pd

ROOTS = ["E:/hl_export", "E:/hl_export/delivery-2"]
DAYS = list(pd.date_range("2026-05-31", "2026-06-05").strftime("%Y-%m-%d")) + \
       list(pd.date_range("2026-08-14", "2026-10-25").strftime("%Y-%m-%d"))
con = duckdb.connect()
con.execute("SET TimeZone = 'UTC'")

print("\nSILENCES LONGER THAN 60 SECONDS IN THE ORDER STREAM (all wallets together)")
print("   coin  from (UTC)            to (UTC)              length")
for day in DAYS:
    for coin in ("SOL", "XRP"):
        files = [p for p in (f"{r}/orders/{coin}/{day}.parquet" for r in ROOTS) if os.path.exists(p)]
        if not files:
            continue
        rows = con.execute(f"""
            WITH b AS (SELECT DISTINCT date_trunc('second', timestamp) AS t
                       FROM read_parquet({files}, union_by_name=true)),
                 g AS (SELECT LAG(t) OVER (ORDER BY t) AS t0, t AS t1 FROM b)
            SELECT strftime(t0, '%Y-%m-%d %H:%M:%S'), strftime(t1, '%Y-%m-%d %H:%M:%S'),
                   date_diff('second', t0, t1)
            FROM g WHERE date_diff('second', t0, t1) > 60 ORDER BY t0""").fetchall()
        for t0, t1, s in rows:
            print(f"   {coin}   {t0}   {t1}   {s // 60} min {s % 60} s")
print("\nKnown outages (expected here): 2026-06-05 22:00:10-22:24:16 and 2026-10-25 09:32:20-10:28:57")
