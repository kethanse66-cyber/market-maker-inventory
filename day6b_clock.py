"""Day 6b: an independent check of the book clock shift (decision 6). About 5-10 minutes.

Run:  python E:\\p3\\day6b_clock.py > E:\\p3\\day6b_output.txt

Idea: a post-only (Alo) quote can never cross the real book - the exchange rejects a buy at or above
the best ask, and a sell at or below the best bid. So if our book looks crossed against a quote, our
book is out of date at that moment. The right clock shift is the one where FEWEST quotes look crossed.
We try shifts from -1 s to +1 s (book time minus shift = true time) on 9 sample days.
Looks only at quote prices vs the book - nothing about inventory or the study's results.
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

EXPORT = ["E:/hl_export", "E:/hl_export/delivery-2"]
BOOK = r"E:\hl_data\book"
DAYS = {"SOL": ["2026-06-03", "2026-08-22", "2026-09-10", "2026-09-21", "2026-10-15"],
        "XRP": ["2026-06-03", "2026-08-22", "2026-09-15", "2026-10-08"]}
SHIFTS = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
MAX_QUOTES = 2_000_000                         # per day, random sample to keep it quick
NS = 1_000_000_000


def pick(cols, options, what):
    for o in options:
        if o in cols:
            return o
    sys.exit(f"STOP: cannot find the {what} column in the book file. Columns are: {cols}")


con = duckdb.connect()
con.execute("SET TimeZone = 'UTC'")
rows = []
print("\nSHARE OF POST-ONLY QUOTES THAT LOOK CROSSED AGAINST OUR BOOK (lower = book closer to the truth)")
print("   coin  day          " + "".join(f"{s:>+8.2f}s" for s in SHIFTS) + "   best")
for coin, days in DAYS.items():
    for d in days:
        of = [p for p in (f"{r}/orders/{coin}/{d}.parquet" for r in EXPORT) if os.path.exists(p)]
        bf = os.path.join(BOOK, coin, f"{coin}_{d}.parquet")
        if not of or not os.path.exists(bf):
            print(f"   {coin}   {d}   (missing files, skipped)")
            continue
        q = con.execute(f"""
            SELECT side, CAST(limit_price AS DOUBLE) AS price, epoch_ns(timestamp) AS ts
            FROM read_parquet({of}, union_by_name=true)
            WHERE status = 'open' AND CAST(tif AS VARCHAR) = 'Alo'""").df()
        if len(q) > MAX_QUOTES:
            q = q.sample(MAX_QUOTES, random_state=0)
        q = q.sort_values("ts")
        b = pd.read_parquet(bf)
        bc = list(b.columns)
        c_t = pick(bc, ["timestamp", "ts", "time", "local_timestamp"], "time")
        c_b = pick(bc, ["bid_px", "best_bid", "bid_price", "bid", "bid_price_1", "bid_px_0", "bid_0_px"], "best bid")
        c_a = pick(bc, ["ask_px", "best_ask", "ask_price", "ask", "ask_price_1", "ask_px_0", "ask_0_px"], "best ask")
        bt = pd.to_datetime(b[c_t], utc=True, format="ISO8601").dt.as_unit("ns").astype("int64")
        base = pd.DataFrame({"bt": bt, "bid": pd.to_numeric(b[c_b]), "ask": pd.to_numeric(b[c_a])}).dropna()
        base = base[base["ask"] > base["bid"]].drop_duplicates("bt", keep="last").sort_values("bt")
        is_bid = q["side"].astype(str) == "B"
        rates = []
        for s in SHIFTS:
            bb = base.assign(bts=base["bt"] - int(s * NS))[["bts", "bid", "ask"]]
            m = pd.merge_asof(q, bb, left_on="ts", right_on="bts", direction="nearest")
            ok = m["bid"].notna() & ((m["ts"] - m["bts"]).abs() <= NS)
            crossed = np.where(is_bid, m["price"] >= m["ask"], m["price"] <= m["bid"]) & ok
            rates.append(crossed.sum() / max(ok.sum(), 1))
        best = SHIFTS[int(np.argmin(rates))]
        rows.append((coin, d, *rates))
        print(f"   {coin}   {d}  " + "".join(f"{r:>9.3%}" for r in rates) + f"   {best:+.2f}s")

if rows:
    avg = np.mean([r[2:] for r in rows], axis=0)
    print("   ALL (average)    " + "".join(f"{r:>9.3%}" for r in avg) +
          f"   {SHIFTS[int(np.argmin(avg))]:+.2f}s")
print("\nReading it: decision 6 uses +0.25 s. If the lowest crossing rate is at +0.25 s (or very close),")
print("the shift is confirmed by a second, independent method. The remaining crossed quotes are moments")
print("when the price moved between two 0.5 s book photos - they get flagged, not fixed.")
