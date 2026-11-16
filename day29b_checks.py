"""Checks on the resting-order rebuild: old orders and partial fills."""
import glob
import os
import sys
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

EXPORT = [p for p in os.environ.get("P3_EXPORT", "E:/hl_export;E:/hl_export/delivery-2").split(";") if p]
BOOK = os.environ.get("P3_BOOK", r"E:\hl_data\book")
PART1 = os.environ.get("P3_PART1", r"E:\p3\part1")
OUT = os.environ.get("P3_RESULTS", r"E:\p3\results")
COINS = os.environ.get("P3_COINS", "SOL,XRP,BTC,ETH,HYPE").split(",")
DAYS = list(pd.date_range("2026-06-01", "2026-06-14").strftime("%Y-%m-%d")) + \
       list(pd.date_range("2026-08-15", "2026-10-25").strftime("%Y-%m-%d"))
DAYS = os.environ.get("P3_DAYS", ",".join(DAYS[3::7])).split(",")        # every 7th day, needs 3 days of history
BAND_BP, MIN_QUOTES, NS = 5.0, 1000, 1_000_000_000
con = duckdb.connect()
con.execute("SET TimeZone = 'UTC'")


def log(s):
    print(s, file=sys.stderr, flush=True)


def back(d, k):
    return (pd.Timestamp(d) - pd.Timedelta(days=k)).strftime("%Y-%m-%d")


keep = {}
for c in COINS:
    cnt = defaultdict(int)
    for f in glob.glob(os.path.join(PART1, c, "*.parquet")):
        if pq.ParquetFile(f).metadata.num_rows == 0:
            continue
        for w, n in pd.read_parquet(f, columns=["wallet"])["wallet"].value_counts().items():
            cnt[w] += int(n)
    keep[c] = sorted(w for w, n in cnt.items() if n >= MIN_QUOTES)


def mid_at(coin, d, t):
    f = os.path.join(BOOK, coin, f"{coin}_{d}.parquet")
    if not os.path.exists(f):
        return None
    b = pd.read_parquet(f)
    tc = next(c for c in ("time", "timestamp", "ts", "local_timestamp") if c in b.columns)
    bt = (pd.to_datetime(b[tc], utc=True, format="ISO8601") - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(nanoseconds=1)
    bid, ask = b["bid_px"].astype(float), b["ask_px"].astype(float)
    ok = (ask > bid) & (bid > 0) & np.isfinite(bid) & np.isfinite(ask) & (bt >= t)
    if not ok.any():
        return None
    i = bt[ok].idxmin()
    return float((bid[i] + ask[i]) / 2)


old_in, all_in, n_days = 0, 0, 0
canc, part, filled_frac = 0, 0, []
for c in COINS:
    con.register("wk", pd.DataFrame({"w": keep[c]}))
    for d in DAYS:
        of = [p for r in EXPORT for p in (f"{r}/orders/{c}/{x}.parquet" for x in (back(d, 3), back(d, 2), back(d, 1)))
              if os.path.exists(p)]
        D0 = pd.Timestamp(d, tz="UTC").value
        m = mid_at(c, d, D0)
        if len(of) < 3 or m is None:
            log(f"   {c} {d}: needs 3 previous days of orders and a book - skipped")
            continue
        con.execute(f"""CREATE OR REPLACE TEMP TABLE ev AS
            SELECT CAST(oid AS BIGINT) AS oid, CAST(status AS VARCHAR) AS status, CAST(side AS VARCHAR) AS side,
                   CAST(limit_price AS DOUBLE) AS price, CAST(orig_size AS DOUBLE) AS osz, CAST(size AS DOUBLE) AS sz,
                   epoch_ns(timestamp) AS ts, CAST(tif AS VARCHAR) AS tif, CAST(order_type AS VARCHAR) AS otype,
                   COALESCE(CAST(reduce_only AS BOOLEAN), FALSE) AS ro
            FROM read_parquet({of}, union_by_name=true)
            WHERE lower(user_address) IN (SELECT w FROM wk) AND epoch_ns(timestamp) < {D0}""")
        # 1) orders alive at midnight, by when they were opened
        r = con.execute(f"""
            WITH op AS (SELECT oid, arg_min(price, ts) AS price, min(ts) AS t_open FROM ev
                        WHERE status = 'open' AND tif = 'Alo' AND otype = 'Limit' AND NOT ro GROUP BY oid),
                 en AS (SELECT e.oid, min(e.ts) AS t_end FROM ev e JOIN op o ON e.oid = o.oid
                        WHERE e.status <> 'open' AND e.ts >= o.t_open GROUP BY e.oid)
            SELECT sum(CASE WHEN o.t_open < {D0 - 86400 * NS} THEN 1 ELSE 0 END), count(*)
            FROM op o LEFT JOIN en ON o.oid = en.oid
            WHERE en.t_end IS NULL AND abs(o.price - {m}) / {m} * 1e4 <= {BAND_BP}""").fetchone()
        old_in += int(r[0] or 0); all_in += int(r[1] or 0); n_days += 1
        # 2) partial fills seen on cancellation
        r2 = con.execute("""
            WITH q AS (SELECT DISTINCT oid FROM ev WHERE status = 'open' AND tif = 'Alo' AND otype = 'Limit' AND NOT ro)
            SELECT e.osz, e.sz FROM ev e JOIN q ON e.oid = q.oid
            WHERE e.status = 'canceled' AND e.osz > 0 AND e.sz IS NOT NULL""").df()
        canc += len(r2)
        p = r2[r2["sz"] < r2["osz"] * (1 - 1e-9)]
        part += len(p)
        filled_frac.append(((p["osz"] - p["sz"]) / p["osz"]).to_numpy())
        log(f"   {c} {d}: done")

print("CHECKS ON THE RESTING-QUOTE RECONSTRUCTION")
print(f"1) OLD ORDERS ({n_days} coin-days, 72 h of history):")
if all_in:
    print(f"   resting in-band orders at midnight: {all_in:,}; opened more than 24 h earlier: {old_in:,} "
          f"({old_in / all_in:.3%})  <- the orders day29 cannot see")
print("2) PARTIAL FILLS (cancelled post-only quotes of the T1 makers):")
if canc:
    ff = np.concatenate(filled_frac) if filled_frac else np.array([])
    print(f"   cancelled quotes: {canc:,}; partly filled before cancel: {part:,} ({part / canc:.3%})")
    if len(ff):
        print(f"   among those, share of the original size already filled: median {np.median(ff):.1%}, mean {ff.mean():.1%}")
pd.DataFrame([dict(coin_days=n_days, inband_at_midnight=all_in, older_than_24h=old_in, cancelled=canc,
                   partly_filled=part)]).to_csv(os.path.join(OUT, "part1_resting_checks.csv"), index=False)
