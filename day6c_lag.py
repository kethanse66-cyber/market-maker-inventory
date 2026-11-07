"""Day 6c: measure the book's clock delay for EVERY HOUR of every day. About 15-30 minutes.

Run:  python E:\\p3\\day6c_lag.py > E:\\p3\\day6c_output.txt
Saves: E:\\p3\\data\\book_lag.csv   (coin, day, hour, lag_s, crossed_at_best, quotes, source)

Method (checked on fake data with a planted delay, which it recovered exactly):
  a post-only (Alo) quote can never cross the real book. For each hour we try delays from -0.5 s to
  +3.0 s and keep the one where the fewest quotes look crossed against the nearest book photo.
  Quiet hours with too few quotes use that day's delay; days with too few use 0.25 s (decision 6).
Looks only at quote prices vs the book - nothing about inventory or the study's results.
"""
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd

EXPORT = ["E:/hl_export", "E:/hl_export/delivery-2"]
BOOK = r"E:\hl_data\book"
OUT = r"E:\p3\data\book_lag.csv"
COINS = [c.upper() for c in sys.argv[1:]] or ["SOL", "XRP"]     # e.g. python day6c_lag.py BTC ETH HYPE
DAYS = list(pd.date_range("2026-06-01", "2026-06-14").strftime("%Y-%m-%d")) + \
       list(pd.date_range("2026-08-15", "2026-10-25").strftime("%Y-%m-%d"))
BOOK_GAP_DAYS = {"2026-06-07", "2026-06-08", "2026-06-09", "2026-06-14"}
GRID = np.round(np.arange(-0.5, 3.0001, 0.125), 3)
PER_HOUR = 20_000                                # quotes sampled per hour
MIN_HOUR = 2_000                                 # fewer than this -> use the day's delay
DEFAULT = 0.25
NS = 1_000_000_000


def log(s):
    print(s, file=sys.stderr, flush=True)


def pick(cols, options, what):
    for o in options:
        if o in cols:
            return o
    sys.exit(f"STOP: cannot find the {what} column in the book file. Columns are: {cols}")


con = duckdb.connect()
con.execute("SET TimeZone = 'UTC'")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out = []
t0 = time.time()
for coin in COINS:
    for d in DAYS:
        of = [p for p in (f"{r}/orders/{coin}/{d}.parquet" for r in EXPORT) if os.path.exists(p)]
        bf = os.path.join(BOOK, coin, f"{coin}_{d}.parquet")
        if not of or not os.path.exists(bf) or d in BOOK_GAP_DAYS:
            continue
        q = con.execute(f"""
            WITH x AS (SELECT side, CAST(limit_price AS DOUBLE) AS price, epoch_ns(timestamp) AS ts,
                              CAST((epoch_ns(timestamp) // 3600000000000) % 24 AS INTEGER) AS hr,
                              row_number() OVER (PARTITION BY (epoch_ns(timestamp) // 3600000000000) % 24
                                                 ORDER BY hash(timestamp, limit_price, side)) AS rn
                       FROM read_parquet({of}, union_by_name=true)
                       WHERE status = 'open' AND CAST(tif AS VARCHAR) = 'Alo')
            SELECT side, price, ts, hr FROM x WHERE rn <= {PER_HOUR} ORDER BY ts""").df()
        b = pd.read_parquet(bf)
        bc = list(b.columns)
        c_t = pick(bc, ["timestamp", "ts", "time", "local_timestamp"], "time")
        c_b = pick(bc, ["bid_px", "best_bid", "bid_price", "bid", "bid_price_1", "bid_px_0", "bid_0_px"], "best bid")
        c_a = pick(bc, ["ask_px", "best_ask", "ask_price", "ask", "ask_price_1", "ask_px_0", "ask_0_px"], "best ask")
        bt = pd.to_datetime(b[c_t], utc=True, format="ISO8601").dt.as_unit("ns").astype("int64")
        base = pd.DataFrame({"bt": bt, "bid": pd.to_numeric(b[c_b]), "ask": pd.to_numeric(b[c_a])}).dropna()
        base = base[base["ask"] > base["bid"]].drop_duplicates("bt", keep="last").sort_values("bt")
        is_bid = (q["side"].astype(str) == "B").to_numpy()
        crossed = np.zeros((len(GRID), 24))
        usable = np.zeros((len(GRID), 24))
        for i, s in enumerate(GRID):
            bb = pd.DataFrame({"bts": base["bt"].to_numpy() - int(s * NS),
                               "bid": base["bid"].to_numpy(), "ask": base["ask"].to_numpy()})
            m = pd.merge_asof(q, bb, left_on="ts", right_on="bts", direction="nearest")
            ok = (m["bid"].notna() & ((m["ts"] - m["bts"]).abs() <= NS)).to_numpy()
            cr = np.where(is_bid, m["price"].to_numpy() >= m["ask"].to_numpy(),
                          m["price"].to_numpy() <= m["bid"].to_numpy()) & ok
            crossed[i] = np.bincount(q["hr"], weights=cr, minlength=24)
            usable[i] = np.bincount(q["hr"], weights=ok, minlength=24)
        rate = crossed / np.maximum(usable, 1)
        day_rate = crossed.sum(axis=1) / np.maximum(usable.sum(axis=1), 1)
        day_n = usable[list(GRID).index(DEFAULT)].sum()
        day_lag = float(GRID[int(np.argmin(day_rate))]) if day_n >= MIN_HOUR else DEFAULT
        for h in range(24):
            n_h = usable[list(GRID).index(DEFAULT), h]
            if n_h >= MIN_HOUR:
                k = int(np.argmin(rate[:, h]))
                out.append((coin, d, h, float(GRID[k]), float(rate[k, h]), int(n_h), "hour"))
            else:
                k = list(GRID).index(day_lag)
                out.append((coin, d, h, day_lag, float(rate[k, h]) if n_h else np.nan, int(n_h),
                            "day" if day_n >= MIN_HOUR else "default"))
        log(f"   {coin} {d}: day delay {day_lag:+.3f} s | crossed at that delay {day_rate.min():.2%} "
            f"| {time.time() - t0:,.0f}s")

lag = pd.DataFrame(out, columns=["coin", "day", "hour", "lag_s", "crossed_at_best", "quotes", "source"])
if os.path.exists(OUT):                          # keep the other coins already measured
    old = pd.read_csv(OUT)
    lag = pd.concat([old[~old["coin"].isin(COINS)], lag], ignore_index=True)
lag.to_csv(OUT, index=False)
lag = lag[lag["coin"].isin(COINS)]

print("\nBOOK CLOCK DELAY, MEASURED HOUR BY HOUR")
for coin, g in lag.groupby("coin"):
    gh = g[g["source"] == "hour"]
    print(f"\n   {coin}: {len(g):,} hours | measured directly {len(gh):,} | from the day {int((g['source'] == 'day').sum())} "
          f"| default 0.25 s {int((g['source'] == 'default').sum())}")
    print(f"      delay: median {gh['lag_s'].median():+.3f} s | middle 50% {gh['lag_s'].quantile(.25):+.3f} to "
          f"{gh['lag_s'].quantile(.75):+.3f} s | at 1 s or more: {(gh['lag_s'] >= 1).mean():.1%} of hours")
    print(f"      quotes still crossed at the best delay: median {gh['crossed_at_best'].median():.2%} | "
          f"hours above 2%: {(gh['crossed_at_best'] > 0.02).mean():.1%} (poor book data, flagged)")
    by_month = gh.assign(month=gh["day"].str[:7]).groupby("month")["lag_s"].median()
    print("      median delay by month: " + " | ".join(f"{m} {v:+.2f} s" for m, v in by_month.items()))
    worst = gh.sort_values("crossed_at_best", ascending=False).head(5)
    print("      worst hours: " + ", ".join(f"{r.day} {r.hour:02d}h ({r.crossed_at_best:.1%} at {r.lag_s:+.2f} s)"
                                          for r in worst.itertuples()))
print(f"\nSaved {OUT}  ({time.time() - t0:,.0f} seconds)")
