"""Day 16d (delivery-3 coins): crossing episodes at a coin's boundary - the frozen Day 7 code.
Run:  python E:\\p3\\day16d_episodes_d3.py PENGU=0.01 > E:\\p3\\results\\day16d_PENGU.txt
Saves E:\\p3\\data\\episodes_d3.csv (other coins already there are kept; the frozen episodes.csv is not touched).

Original Day 7 description:

Run:  python E:\\p3\\day7_episodes.py > E:\\p3\\day7_output.txt

An episode = the minute mid price crossing a level (SOL $100, XRP $1.00, and the 8 placebo levels from
decision 3); crossings closer together than MERGE minutes are one episode (decision 1 merges overlapping
episodes). For each episode we look at the hours its +-30 minute window touches, and call it
  GOOD     if none of those hours is a poor-book hour (Day 6c: > 2% of quotes crossed at the best delay)
  POOR     otherwise.
Counts only - nothing about quotes, inventory or the study's results is looked at.
"""
import os
import sys

import numpy as np
import pandas as pd

BOOK = r"E:\hl_data\book"
LAG = r"E:\p3\data\book_lag_d3.csv"
OUT = r"E:\p3\data\episodes_d3.csv"
LEVELS = {a.split("=")[0].upper(): [float(x) for x in a.split("=")[1].split(",")] for a in sys.argv[1:]}
if not LEVELS:
    sys.exit("Give coin=level, e.g.  python day16d_episodes_d3.py PENGU=0.01")
DAYS = list(pd.date_range("2026-06-01", "2026-06-14").strftime("%Y-%m-%d")) + \
       list(pd.date_range("2026-08-15", "2026-10-25").strftime("%Y-%m-%d"))
WINDOW_MIN = 30
D3 = os.environ.get("P3_D3", r"G:\hl_export\delivery-3")


def days_for(coin):
    """Days with vendor order files for this coin in delivery 3 (sorted)."""
    import glob as _g
    return sorted(os.path.basename(f)[:10] for f in _g.glob(os.path.join(D3, "orders", coin, "*.parquet")))


POOR = 0.02


def pick(cols, options):
    return next(o for o in options if o in cols)


lag = pd.read_csv(LAG)
poor_hours = set((r.coin, r.day, int(r.hour)) for r in lag.itertuples()
                 if pd.notna(r.crossed_at_best) and r.crossed_at_best > POOR)
measured = set((r.coin, r.day, int(r.hour)) for r in lag.itertuples() if pd.notna(r.crossed_at_best))
rows = []
coverage = {}
for coin, levels in LEVELS.items():
    mids, missing, rejected, rows_total = [], [], 0, 0
    for d in days_for(coin):
        p = os.path.join(BOOK, coin, f"{coin}_{d}.parquet")
        if not os.path.exists(p):
            missing.append(d)
            continue
        b = pd.read_parquet(p)
        c_t = pick(b.columns, ["timestamp", "ts", "time", "local_timestamp"])
        c_b = pick(b.columns, ["bid_px", "best_bid", "bid_price", "bid", "bid_price_1", "bid_px_0", "bid_0_px"])
        c_a = pick(b.columns, ["ask_px", "best_ask", "ask_price", "ask", "ask_price_1", "ask_px_0", "ask_0_px"])
        t = pd.to_datetime(b[c_t], utc=True, format="ISO8601")
        bid = pd.to_numeric(b[c_b], errors="coerce")
        ask = pd.to_numeric(b[c_a], errors="coerce")
        ok = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > bid)     # a bad row cannot make a crossing
        rows_total += len(b)
        rejected += int((~ok).sum())
        m = ((bid[ok] + ask[ok]) / 2).groupby(t[ok].dt.floor("min")).last()
        mids.append(m)
    mid = pd.concat(mids).sort_index()
    n_dup = int(mid.index.duplicated().sum())
    mid = mid[~mid.index.duplicated(keep="last")]
    gaps = mid.index.to_series().diff()
    coverage[coin] = (missing, rejected, rows_total, n_dup, int((gaps > pd.Timedelta(minutes=10)).sum()))
    for lv in levels:
        side = np.sign(mid - lv)
        side = side[side != 0]
        change = side.index[1:][(side.to_numpy()[1:] != side.to_numpy()[:-1])]
        # minutes in a row with a gap > 10 min (missing data) do not count as a crossing
        prev = side.index[:-1][(side.to_numpy()[1:] != side.to_numpy()[:-1])]
        change = change[(change - prev) <= pd.Timedelta(minutes=10)]
        for merge in (30, 60):
            eps = []
            for c in change:
                if eps and (c - eps[-1][1]) <= pd.Timedelta(minutes=merge):
                    eps[-1][1] = c
                else:
                    eps.append([c, c])
            for a, z in eps:
                w0, w1 = a - pd.Timedelta(minutes=WINDOW_MIN), z + pd.Timedelta(minutes=WINDOW_MIN)
                hrs = pd.date_range(w0.floor("h"), w1.floor("h"), freq="h")
                keys = [(coin, h.strftime("%Y-%m-%d"), h.hour) for h in hrs]
                bad = sum(k in poor_hours for k in keys)
                unmeasured = sum(k not in measured for k in keys)
                quality = "POOR" if bad else ("UNKNOWN" if unmeasured else "GOOD")
                rows.append(dict(coin=coin, level=lv, merge_min=merge, start=a, end=z,
                                 day=a.strftime("%Y-%m-%d"), hours=len(hrs), poor_hours=bad,
                                 unmeasured_hours=unmeasured, quality=quality, good=quality == "GOOD"))

ep = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
if os.path.exists(OUT):                          # keep other coins already there
    old = pd.read_csv(OUT)
    ep = pd.concat([old[~old["coin"].isin(LEVELS)], ep], ignore_index=True)
ep.to_csv(OUT, index=False)
ep = ep[ep["coin"].isin(LEVELS)]
print("\nBOOK COVERAGE")
for coin, (missing, rej, tot_rows, dup, big_gaps) in coverage.items():
    print(f"   {coin}: missing book days {len(missing)} {missing} | bad rows rejected {rej:,} of {tot_rows:,} | "
          f"duplicate minutes {dup} | gaps > 10 min {big_gaps}")
print("\nCROSSING EPISODES BY BOOK QUALITY (window = episode +- 30 min)")
print("   (decision 1 counted SOL 100: 70 episodes, XRP 1.00: 59 - our counts should be close)")
print("   merge 30 min and 60 min are two settings: with 60 min, any two windows that overlap are one episode\n")
print("   coin level   merge | episodes  days |  GOOD  POOR  UNKNOWN | GOOD days")
for (coin, lv, mg), g in ep.groupby(["coin", "level", "merge_min"]):
    tag = "  <- MAIN" if lv in (100.0, 1.0) else ""
    print(f"   {coin}  {lv:>6g}  {mg:>3} min | {len(g):>8} {g['day'].nunique():>5} | {int((g['quality'] == 'GOOD').sum()):>5} "
          f"{int((g['quality'] == 'POOR').sum()):>5} {int((g['quality'] == 'UNKNOWN').sum()):>8} | "
          f"{g.loc[g['good'], 'day'].nunique():>9}{tag}")
main = ep[ep["merge_min"] == 60]
print("\n   MAIN episodes by day (merge 60 min): day: total / good")
for (coin, d), g in main.groupby(["coin", "day"]):
    print(f"      {coin} {d}: {len(g):>3} / {int(g['good'].sum()):>3}")
print(f"\nSaved {OUT}")
print("Reading it: if most MAIN episodes are GOOD, the Hyperliquid book is fine as the primary price.")
print("If many are POOR, the main test uses Binance prices in those hours (the fuzzy design in the plan).")
