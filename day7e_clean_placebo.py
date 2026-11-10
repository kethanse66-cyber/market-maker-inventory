"""Day 7e: the CLEAN PLACEBO - coins that crossed a power of ten where the tick does NOT change.
Free (public API). ~10 minutes.

Run:  python E:\\p3\\day7e_clean_placebo.py > E:\\p3\\day7e_output.txt
Saves: E:\\p3\\data\\clean_placebo_scan.csv   and   E:\\p3\\data\\clean_placebo_coin_days.csv

Why: SOL $100 and XRP $1.00 are round numbers AND tick boundaries. Our 8 placebo levels ($95, $1.30...)
are not as round, so a pure round-number effect could survive them. The clean placebo is a power of ten
where the decimals cap (6 - szDecimals) already binds on both sides, so the price is just as round but
the tick stays the same (e.g. a coin with szDecimals 2 crossing $1.00, or szDecimals 3 crossing $10).
If makers change behaviour there too, it is the round number, not the tick.
Prices only - no orders, nothing about results. Same scan as Day 7b with the tick condition reversed.
"""
import math
import os
import sys
import time

import pandas as pd
import requests

API = "https://api.hyperliquid.xyz/info"
OUT = r"E:\p3\data\clean_placebo_scan.csv"
OUT_DAYS = r"E:\p3\data\clean_placebo_coin_days.csv"
MIN_USD = 250_000                               # same liquidity floor as the replication coins
START = pd.Timestamp("2026-03-10", tz="UTC")
END = pd.Timestamp("2026-10-26", tz="UTC")
HAVE = {("SOL", 100.0), ("XRP", 1.0)}            # already downloaded (Jun 1-14, Aug 15-Oct 25)


def post(body, tries=5):
    for k in range(tries):
        try:
            r = requests.post(API, json=body, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (k + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))
    return None


def tick(p, szd):
    return max(10.0 ** (math.floor(math.log10(p)) - 4), 10.0 ** -(6 - szd))


def candles(coin, interval, t0, t1):
    out, cur = [], t0
    while cur < t1:
        rows = post({"type": "candleSnapshot",
                     "req": {"coin": coin, "interval": interval, "startTime": cur, "endTime": t1}}) or []
        if not rows:
            break
        out += rows
        last = rows[-1]["t"]
        if last <= cur:
            break
        cur = last + 1
        time.sleep(0.2)
    return out


meta = post({"type": "meta"})
coins = [(u["name"], u["szDecimals"]) for u in meta["universe"] if not u.get("isDelisted")]
print(f"\nPerps listed: {len(coins)}", flush=True)
t0, t1 = int(START.value // 1_000_000), int(END.value // 1_000_000)
rows, day_rows = [], []
for i, (coin, szd) in enumerate(coins):
    try:
        c = candles(coin, "1h", t0, t1)
        first = min((x["t"] for x in c), default=t1)
        if first > t0 + 3_600_000:                       # hourly not served that far back: fill with 4h
            c = candles(coin, "4h", t0, first) + c
    except Exception as e:
        print(f"   {coin}: failed ({type(e).__name__})", file=sys.stderr, flush=True)
        continue
    if not c:
        continue
    df = pd.DataFrame(c)
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["T"] = pd.to_datetime(df["T"], unit="ms", utc=True)
    for col in ("h", "l", "c", "v"):
        df[col] = pd.to_numeric(df[col])
    df = df.drop_duplicates("t").sort_values("t")
    lo, hi = df["l"].min(), df["h"].max()
    if lo <= 0:
        continue
    for k in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        b = 10.0 ** k
        if not (lo < b <= hi):
            continue
        t_below, t_above = tick(b * 0.999, szd), tick(b, szd)
        if abs(t_above / t_below - 1) > 1e-9:
            continue                                    # tick changes here: not a clean placebo
        cross = (df["l"] < b) & (df["h"] >= b)
        if not cross.any():
            continue
        run = (cross != cross.shift()).cumsum()
        eps = df[cross].groupby(run[cross])
        ep_days = sorted({g["t"].iloc[0].strftime("%Y-%m-%d") for _, g in eps})
        near = df[(df["c"] > b * 0.98) & (df["c"] < b * 1.02)]
        usd_per_hour = float((near["v"] * near["c"]).median()) if len(near) else 0.0
        day_rows += [(coin, b, d) for d in ep_days]
        rows.append(dict(coin=coin, szDecimals=szd, boundary=b, tick_below=t_below, tick_above=t_above,
                         episodes=eps.ngroups, days=len(ep_days), crossing_hours=int(cross.sum()),
                         hours_within_2pct=len(near), median_usd_per_hour_near=round(usd_per_hour),
                         first_day=ep_days[0], last_day=ep_days[-1],
                         already_have=(coin, b) in HAVE, resolution="1h" if df["t"].diff().median()
                         <= pd.Timedelta(hours=1) else "mixed 1h/4h"))
    if (i + 1) % 25 == 0:
        print(f"   ... {i + 1}/{len(coins)} coins scanned", file=sys.stderr, flush=True)

if not rows:
    print("\nNo coin crossed a power of ten with an unchanged tick between Mar 10 and Oct 25.")
    print("-> a clean round-number placebo does not exist in this window; say so in the limitations.")
    sys.exit(0)
scan = pd.DataFrame(rows).sort_values(["days", "episodes"], ascending=False)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
scan.to_csv(OUT, index=False)
liquid = scan[scan["median_usd_per_hour_near"] >= MIN_USD]
print(f"\nPowers of ten crossed with NO tick change (Mar 10 - Oct 25): {len(scan)} coin-boundaries")
print(f"   with >= ${MIN_USD:,} traded per hour near the level: {len(liquid)} "
      f"({int(liquid['episodes'].sum())} episodes on {int(liquid['days'].sum())} coin-days)")
print(scan[["coin", "szDecimals", "boundary", "tick_below", "tick_above", "episodes", "days", "crossing_hours",
            "median_usd_per_hour_near", "first_day", "last_day"]].head(30).to_string(index=False))
# coin-days to request: crossing day +-1, liquid ones only
keep = set(zip(liquid["coin"], liquid["boundary"]))
req = set()
for coin, b, d in day_rows:
    if (coin, b) in keep:
        for k in (-1, 0, 1):
            x = pd.Timestamp(d) + pd.Timedelta(days=k)
            if START.tz_localize(None) <= x < END.tz_localize(None):
                req.add((coin, x.strftime("%Y-%m-%d")))
pd.DataFrame(sorted(req), columns=["coin", "day"]).to_csv(OUT_DAYS, index=False)
print(f"\nSaved {OUT} and {OUT_DAYS} ({len(req)} coin-days to add to the export request)")
