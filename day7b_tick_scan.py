"""Find coins that crossed a 10x tick boundary, and when. Writes data/tick_scan.csv."""
import math
import os
import sys
import time

import pandas as pd
import requests

API = "https://api.hyperliquid.xyz/info"
OUT = r"E:\p3\data\tick_scan.csv"
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
rows = []
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
        if abs(t_above / t_below - 1) < 1e-9:
            continue                                    # the decimals cap binds: no tick change here
        cross = (df["l"] < b) & (df["h"] >= b)
        if not cross.any():
            continue
        run = (cross != cross.shift()).cumsum()
        eps = df[cross].groupby(run[cross])
        ep_days = sorted({g["t"].iloc[0].strftime("%Y-%m-%d") for _, g in eps})
        near = df[(df["c"] > b * 0.98) & (df["c"] < b * 1.02)]
        usd_per_hour = float((near["v"] * near["c"]).median()) if len(near) else 0.0
        rows.append(dict(coin=coin, boundary=b, tick_below=t_below, tick_above=t_above,
                         episodes=eps.ngroups, days=len(ep_days), crossing_hours=int(cross.sum()),
                         hours_within_2pct=len(near), median_usd_per_hour_near=round(usd_per_hour),
                         first_day=ep_days[0], last_day=ep_days[-1],
                         already_have=(coin, b) in HAVE, resolution="1h" if df["t"].diff().median()
                         <= pd.Timedelta(hours=1) else "mixed 1h/4h"))
    if (i + 1) % 25 == 0:
        print(f"   ... {i + 1}/{len(coins)} coins scanned", file=sys.stderr, flush=True)

scan = pd.DataFrame(rows).sort_values(["days", "episodes"], ascending=False)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
scan.to_csv(OUT, index=False)
liquid = scan[scan["median_usd_per_hour_near"] >= 1_000_000]
print(f"\nCoin-boundaries with a 10x tick change that was crossed (Mar 10 - Oct 25): {len(scan)}")
print(f"   of those with >= $1M traded per hour near the boundary (enough market makers): {len(liquid)}")
print(f"   episodes in the liquid ones: {int(liquid['episodes'].sum())} on {int(liquid['days'].sum())} coin-days")
print("\nTOP 30 by number of crossing days (liquid first):")
show = pd.concat([liquid, scan.drop(liquid.index)]).head(30)
print(show[["coin", "boundary", "tick_below", "tick_above", "episodes", "days", "crossing_hours",
            "median_usd_per_hour_near", "first_day", "last_day", "already_have"]].to_string(index=False))
print(f"\nSaved {OUT}")
print("Next: we pick the liquid coin-boundaries with the most days and download only those coin-months.")
