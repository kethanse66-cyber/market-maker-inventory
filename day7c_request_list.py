"""Coin-days to request from 0xArchive (crossing days +-1). Writes data/request_coin_days.csv."""
import os
import time

import pandas as pd
import requests

API = "https://api.hyperliquid.xyz/info"
SCAN = r"E:\p3\data\tick_scan.csv"
OUT_DAYS = r"E:\p3\data\request_coin_days.csv"
OUT_WALLETS = r"E:\p3\data\request_wallets.csv"
FIRMS = r"E:\hl_data\firm_wallets_all.csv"
MIN_USD_PER_HOUR = 250_000
HAVE_DAYS = set(pd.date_range("2026-06-01", "2026-06-14").strftime("%Y-%m-%d")) | \
            set(pd.date_range("2026-08-15", "2026-10-25").strftime("%Y-%m-%d"))
HAVE_COINS = {"SOL", "XRP", "BTC", "ETH", "HYPE"}
START, END = "2026-03-10", "2026-10-26"


def candles(coin, interval, t0, t1):
    out, cur = [], t0
    while cur < t1:
        r = requests.post(API, json={"type": "candleSnapshot", "req": {
            "coin": coin, "interval": interval, "startTime": cur, "endTime": t1}}, timeout=30)
        r.raise_for_status()
        rows = r.json()
        if not rows or rows[-1]["t"] <= cur:
            out += rows or []
            break
        out += rows
        cur = rows[-1]["t"] + 1
        time.sleep(0.2)
    return out


scan = pd.read_csv(SCAN)
pick = scan[scan["median_usd_per_hour_near"] >= MIN_USD_PER_HOUR].copy()
print(f"\nCoin-boundaries at >= ${MIN_USD_PER_HOUR:,} per hour near the boundary: {len(pick)}")
t0 = int(pd.Timestamp(START, tz="UTC").value // 1_000_000)
t1 = int(pd.Timestamp(END, tz="UTC").value // 1_000_000)
need = []
for r in pick.itertuples():
    c = candles(r.coin, "1h", t0, t1)
    first = min((x["t"] for x in c), default=t1)
    if first > t0 + 3_600_000:
        c = candles(r.coin, "4h", t0, first) + c
    df = pd.DataFrame(c)
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    for col in ("h", "l"):
        df[col] = pd.to_numeric(df[col])
    cross_days = sorted(set(df.loc[(df["l"] < r.boundary) & (df["h"] >= r.boundary), "t"].dt.strftime("%Y-%m-%d")))
    days = set()
    for d in cross_days:                                     # each crossing day +- 1 day
        for k in (-1, 0, 1):
            days.add((pd.Timestamp(d) + pd.Timedelta(days=k)).strftime("%Y-%m-%d"))
    days = sorted(x for x in days if START <= x <= "2026-10-25")
    if r.coin in HAVE_COINS:
        days = [x for x in days if x not in HAVE_DAYS]
    for d in days:
        need.append((r.coin, r.boundary, d))
    print(f"   {r.coin:<8} ${r.boundary:<8g} crossing days {len(cross_days):>3} -> request {len(days):>3} days")

req = pd.DataFrame(need, columns=["coin", "boundary", "day"]).drop_duplicates(["coin", "day"])
os.makedirs(os.path.dirname(OUT_DAYS), exist_ok=True)
req.to_csv(OUT_DAYS, index=False)
fw = pd.read_csv(FIRMS)
fw[["wallet"]].drop_duplicates().to_csv(OUT_WALLETS, index=False)
print(f"\nTOTAL: {len(req)} coin-days across {req['coin'].nunique()} coins | firm wallets: {fw['wallet'].nunique():,}")
print(f"Saved {OUT_DAYS} and {OUT_WALLETS} - attach both to the email to 0xArchive.")
