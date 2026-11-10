"""Day 7f: who makes the market in the CLEAN PLACEBO coins (PENGU and the other liquid ones)?
Uses your API credits (trades only). ~15-40 minutes.

Run:  python E:\\p3\\day7f_placebo_makers.py
Reads:  E:\\p3\\data\\clean_placebo_coin_days.csv  (from Day 7e)
Adds to: E:\\p3\\data\\request_wallets.csv  (keeps every wallet already in it, adds each placebo coin's
         top 30 pure makers) and writes E:\\p3\\data\\request_wallets_added.csv (only the new ones).
Same rule as Day 7d: >= 90% maker share, ranked by maker-side dollar volume, up to 6 crossing days per coin.
Safe to stop and re-run: finished days are skipped.
"""
import argparse
import functools
import os
import re
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from oxarchive import Client

DL = r"E:\sentinel\hl_dl3.py"
REQ = r"E:\p3\data\clean_placebo_coin_days.csv"
FIRMS = r"E:\hl_data\firm_wallets_all.csv"
OUT = r"E:\p3\data\request_wallets.csv"
ROOT = r"E:\hl_data"
TOP, MAX_DAYS, PURE = 30, 6, 0.90
print = functools.partial(print, flush=True)

# ---- borrow as_dict / pull / days / save from the downloader, unchanged ----
src = open(DL, encoding="utf-8").read().splitlines()
start = next(i for i, l in enumerate(src) if l.startswith("def as_dict"))
block = []
for l in src[start:]:
    if l and not l[0].isspace() and not l.startswith(("def ", "@", "#")) and block:
        break
    block.append(l)
H = Client(api_key=os.environ["OXARCHIVE_API_KEY"], timeout=120).hyperliquid
a = argparse.Namespace(start=None, end=None)
env = dict(os=os, time=time, pd=pd, datetime=datetime, timedelta=timedelta, timezone=timezone,
           print=print, a=a, H=H, re=re)
exec("\n".join(block), env)
for need in ("pull", "days", "save"):
    if need not in env:
        raise SystemExit(f"STOP: could not find {need}() in {DL}")

req = pd.read_csv(REQ)
firm = pd.read_csv(FIRMS)
firm_w = set(firm["wallet"].str.lower())
before = set(pd.read_csv(OUT)["wallet"].str.lower()) if os.path.exists(OUT) else set()
chosen = set(firm_w) | before
summary = []
for coin, g in req.groupby("coin"):
    base = os.path.join(ROOT, coin)
    tdir = os.path.join(base, "trades")
    # crossing days are the middle of each +-1 block: pick days spread through the list
    days_all = sorted(g["day"])
    pick_days = days_all[1::3][:MAX_DAYS] or days_all[:MAX_DAYS]
    env["COIN"], env["BASE"] = coin, base
    for d in pick_days:
        path = os.path.join(tdir, f"{coin}_{d}.parquet")
        if os.path.exists(path):
            continue
        a.start = a.end = d
        for day, s, e in env["days"]():
            t0 = time.time()
            rows, pages = env["pull"](H.trades.list, symbol=coin, start=s, end=e, limit=1000)
            env["save"](rows, os.path.join(tdir, f"{coin}_{day}.parquet"))
            print(f"   {coin} {day}: {len(rows):,} trades, {pages} pages, {time.time() - t0:,.0f}s")
    files = [os.path.join(tdir, f"{coin}_{d}.parquet") for d in pick_days]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        continue
    tr = pd.concat([pd.read_parquet(f, columns=["user_address", "price", "size", "crossed"]) for f in files])
    tr["user_address"] = tr["user_address"].str.lower()
    tr["usd"] = pd.to_numeric(tr["price"]) * pd.to_numeric(tr["size"])
    tr["maker"] = tr["crossed"].astype(str).str.lower().eq("false")
    share = tr.groupby("user_address")["maker"].mean()
    vol = tr[tr["maker"]].groupby("user_address")["usd"].sum()
    pure = vol[share.reindex(vol.index) >= PURE].sort_values(ascending=False)
    top = list(pure.head(TOP).index)
    chosen.update(top)
    in_firm = sum(w in firm_w for w in top)
    cover = pure.head(TOP).sum() / pure.sum() if pure.sum() else 0
    summary.append((coin, len(files), len(pure), len(top), in_firm, cover))

pd.DataFrame({"wallet": sorted(chosen)}).to_csv(OUT, index=False)
pd.DataFrame({"wallet": sorted(chosen - before)}).to_csv(OUT.replace(".csv", "_added.csv"), index=False)
print("\nTOP MAKERS IN THE CLEAN PLACEBO COINS")
print("   coin      days used  pure makers  top  already in our firms  top-30 share of pure-maker volume")
for coin, nd, npure, ntop, nin, cov in summary:
    print(f"   {coin:<9} {nd:>9} {npure:>12} {ntop:>4} {nin:>21} {cov:>12.1%}")
print(f"\nWallets for the request: {len(chosen):,} (before {len(before):,}, added {len(chosen - before):,}) "
      f"-> saved {OUT} and the added ones in {OUT.replace('.csv', '_added.csv')}")
