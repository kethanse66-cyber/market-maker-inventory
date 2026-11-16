"""Download trades for the new coins and find who makes the market there."""
import argparse
import functools
import os
import re
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from oxarchive import Client

DL = r"E:\sentinel\hl_dl3.py"
REQ = r"E:\p3\data\request_coin_days.csv"
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
chosen = set(firm_w)
summary = []
for coin, g in req.groupby("coin"):
    base = os.path.join(ROOT, coin)
    tdir = os.path.join(base, "trades")
    # crossing days are the middle of each +-1 block
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
print("\nTOP MAKERS IN THE NEW COINS")
print("   coin      days used  pure makers  top  already in our firms  top-30 share of pure-maker volume")
for coin, nd, npure, ntop, nin, cov in summary:
    print(f"   {coin:<9} {nd:>9} {npure:>12} {ntop:>4} {nin:>21} {cov:>12.1%}")
print(f"\nWallets for the request: {len(chosen):,} (our firms {len(firm_w):,} + new makers "
      f"{len(chosen) - len(firm_w):,}) -> saved {OUT}")
