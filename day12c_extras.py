"""Download book and trade data for the delivery 3 coins.
Usage: python day12c_extras.py [--plan]"""
import argparse
import csv
import io
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

import pandas as pd

ROOT = r"E:\hl_data"
LISTS = [r"E:\p3\data\request_coin_days.csv", r"E:\p3\data\clean_placebo_coin_days.csv"]
BINANCE_URL = "https://data.binance.vision/data/spot/daily/klines/{s}/1s/{s}-1s-{d}.zip"
BOOK_COLS = ["time", "bid_px", "bid_sz", "bid_n", "ask_px", "ask_sz", "ask_n", "mid", "spread_bps"]
NUM_TRADE = ["price", "size", "fee", "closed_pnl", "start_position", "builder_fee", "deployer_fee"]


# ---------------------------------------------------------------- helpers
def log(msg):
    line = f"{datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    try:
        os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
        with open(os.path.join(ROOT, "logs", "extras_log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    df.to_parquet(part, index=False)
    os.replace(part, path)


def utc(d):
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def read_coin_days():
    """Union of (coin, day) from the request lists."""
    out = set()
    for f in LISTS:
        if not os.path.exists(f):
            log(f"WARNING: {f} not found, skipped")
            continue
        df = pd.read_csv(f)
        low = {c.lower(): c for c in df.columns}
        cc = next(low[k] for k in ("coin", "symbol") if k in low)
        dc = next(low[k] for k in ("day", "date") if k in low)
        for c, d in zip(df[cc], df[dc]):
            out.add((str(c).strip().upper(), pd.Timestamp(d).date()))
    return out


def prev_days(coin_days):
    """Day before the first day of every run of consecutive days, per coin."""
    have = coin_days
    return {(c, d - timedelta(days=1)) for c, d in coin_days if (c, d - timedelta(days=1)) not in have}


def page_all(fn, coin, start, end, **kw):
    """All pages of a cursor-paged 0xArchive call, with retries."""
    recs, pages, cursor = [], 0, None
    while True:
        for attempt in range(6):
            try:
                r = fn(coin, start=start, end=end, cursor=cursor, limit=1000, **kw)
                break
            except Exception as ex:  # network / 429 / 5xx
                if attempt == 5:
                    raise
                wait = 5 * 2 ** attempt
                log(f"   retry {coin} {start:%Y-%m-%d} in {wait}s ({str(ex)[:120]})")
                time.sleep(wait)
        pages += 1
        recs.extend(r.data)
        more = getattr(r, "has_more", r.next_cursor is not None)
        if not more or not r.next_cursor:
            return recs, pages
        cursor = r.next_cursor


def dump(x):
    return x.model_dump() if hasattr(x, "model_dump") else dict(x)


# ---------------------------------------------------------------- 1
def get_trades(H, coin, d):
    path = os.path.join(ROOT, coin, "trades", f"{coin}_{d}.parquet")
    if os.path.exists(path):
        return "skip", path
    recs, pages = page_all(H.trades.list, coin, utc(d), utc(d + timedelta(days=1)))
    df = pd.DataFrame([dump(x) for x in recs])
    if len(df):
        df = df.dropna(axis=1, how="all")
        for c in NUM_TRADE:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        key = [c for c in ("trade_id", "user_address", "crossed") if c in df]
        df = df.drop_duplicates(subset=key or None).sort_values("timestamp").reset_index(drop=True)
    save(df, path)
    return f"{len(df):,} fills, {pages} pages", path


# ---------------------------------------------------------------- 2
def get_binance(coin, d):
    sym = f"{coin}USDT"
    path = os.path.join(ROOT, "binance", sym, f"{sym}_{d}.parquet")
    if os.path.exists(path):
        return "skip", path
    url = BINANCE_URL.format(s=sym, d=d)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                blob = r.read()
            break
        except urllib.error.HTTPError as ex:
            if ex.code == 404:
                return "not on Binance", None
            if attempt == 3:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(5 * 2 ** attempt)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0]).decode()
    df = parse_klines(raw)
    save(df, path)
    return f"{len(df):,} seconds", path


def parse_klines(raw):
    rows = [r for r in csv.reader(io.StringIO(raw)) if r and r[0].strip().isdigit()]  # drops a header if any
    df = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4, 5, 8]]
    df.columns = ["time", "open", "high", "low", "close", "volume", "trades"]
    t = df["time"].astype("int64")
    unit = "us" if t.iloc[0] > 10 ** 14 else "ms"  # Binance spot files switched to microseconds in 2025
    df["time"] = pd.to_datetime(t, unit=unit, utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype("int64")
    return df.sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------------- 3
def get_book(H, coin, d):
    path = os.path.join(ROOT, "book", coin, f"{coin}_{d}.parquet")
    if os.path.exists(path):
        return "skip", path
    s, e = utc(d), utc(d + timedelta(days=1))
    recs, pages = page_all(H.orderbook.history, coin, s, e, depth=1)
    df = book_frame([dump(x) for x in recs], s, e)
    save(df, path)
    return f"{len(df):,} snapshots, {pages} pages", path


def book_frame(snaps, s, e):
    rows = []
    for x in snaps:
        b = x["bids"][0] if x.get("bids") else None
        a = x["asks"][0] if x.get("asks") else None
        g = lambda lv, k: (lv[k] if isinstance(lv, dict) else getattr(lv, k)) if lv is not None else None
        rows.append((x["timestamp"], g(b, "px"), g(b, "sz"), g(b, "n"), g(a, "px"), g(a, "sz"), g(a, "n")))
    df = pd.DataFrame(rows, columns=BOOK_COLS[:7])
    if not len(df):
        return pd.DataFrame(columns=BOOK_COLS)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ("bid_px", "bid_sz", "ask_px", "ask_sz"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("bid_n", "ask_n"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["mid"] = (df["bid_px"] + df["ask_px"]) / 2
    df["spread_bps"] = (df["ask_px"] - df["bid_px"]) / df["mid"] * 1e4
    df = df[(df["time"] >= s) & (df["time"] < e)]
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------------- main
def run(jobs, workers, label):
    todo = [j for j in jobs if not os.path.exists(j[3])]
    log(f"== {label}: {len(jobs)} files, {len(jobs) - len(todo)} already on disk, {len(todo)} to get")
    t0, done, fails, notes = time.time(), 0, [], []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(j[0], *j[1:3]): j for j in todo}
        for f in as_completed(futs):
            args = futs[f][1:3]
            done += 1
            try:
                msg, _p = f.result()
                if msg == "not on Binance":
                    notes.append(args)
            except Exception as e:
                msg = f"FAILED {str(e)[:150]}"
                fails.append(args)
            el = time.time() - t0
            eta = el / done * (len(todo) - done) / 60
            log(f"   {label} {done}/{len(todo)} {args[0]} {args[1]}: {msg} | {el/60:.0f} min gone, ~{eta:.0f} min left")
    log(f"== {label} finished: {done - len(fails)} ok, {len(fails)} failed (re-run to retry)")
    return fails, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="only count what is missing")
    ap.add_argument("--workers", type=int, default=4, help="parallel API downloads (lower if you see many retries)")
    ap.add_argument("--only", choices=["trades", "binance", "book"], help="do one part only")
    a = ap.parse_args()

    cd = read_coin_days()
    pv = prev_days(cd)
    coins = sorted({c for c, _ in cd})
    log(f"coin-days: {len(cd)} | coins {len(coins)}: {' '.join(coins)} | runs (= day-before fill files): {len(pv)}")

    tr = [(None, c, d, os.path.join(ROOT, c, "trades", f"{c}_{d}.parquet")) for c, d in sorted(pv)]
    bn = [(get_binance, c, d, os.path.join(ROOT, "binance", f"{c}USDT", f"{c}USDT_{d}.parquet")) for c, d in sorted(cd)]
    bk = [(None, c, d, os.path.join(ROOT, "book", c, f"{c}_{d}.parquet")) for c, d in sorted(cd)]
    if a.plan:
        for lab, js in (("day-before fills", tr), ("Binance 1s", bn), ("top of book", bk)):
            miss = sum(not os.path.exists(j[3]) for j in js)
            log(f"   {lab}: {len(js)} files, {miss} missing")
        log(f"   book needs about {sum(not os.path.exists(j[3]) for j in bk) * 175:,} API calls")
        return

    H = None
    if a.only in (None, "trades", "book"):
        from oxarchive import Client
        H = Client(api_key=os.environ["OXARCHIVE_API_KEY"], timeout=180).hyperliquid
    tr = [(lambda c, d: get_trades(H, c, d), c, d, p) for _, c, d, p in tr]
    bk = [(lambda c, d: get_book(H, c, d), c, d, p) for _, c, d, p in bk]

    allfail = []
    if a.only in (None, "trades"):
        allfail += run(tr, a.workers, "fills")[0]
    if a.only in (None, "binance"):
        f, notes = run(bn, 8, "binance")
        allfail += f
        if notes:
            miss = pd.DataFrame(sorted(notes), columns=["coin", "day"])
            os.makedirs(os.path.join(ROOT, "binance"), exist_ok=True)
            miss.to_csv(os.path.join(ROOT, "binance", "missing_new_coins.csv"), index=False)
            log("   not on Binance spot (coin: days): " +
                ", ".join(f"{c}: {n}" for c, n in miss.groupby("coin").size().items()))
    if a.only in (None, "book"):
        allfail += run(bk, a.workers, "book")[0]
    log("ALL DONE" if not allfail else f"DONE WITH {len(allfail)} FAILED FILES - run the same command again")


if __name__ == "__main__":
    main()
