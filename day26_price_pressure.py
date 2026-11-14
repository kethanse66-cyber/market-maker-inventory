"""Day 26 (Nov 14): DOES MAKERS' INVENTORY MOVE THE PRICE? Pre-registration addendum v7 (tag prereg-v7).

Run:  python -W ignore E:\\p3\\day26_price_pressure.py > E:\\p3\\results\\day26_output.txt
About 10-30 minutes. Saves E:\\p3\\results\\part1_pressure.csv and part1_pressure_minutes.parquet (no wallets).

Unit: coin x minute. z = all maker firms' summed position / its std over the previous 1,440 minutes (cap +-5).
B = (log Hyperliquid mid - log Binance price) x 1e4 at minute end (both prices at most 5 s old).
H7a: B on z (a < 0).  H7b: B(t+5) - B(t) on z + past 5-min HL and Binance returns (b > 0).
Coin x day fixed effects, every coin weighted equally, standard errors clustered by day.
"""
import glob
import os
import re
import sys
import time

import duckdb
import numpy as np
import pandas as pd

INV = os.environ.get("P3_INV", r"E:\p3\data\inventory")
BOOK = os.environ.get("P3_BOOK", r"E:\hl_data\book")
BIN = os.environ.get("P3_BINANCE", r"E:\hl_data\binance")
DATA = os.environ.get("P3_DATA", r"E:\p3\data")
OUT = os.environ.get("P3_RESULTS", r"E:\p3\results")
COINS = os.environ.get("P3_COINS", "SOL,XRP,BTC,ETH,HYPE").split(",")   # env only for testing
CAP, STALE_S, H_MAIN, HS = 5.0, 5, 5, (1, 5, 15, 60)
NS = 1_000_000_000
MIN_NS = 60 * NS
t00 = time.time()
con = duckdb.connect()


def log(s):
    print(s, file=sys.stderr, flush=True)


def binance_seconds(coin, lo, hi):
    """Last Binance price in each second (same reader as day10_natives.py)."""
    sym = re.compile(rf"(^|[^A-Z0-9]){coin}(USDT|USDC|USD)?([^A-Z0-9]|$)")
    files = [f for f in glob.glob(os.path.join(BIN, "**", "*"), recursive=True)
             if os.path.isfile(f) and f.lower().endswith((".parquet", ".csv", ".csv.gz"))
             and sym.search(os.path.relpath(f, BIN).upper().replace("\\", "/"))]
    pqf = [f for f in files if f.lower().endswith(".parquet")]
    if pqf and len(pqf) != len(files):
        sys.exit(f"STOP: Binance {coin} files mix Parquet and CSV ({len(pqf)} of {len(files)} Parquet) - convert first")
    files, reader = (pqf, "read_parquet") if pqf else (files, "read_csv_auto")
    if not files:
        return None
    print(f"   Binance {coin}: {len(files)} files | first {os.path.relpath(sorted(files)[0], BIN)} | "
          f"last {os.path.relpath(sorted(files)[-1], BIN)}")
    name = {c.lower(): c for c in con.execute(f"DESCRIBE SELECT * FROM {reader}({files[:1]})").df()["column_name"]}
    if "symbol" in name:
        syms = [x[0] for x in con.execute(f'SELECT DISTINCT "{name["symbol"]}" FROM {reader}({files}, union_by_name=true)').fetchall()]
        bad = [x for x in syms if not sym.search(str(x).upper())]
        if bad:
            sys.exit(f"STOP: other symbols inside the Binance {coin} files: {bad[:5]}")
        print(f"   Binance {coin} symbols: {syms}")
    t = next((name[c] for c in ("open_time", "timestamp", "ts", "time", "t", "datetime", "date") if c in name), None)
    if "bid" in name and "ask" in name:
        px = f'("{name["bid"]}" + "{name["ask"]}") / 2'
    elif "bid_price" in name and "ask_price" in name:
        px = f'("{name["bid_price"]}" + "{name["ask_price"]}") / 2'
    else:
        p = next((name[c] for c in ("close", "mid", "price", "c", "close_price", "last") if c in name), None)
        px = f'"{p}"' if p else None
    if t is None or px is None:
        sys.exit(f"STOP: cannot find time/price columns in {files[0]}")
    print(f"   Binance {coin}: time column {t}, price {px} (a last price differs from a book mid by noise; "
          f"the constant part is absorbed by coin x day fixed effects)")
    ttype = con.execute(f'SELECT typeof("{t}") FROM {reader}({files[:1]}) LIMIT 1').fetchone()[0].upper()
    if "TIMESTAMP" in ttype or "DATE" in ttype:
        raw = f'epoch_ns("{t}")'
    else:
        mx = con.execute(f'SELECT max("{t}") FROM {reader}({files[:1]})').fetchone()[0]
        mult = {19: 1, 16: 1_000, 13: 1_000_000, 10: NS}.get(len(str(int(mx))))
        if mult is None:
            sys.exit(f"STOP: cannot tell the time unit of Binance {t} (max {mx})")
        raw = f'(CAST("{t}" AS BIGINT) * {mult})'
    return con.execute(f"""SELECT {raw} // {NS} AS s, arg_max(CAST({px} AS DOUBLE), {raw}) AS px
                           FROM {reader}({files}, union_by_name=true)
                           WHERE {raw} >= {lo} AND {raw} < {hi} GROUP BY 1 ORDER BY 1""").df()


def minute_end(sec, px):
    """Price at each minute's end: last observation in the minute, only if at most 5 s before the end."""
    s = pd.DataFrame({"s": sec, "px": px})
    s = s[np.isfinite(s["px"]) & (s["px"] > 0)]
    s["m"] = s["s"] // 60
    last = s.groupby("m").last()
    last = last[last["s"] >= last.index * 60 + 60 - STALE_S - 1]
    return np.log(last["px"])


frames, cover = [], []
for coin in COINS:
    fmf = os.path.join(INV, f"{coin}_firm_minute.parquet")
    if not os.path.exists(fmf):
        sys.exit(f"STOP: {fmf} not found")
    fm = pd.read_parquet(fmf)
    fm["m"] = (pd.to_datetime(fm["minute"], utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(minutes=1)  # any unit
    nat = None
    nf = os.path.join(DATA, "natives.csv")
    if os.path.exists(nf):
        n_ = pd.read_csv(nf)
        n_ = n_[(n_["level"].astype(str).str.lower() == "firm") & (n_["coin"].astype(str).str.upper() == coin)]
        lab = next((x for x in ("label_cal", "label") if x in n_.columns), None)
        if lab:
            nat = dict(zip(n_["unit"].astype(str), n_[lab].astype(str)))
    fm["lab"] = fm["firm"].astype(str).map(nat) if nat else "NA"
    Q = fm.groupby("m")["position"].sum(min_count=1)
    grid = np.arange(Q.index.min(), Q.index.max() + 1)
    Q = Q.reindex(grid)

    def z_of(series):
        sd = series.rolling(1440, min_periods=1440).std().shift(1)
        return (series / sd.where(sd > 0)).clip(-CAP, CAP)
    z = z_of(Q)
    zn = z_of(fm[fm["lab"] == "NATIVE"].groupby("m")["position"].sum(min_count=1).reindex(grid))
    zh = z_of(fm[fm["lab"] == "HEDGER"].groupby("m")["position"].sum(min_count=1).reindex(grid))

    # Hyperliquid mid at minute end (book snapshots), Binance price at minute end
    days = sorted({pd.Timestamp(int(m) * 60, unit="s", tz="UTC").strftime("%Y-%m-%d") for m in grid[::1440]} |
                  {pd.Timestamp(int(grid[-1]) * 60, unit="s", tz="UTC").strftime("%Y-%m-%d")})
    hl = []
    for d in days:
        f = os.path.join(BOOK, coin, f"{coin}_{d}.parquet")
        if not os.path.exists(f):
            continue
        b = pd.read_parquet(f, columns=["time", "bid_px", "ask_px"])
        b = b[(b["bid_px"] > 0) & (b["ask_px"] > b["bid_px"])]
        if b.empty:
            continue
        sec = (pd.to_datetime(b["time"], utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
        hl.append(minute_end(sec.to_numpy(), ((b["bid_px"] + b["ask_px"]) / 2).to_numpy(float)))
    if not hl:
        log(f"   {coin}: no book files - skipped")
        continue
    lhl = pd.concat(hl)
    lhl = lhl[~lhl.index.duplicated()]
    bn = binance_seconds(coin, int(grid[0]) * MIN_NS, (int(grid[-1]) + 1) * MIN_NS)
    if bn is None or bn.empty:
        log(f"   {coin}: no Binance prices - skipped")
        cover.append(dict(coin=coin, minutes=len(grid), with_z=int(z.notna().sum()), with_both_prices=0))
        continue
    lbn = minute_end(bn["s"].to_numpy(), bn["px"].to_numpy(float))
    bdays = pd.to_datetime(bn["s"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    print(f"   Binance {coin}: {bdays.nunique()} days with prices, {bdays.min()} to {bdays.max()}")
    if nat:
        lab_share = fm.assign(a=fm["position"].abs()).groupby("lab", dropna=False)["a"].sum()
        lab_share = (lab_share / lab_share.sum()).round(3).to_dict()
        print(f"   {coin} firm labels, share of |position|: {lab_share} (unlabelled firms count in z, not in the native/hedger split)")
    D = pd.DataFrame({"z": z, "zn": zn, "zh": zh}, index=grid)
    D["lhl"], D["lbn"] = lhl.reindex(grid), lbn.reindex(grid)
    D["B"] = (D["lhl"] - D["lbn"]) * 1e4
    for h in HS:
        D[f"dB{h}"] = D["B"].shift(-h) - D["B"]
    D["r_hl_past"] = (D["lhl"] - D["lhl"].shift(5)) * 1e4
    D["r_bn_past"] = (D["lbn"] - D["lbn"].shift(5)) * 1e4
    D["r_hl_fut"] = (D["lhl"].shift(-H_MAIN) - D["lhl"]) * 1e4
    D["coin"] = coin
    D["day"] = [pd.Timestamp(int(m) * 60, unit="s", tz="UTC").strftime("%Y-%m-%d") for m in grid]
    D.index.name = "minute"
    frames.append(D.reset_index())
    cover.append(dict(coin=coin, minutes=len(grid), with_z=int(z.notna().sum()),
                      with_both_prices=int(D["B"].notna().sum()), with_z_and_B=int((D["z"].notna() & D["B"].notna()).sum())))
    log(f"   {coin}: {int((D['z'].notna() & D['B'].notna()).sum()):,} minutes with z and basis | {time.time() - t00:,.0f}s")

if not frames:
    sys.exit("STOP: no coin has both maker inventory and Binance prices")
A = pd.concat(frames, ignore_index=True)
os.makedirs(OUT, exist_ok=True)
A.drop(columns=["lhl", "lbn"]).to_parquet(os.path.join(OUT, "part1_pressure_minutes.parquet"), index=False)


def pval(t):
    from math import erf, sqrt
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def fit(y, xs, data=None, coins=None):
    """WLS with coin x day FE (demeaned), every coin weighted equally, SE clustered by day."""
    d = (data if data is not None else A)
    if coins:
        d = d[d["coin"].isin(coins)]
    d = d[[y] + xs + ["coin", "day"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 100 or d["day"].nunique() < 5:
        return None
    g = pd.factorize(d["coin"] + d["day"])[0]
    w = (1.0 / d.groupby("coin")[y].transform("size")).to_numpy()
    sw = np.bincount(g, w)
    dm = lambda x: x - (np.bincount(g, w * x) / sw)[g]
    X = np.column_stack([dm(d[x].to_numpy(float)) for x in xs])
    Y = dm(d[y].to_numpy(float))
    Xw = X * w[:, None]
    Ai = np.linalg.inv(Xw.T @ X)
    beta = Ai @ (Xw.T @ Y)
    e = Y - X @ beta
    cl = pd.factorize(d["day"])[0]
    G = cl.max() + 1
    S = np.column_stack([np.bincount(cl, w * e * X[:, j], minlength=G) for j in range(X.shape[1])])
    se = np.sqrt(np.diag(Ai @ (S.T @ S) @ Ai) * G / (G - 1))
    return beta, se, len(d), G, d["coin"].nunique()


rows = []


def show(lab, r, j=0, test=""):
    if r is None:
        print(f"   {lab:<44} (no data)")
        return None
    beta, se, n, G, nc = r
    p = pval(beta[j] / se[j])
    print(f"   {lab:<44} {beta[j]:+.4f} (se {se[j]:.4f}, 95% [{beta[j] - 1.96 * se[j]:+.4f}, {beta[j] + 1.96 * se[j]:+.4f}], "
          f"p {p:.4f}) | {n:,} minutes, {nc} coins, {G} days")
    rows.append(dict(test=test, estimate=lab, value=beta[j], se=se[j], p=p, minutes=n, days=G, coins=nc))
    return beta[j], p


print("PRICE PRESSURE TEST (prereg-v7) - does makers' inventory move the Hyperliquid price vs Binance?")
print("   z = makers' summed position in sd units (+ = makers long); B = Hyperliquid minus Binance, bp")
print("   coverage: " + " | ".join(f"{c['coin']} {c.get('with_z_and_B', 0):,} of {c['minutes']:,} minutes" for c in cover))
print(f"   basis B: median |B| {A['B'].abs().median():.2f} bp, sd of B within coin-day "
      f"{A.groupby(['coin', 'day'])['B'].transform(lambda x: x - x.mean()).std():.2f} bp")

print("\nH7a  LEVEL: B on z  (pressure predicts a < 0; arbitrage-driven inventory would give a > 0)")
ra = show("a: B per 1 sd of maker inventory", fit("B", ["z"]), test="H7a")
print("\nH7b  REVERSAL: B(t+5) - B(t) on z + past 5-min returns  (predicts b > 0)")
rb = show("b: next-5-min HL-minus-Binance return per sd", fit("dB5", ["z", "r_hl_past", "r_bn_past"]), test="H7b")
if ra and rb:
    a_ok, b_ok = ra[0] < 0 and ra[1] < 0.05, rb[0] > 0 and rb[1] < 0.05
    v = ("SUPPORTED - makers' inventory moves the Hyperliquid price, and the move reverses" if a_ok and b_ok else
         "PARTIAL - level only (H7a)" if a_ok else "PARTIAL - reversal only (H7b)" if b_ok else "NOT SUPPORTED")
    print(f"\n   VERDICT (H7a and H7b): {v}")

print("\nSECONDARY (reported, not decisive)")
show("H7b + control for the current gap B", fit("dB5", ["z", "r_hl_past", "r_bn_past", "B"]), test="H7b + B")
for h in HS:
    if h != H_MAIN:
        show(f"H7b at h = {h} min", fit(f"dB{h}", ["z", "r_hl_past", "r_bn_past"]), test=f"H7b h={h}")
show("Hyperliquid's own next-5-min return per sd", fit("r_hl_fut", ["z", "r_hl_past", "r_bn_past"]), test="HL own return")
print("   each coin (H7a level | H7b reversal):")
for c in COINS:
    r1, r2 = fit("B", ["z"], coins=[c]), fit("dB5", ["z", "r_hl_past", "r_bn_past"], coins=[c])
    if r1 is not None and r2 is not None:
        print(f"      {c:<5} a {r1[0][0]:+.4f} (p {pval(r1[0][0] / r1[1][0]):.3f}) | b {r2[0][0]:+.4f} "
              f"(p {pval(r2[0][0] / r2[1][0]):.3f}) | {r1[2]:,} minutes")
        rows.append(dict(test="by coin", estimate=f"{c} a", value=r1[0][0], se=r1[1][0], p=pval(r1[0][0] / r1[1][0]),
                         minutes=r1[2], days=r1[3], coins=1))
        rows.append(dict(test="by coin", estimate=f"{c} b", value=r2[0][0], se=r2[1][0], p=pval(r2[0][0] / r2[1][0]),
                         minutes=r2[2], days=r2[3], coins=1))
print("   natives vs hedgers (firm labels; both in one model):")
r = fit("B", ["zn", "zh"])
if r is not None:
    show("level a, NATIVE firms' inventory", r, 0, test="natives a")
    show("level a, HEDGER firms' inventory", r, 1, test="hedgers a")
r = fit("dB5", ["zn", "zh", "r_hl_past", "r_bn_past"])
if r is not None:
    show("reversal b, NATIVE firms' inventory", r, 0, test="natives b")
    show("reversal b, HEDGER firms' inventory", r, 1, test="hedgers b")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "part1_pressure.csv"), index=False)
print(f"\nSaved part1_pressure.csv and part1_pressure_minutes.parquet in {OUT} | {(time.time() - t00) / 60:.0f} min")
