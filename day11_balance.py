"""Balance check: are quotes just above and below the boundary alike apart from the tick?"""
import glob
import os
import re
import sys
import time

import duckdb
import numpy as np
import pandas as pd

QUOTES = os.environ.get("P3_QUOTES", r"E:\p3\data\quotes")
INV = os.environ.get("P3_INV", r"E:\p3\data\inventory")
EPISODES = os.environ.get("P3_EPISODES", r"E:\p3\data\episodes.csv")
BIN = os.environ.get("P3_BINANCE", r"E:\hl_data\binance")
OUT = os.environ.get("P3_BALANCE", r"E:\p3\data\balance.csv")
LEVEL = {"SOL": 100.0, "XRP": 1.0}
WINDOWS = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]
BAND_BP, CLIP, MAX_SMD, MIN_EPS, STALE_S = 5.0, 5.0, 0.10, 20, 5
COVS = ["vol5", "spread", "q", "tod_sin", "tod_cos", "basis"]
NS = 1_000_000_000


def log(s):
    print(s, file=sys.stderr, flush=True)


def binance_px(con, coin, days):
    sym = re.compile(rf"(^|[^A-Z0-9]){coin}(USDT|USDC|USD)?([^A-Z0-9]|$)")
    files = [f for f in glob.glob(os.path.join(BIN, "**", "*"), recursive=True)
             if os.path.isfile(f) and f.lower().endswith((".parquet", ".csv", ".csv.gz"))
             and sym.search(os.path.relpath(f, BIN).upper().replace("\\", "/"))]
    pq = [f for f in files if f.lower().endswith(".parquet")]
    files, reader = (pq, "read_parquet") if pq else (files, "read_csv_auto")
    if not files:
        return None
    name = {c.lower(): c for c in con.execute(f"DESCRIBE SELECT * FROM {reader}({files[:1]})").df()["column_name"]}
    t = next((name[c] for c in ("open_time", "timestamp", "ts", "time", "t") if c in name), None)
    p = next((name[c] for c in ("close", "mid", "price", "c") if c in name), None)
    if t is None or p is None:
        return None
    ttype = con.execute(f'SELECT typeof("{t}") FROM {reader}({files[:1]}) LIMIT 1').fetchone()[0].upper()
    if "TIMESTAMP" in ttype:
        raw = f'epoch_ns("{t}")'
    else:
        mx = con.execute(f'SELECT max("{t}") FROM {reader}({files[:1]})').fetchone()[0]
        raw = f'(CAST("{t}" AS BIGINT) * { {19: 1, 16: 1_000, 13: 1_000_000, 10: NS}[len(str(int(mx)))]})'
    lo = pd.Timestamp(min(days), tz="UTC").value
    hi = (pd.Timestamp(max(days), tz="UTC") + pd.Timedelta(days=1)).value
    return con.execute(f"""SELECT {raw} // {NS} AS s, arg_max(CAST("{p}" AS DOUBLE), {raw}) AS bpx
                           FROM {reader}({files}, union_by_name=true) WHERE {raw} >= {lo} AND {raw} < {hi}
                           GROUP BY 1 ORDER BY 1""").df()


ep = pd.read_csv(EPISODES, parse_dates=["start", "end"])
ep = ep[ep["coin"].isin(LEVEL) & (ep["level"] == ep["coin"].map(LEVEL)) & ep["good"].astype(bool)]
if not (ep["merge_min"] == 60).any():
    sys.exit("STOP: the frozen 60-minute episode merge is not in episodes.csv")
ep = ep[ep["merge_min"] == 60].reset_index(drop=True)
print(f"BALANCE CHECK - merge {int(ep['merge_min'].iloc[0])} min | {len(ep)} good episodes ({ep.groupby('coin').size().to_dict()}), outcomes not used")
con = duckdb.connect()
rows, bad_b = [], []
t0 = time.time()
for coin, E in ep.groupby("coin"):
    days = sorted(E["day"].unique())
    # ---- inventory scale
    prev = [(pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d") for d in days]
    miss_prev = [p_ for p_ in prev if not glob.glob(os.path.join(INV, coin, f"{p_}.parquet"))]
    if miss_prev:
        print(f"   {coin}: previous-day inventory missing for {miss_prev} -> q on those mornings uses less history")
    invf = sorted({f for d in set(days) | set(prev) for f in glob.glob(os.path.join(INV, coin, f"{d}.parquet"))})
    tcol = con.execute(f"DESCRIBE SELECT ts FROM read_parquet({invf}, union_by_name=true)").fetchone()[1]
    T = "epoch_ns(ts)" if "TIMESTAMP" in tcol.upper() else "ts"
    pm = con.execute(f"""SELECT lower(wallet) AS wallet, {T} // {60 * NS} AS minute, arg_max(pos_after, {T}) AS pos
                         FROM read_parquet({invf}, union_by_name=true) GROUP BY 1, 2""").df()
    sd_parts = []
    for d in days:
        g0 = pd.Timestamp(d, tz="UTC").value // (60 * NS)
        grid = np.arange(g0 - 1440, g0 + 1440)                        # previous day + episode day
        sub = pm[(pm["minute"] >= grid[0] - 1440) & (pm["minute"] < grid[-1] + 1)]
        P = sub.pivot_table(index="minute", columns="wallet", values="pos", aggfunc="last")
        P = P.reindex(np.arange(min(P.index.min(), grid[0]), grid[-1] + 1)).ffill()
        S = P.rolling(1440, min_periods=1440).std().shift(1)   # full 24 h required.loc[g0:g0 + 1439]      # strictly before the minute
        sd_parts.append(S.stack().rename("sd").reset_index().rename(columns={"level_0": "minute"}))
    sd = pd.concat(sd_parts, ignore_index=True)
    sd = sd[sd["sd"] > 0].sort_values("minute")
    bx = binance_px(con, coin, days)
    if bx is None:
        print(f"   {coin}: no Binance prices found - basis left out")
    L = LEVEL[coin]
    for e in E.itertuples():
        w0 = (e.start - pd.Timedelta(minutes=30)).value
        w1 = (e.end + pd.Timedelta(minutes=30)).value
        qf = sorted(glob.glob(os.path.join(QUOTES, coin, f"{e.day}*.parquet")))
        cols = set(con.execute(f"DESCRIBE SELECT * FROM read_parquet({qf}, union_by_name=true)").df()["column_name"])
        good = "book_quality = 'GOOD'" if "book_quality" in cols else "hour_crossed IS NOT NULL AND NOT poor_book"
        # all quotes in window + 5 min before
        q = con.execute(f"""
            SELECT lower(wallet) AS wallet, ts, mid, spread_bp, inventory, best_bid, best_ask,
                   (ticks_behind_best * tick / mid * 1e4 <= {BAND_BP}) AND NOT in_gap AND {good}
                   AND NOT crosses_book AND inventory IS NOT NULL AS main
            FROM read_parquet({qf}, union_by_name=true)
            WHERE ts >= {w0 - 300 * NS} AND ts < {w1} AND mid IS NOT NULL ORDER BY ts""").df()
        if q.empty:
            continue
        # vol5: realized vol of the per-second mid over the previous 300 s
        sec = (q["ts"] // NS).to_numpy()
        m = pd.Series(q["mid"].to_numpy(), index=sec).groupby(level=0).last()
        full = np.arange(m.index.min(), m.index.max() + 1)
        seen = pd.Series(full, index=full).where(pd.Series(full, index=full).isin(m.index)).ffill()
        age = full - seen.to_numpy()                                           # seconds since the last real mid
        m = m.reindex(full).ffill()
        fresh = age <= STALE_S
        ret = (np.log(m).diff() * 1e4).where(fresh & np.r_[False, fresh[:-1]])  # both ends fresh (<= 5 s)
        vol = ret.rolling(300, min_periods=150).std().shift(1)                 # half the 5 minutes must be valid
        q = q[q["main"] & (q["ts"] >= w0)]
        q = q[~((q["best_bid"] < L) & (q["best_ask"] >= L))]                  # straddle excluded
        if q.empty:
            continue
        tod = ((q["ts"] // NS) % 86400) / 3600
        q = q.assign(vol5=vol.reindex(q["ts"] // NS).to_numpy(), spread=q["spread_bp"],
                     tod_sin=np.sin(2 * np.pi * tod / 24), tod_cos=np.cos(2 * np.pi * tod / 24),
                     minute=q["ts"] // (60 * NS))
        q = pd.merge_asof(q.sort_values("minute"), sd[["wallet", "minute", "sd"]], on="minute", by="wallet",
                          direction="backward", allow_exact_matches=True)
        q["q"] = (q["inventory"] / q["sd"]).clip(-CLIP, CLIP)
        if bx is not None:
            b = bx.rename(columns={"s": "bs"})
            q = pd.merge_asof(q.assign(s=q["ts"] // NS).sort_values("s"), b, left_on="s", right_on="bs",
                              direction="backward")
            okb = (q["s"] - q["bs"] <= STALE_S) & np.isfinite(q["bpx"]) & (q["bpx"] > 0)
            q["basis"] = np.where(okb, (q["mid"] / q["bpx"].where(q["bpx"] > 0) - 1) * 1e4, np.nan)
            bad_b.append((int((~okb).sum()), len(q)))
        else:
            q["basis"] = np.nan
        q["dist"] = (q["mid"] - L).abs() / L
        q["above"] = q["mid"] >= L
        q["episode"] = f"{coin}_{e.Index}"
        rows.append(q[["episode", "above", "dist"] + COVS])
    log(f"   {coin} done | {time.time() - t0:,.0f}s")

D = pd.concat(rows, ignore_index=True)
nb_bad = sum(x for x, _ in bad_b)
nb_all = sum(y for _, y in bad_b)
print(f"   basis removed (Binance price stale, missing or not positive): {nb_bad / max(nb_all, 1):.2%} of quotes")
print(f"   quotes in the sample: {len(D):,} | covariate missing: " +
      ", ".join(f"{c} {D[c].isna().mean():.1%}" for c in COVS))
res = []
for w in WINDOWS:
    d = d_all = D[D["dist"] <= w]
    diff = {c: [] for c in COVS}
    scale = {c: D[c].std() for c in COVS}
    n_eps = 0
    for _, g in d.groupby("episode"):
        a, b = g[g["above"]], g[~g["above"]]
        if len(a) < 20 or len(b) < 20:
            continue
        n_eps += 1
        for c in COVS:
            xa, xb = a[c].dropna(), b[c].dropna()
            if len(xa) >= 10 and len(xb) >= 10:
                diff[c].append(xa.mean() - xb.mean())
    for c in COVS:
        res.append(dict(window_pct=w * 100, covariate=c, episodes_cov=len(diff[c]),
                        smd=np.mean(diff[c]) / scale[c] if diff[c] and scale[c] > 0 else np.nan,
                        episodes_both_sides=n_eps, quotes=len(d_all)))
res = pd.DataFrame(res)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
res.to_csv(OUT, index=False)

print("\n   standardized difference above - below (want |value| <= 0.10), averaged over episodes")
print("   episodes = episodes with 20+ quotes on each side; 'fewest' = the covariate with the fewest episodes\n"
      "   having 10+ valid values on each side (each covariate needs 20+)")
print("   window      quotes  episodes  " + "  ".join(f"{c:>7}" for c in COVS) + "   balanced?")
passing = []
for w, g in res.groupby("window_pct"):
    v = g.set_index("covariate")["smd"]
    ne = g.set_index("covariate")["episodes_cov"]
    ok = bool(v.notna().all() and v.abs().max() <= MAX_SMD and ne.min() >= MIN_EPS)
    if ok:
        passing.append(w)
    print(f"   {w:>5.2f}%  {int(g['quotes'].iloc[0]):>10,}  {int(g['episodes_both_sides'].iloc[0]):>8}  " +
          "  ".join(f"{v.get(c, np.nan):>+7.3f}" for c in COVS) +
          f"   {'YES' if ok else 'no'} (fewest episodes for one covariate: {int(ne.min())})")
if passing:
    print(f"\n   CHOSEN WINDOW: {max(passing):.2f}% of the boundary (widest balanced window) -> goes into the freeze")
else:
    print("\n   NO WINDOW BALANCES -> the tick result is reported as descriptive, not causal (plan rule)")
print(f"\nSaved {OUT}  ({time.time() - t0:,.0f} seconds)")
