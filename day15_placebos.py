"""Day 15 (Nov 13): PLACEBO LEVELS + per-coin results - pre-registration prereg-v1, section 7.

Run:  python E:\\p3\\day15_placebos.py > E:\\p3\\results\\day15_output.txt      (about 1-2 hours)
Saves E:\\p3\\results\\placebo_result.csv

At levels where the tick does NOT change (SOL $95, $105, $110, $120; XRP $1.10, $1.30, $1.40, $1.50) the same
sample rules, model and circular-shift labels are applied with that level as the "boundary".
Only episodes inside the frozen test window (Aug 15 - Oct 25, 2026) are used, as in the main test. If the real effects
come from the tick, these placebo effects must be ~0.
Frozen rule (section 7): each placebo estimate must be shown smaller than 25% of the main estimate:
its 90% interval (estimate +- 1.645 x the sd of its circular-shift draws) must lie inside +-0.25 x |main|.
Applied to the side tau (primary; main was not significant, so this is descriptive) and to the three secondary
outcomes. The real SOL $100 and XRP $1.00 levels are also run one coin at a time (component results).
500 circular-shift draws per level (enough for the sd and a placebo p), seed 20261112.
"""
import glob
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd

QUOTES = os.environ.get("P3_QUOTES", r"E:\p3\data\quotes")
INV = os.environ.get("P3_INV", r"E:\p3\data\inventory")
EPISODES = os.environ.get("P3_EPISODES", r"E:\p3\data\episodes.csv")
BOOK = os.environ.get("P3_BOOK", r"E:\hl_data\book")
POWER = os.environ.get("P3_POWER_OUT", r"E:\p3\data\power_check.csv")
OUT = os.environ.get("P3_RESULTS", r"E:\p3\results")
LEVEL = {"SOL": 100.0, "XRP": 1.0}
BAND_BP, CLIP, WINDOW_PCT, STALE_S = 5.0, 5.0, 0.005, 5
DRAWS, SEED, MIN_MIN, WILD = 2000, 20261112, 5, 1999
DRAWS = int(os.environ.get("P3_TEST_DRAWS", DRAWS))  # only for testing; the real run uses 2,000
NS = 1_000_000_000
MIN_NS = 60 * NS


def log(s):
    print(s, file=sys.stderr, flush=True)


con = duckdb.connect()


def files_for(folder, coin, days):
    return sorted({f for d in days for f in glob.glob(os.path.join(folder, coin, f"{d}*.parquet"))})


def near_days(t0, t1):
    return sorted({(t0 - pd.Timedelta(days=k)).strftime("%Y-%m-%d") for k in (0, 1)} |
                  {t1.strftime("%Y-%m-%d")})


# ---------------------------------------------------------------- 2. sample, episode by episode
def load(ep):
    global D, EP, LAB, OFF, MM, ROT
    parts, ep_rows, lab_all, lab_off = [], [], [], []
    drop = dict(no_q=0, no_vol=0, outside_window=0, straddle=0)
    t0 = time.time()
    for e in ep.itertuples():
        coin, L = e.coin, float(e.level)
        W0 = (e.start - pd.Timedelta(minutes=30)).value // MIN_NS          # first minute of the window
        W1 = (e.end + pd.Timedelta(minutes=30)).value // MIN_NS            # last minute (inclusive)
        M = int(W1 - W0 + 1)
        days = near_days(pd.Timestamp(W0 * MIN_NS, tz="UTC"), pd.Timestamp(W1 * MIN_NS, tz="UTC"))

        # minute labels from the Hyperliquid book: median mid in the minute >= boundary
        bf = [os.path.join(BOOK, coin, f"{coin}_{d}.parquet") for d in days]
        bf = [f for f in bf if os.path.exists(f)]
        if not bf:
            sys.exit(f"STOP: no book file for {coin} on {days} in {BOOK}")
        b = pd.concat([pd.read_parquet(f, columns=["time", "bid_px", "ask_px"]) for f in bf], ignore_index=True)
        bt = (pd.to_datetime(b["time"], utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(minutes=1)  # any time unit
        ok = (b["bid_px"] > 0) & (b["ask_px"] > b["bid_px"]) & (bt >= W0 - 60) & (bt <= W1)
        med = ((b["bid_px"] + b["ask_px"]) / 2)[ok].groupby(bt[ok]).median()
        mids = med.reindex(np.arange(W0 - 60, W1 + 1))
        no_book = int(mids.loc[W0:].isna().sum())
        if no_book > M // 2:
            log(f"   {coin} {L:g}: episode {e.day} {e.start:%H:%M} SKIPPED - {no_book} of {M} minutes without book data")
            drop["skipped_no_book"] = drop.get("skipped_no_book", 0) + 1
            continue
        # a minute without a snapshot keeps the LAST KNOWN label (past data only, never a later minute)
        mids = mids.ffill().loc[W0:W1]
        if mids.isna().any():
            log(f"   {coin} {L:g}: episode {e.day} {e.start:%H:%M} SKIPPED - window starts inside a book gap")
            drop["skipped_no_book"] = drop.get("skipped_no_book", 0) + 1
            continue
        lab = (mids.to_numpy() >= L).astype(np.int8)
        n_up, n_dn = int(lab.sum()), int(M - lab.sum())
        rotate = n_up >= MIN_MIN and n_dn >= MIN_MIN

        # quotes: window + 5 min before (volatility), all quotes for the mid series
        qf = files_for(QUOTES, coin, days)
        if not qf:
            sys.exit(f"STOP: no quote files for {coin} on {days} in {QUOTES}")
        cols = set(con.execute(f"DESCRIBE SELECT * FROM read_parquet({qf}, union_by_name=true)").df()["column_name"])
        good = "book_quality = 'GOOD'" if "book_quality" in cols else "hour_crossed IS NOT NULL AND NOT poor_book"
        q = con.execute(f"""
            SELECT lower(wallet) AS wallet, ts, side, mid, spread_bp, inventory, best_bid, best_ask,
                   price, size, ticks_behind_best,
                   (ticks_behind_best * tick / mid * 1e4 <= {BAND_BP}) AND NOT in_gap AND {good}
                   AND NOT crosses_book AND inventory IS NOT NULL AS main
            FROM read_parquet({qf}, union_by_name=true)
            WHERE ts >= {W0 * MIN_NS - 300 * NS} AND ts < {(W1 + 1) * MIN_NS} AND mid IS NOT NULL
            ORDER BY ts""").df()

        # vol5: identical to the frozen balance code (day11_balance.py)
        sec = (q["ts"] // NS).to_numpy()
        m = pd.Series(q["mid"].to_numpy(), index=sec).groupby(level=0).last()
        full = np.arange(m.index.min(), m.index.max() + 1)
        seen = pd.Series(full, index=full).where(pd.Series(full, index=full).isin(m.index)).ffill()
        age = full - seen.to_numpy()
        m = m.reindex(full).ffill()
        fresh = age <= STALE_S
        ret = (np.log(m).diff() * 1e4).where(fresh & np.r_[False, fresh[:-1]])
        vol = ret.rolling(300, min_periods=150).std().shift(1)

        q = q[q["main"] & (q["ts"] >= W0 * MIN_NS)]
        n0 = len(q)
        q = q[~((q["best_bid"] < L) & (q["best_ask"] >= L))]
        drop["straddle"] += n0 - len(q)
        n0 = len(q)
        q = q[(q["mid"] - L).abs() / L <= WINDOW_PCT]
        drop["outside_window"] += n0 - len(q)

        # q: position / std of the minute-by-minute position over the previous 1,440 minutes (flat minutes count)
        invf = files_for(INV, coin, sorted(set(days) | {(pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                                                         for d in days}))
        if not invf:
            sys.exit(f"STOP: no inventory files for {coin} on {days} in {INV}")
        tcol = con.execute(f"DESCRIBE SELECT ts FROM read_parquet({invf}, union_by_name=true)").fetchone()[1]
        T = "epoch_ns(ts)" if "TIMESTAMP" in tcol.upper() else "ts"
        pm = con.execute(f"""SELECT lower(wallet) AS wallet, {T} // {MIN_NS} AS minute, arg_max(pos_after, {T}) AS pos
                             FROM read_parquet({invf}, union_by_name=true)
                             WHERE {T} // {MIN_NS} BETWEEN {W0 - 2880} AND {W1} GROUP BY 1, 2""").df()
        wal = set(q["wallet"])
        pm = pm[pm["wallet"].isin(wal)]
        if len(pm):
            P = pm.pivot_table(index="minute", columns="wallet", values="pos", aggfunc="last")
            P = P.reindex(np.arange(min(P.index.min(), W0 - 1440), W1 + 1)).ffill()
            S = P.rolling(1440, min_periods=1440).std().shift(1).loc[W0:W1]
            sd = S.stack().rename("sd").reset_index().rename(columns={"level_0": "minute"})
            sd = sd[sd["sd"] > 0]
        else:
            sd = pd.DataFrame(columns=["minute", "wallet", "sd"])
        q = q.assign(minute=q["ts"] // MIN_NS).merge(sd, on=["minute", "wallet"], how="left")
        q["q"] = (q["inventory"] / q["sd"]).clip(-CLIP, CLIP)
        q["vol5"] = vol.reindex(q["ts"] // NS).to_numpy()
        drop["no_q"] += int(q["q"].isna().sum())
        drop["no_vol"] += int((q["q"].notna() & q["vol5"].isna()).sum())
        q = q[q["q"].notna() & q["vol5"].notna()]

        k = len(lab_off)
        q = q.assign(episode=k, coin=coin, day=e.day, mi=(q["minute"] - W0).astype(np.int64),
                     above_own=(q["mid"] >= L).astype(np.int8), bid=(q["side"] == "B").astype(np.int8),
                     maker=coin + "_" + q["wallet"])
        parts.append(q[["episode", "coin", "day", "maker", "ts", "mi", "bid", "q", "vol5", "spread_bp",
                        "above_own", "mid", "inventory", "price", "size", "ticks_behind_best"]])
        lab_off.append(sum(len(x) for x in lab_all))
        lab_all.append(lab)
        ep_rows.append(dict(episode=k, coin=coin, day=e.day, start=e.start, end=e.end, minutes=M, above_min=n_up,
                            below_min=n_dn, rotated=rotate, minutes_without_book=no_book, quotes=len(q),
                            makers=q["maker"].nunique()))
        log(f"   {coin} {L:g}: episode {k + 1}/{len(ep)} {e.day}: {len(q):,} quotes | {time.time() - t0:,.0f}s")

    if not parts:
        return None
    D = pd.concat(parts, ignore_index=True)
    if len(D) < 1000 or D["maker"].nunique() < 5 or D["bid"].nunique() < 2:
        return None
    EP = pd.DataFrame(ep_rows)
    LAB = np.concatenate(lab_all)
    OFF = np.array(lab_off, dtype=np.int64)
    MM = EP["minutes"].to_numpy(np.int64)
    ROT = EP["rotated"].to_numpy(bool)
    return drop


class Fit:
    """WLS of y on [z, a, z*a, v, v*a] with group FE; only a (the minute label) changes per draw."""

    def __init__(self, y, z, groups):
        self.y, self.z, self.v = y, z, D["vol5"].to_numpy(float)
        mk = pd.factorize(D["maker"])[0]
        self.w = 1.0 / np.bincount(mk)[mk]                       # every maker counts equally
        self.g = pd.factorize(groups)[0]
        self.sw = np.bincount(self.g, self.w)
        self.cl = pd.factorize(D["coin"] + D["day"])[0]
        self.yd, self.zd, self.vd = self.dm(y), self.dm(z), self.dm(self.v)

    def dm(self, x):
        return x - (np.bincount(self.g, self.w * x) / self.sw)[self.g]

    def X(self, a):
        return np.column_stack([self.zd, self.dm(a), self.dm(self.z * a), self.vd, self.dm(self.v * a)])

    def beta(self, a):
        X = self.X(a)
        Xw = X * self.w[:, None]
        return np.linalg.solve(Xw.T @ X, Xw.T @ self.yd)

    def clustered(self, a):
        X = self.X(a)
        Xw = X * self.w[:, None]
        Ai = np.linalg.inv(Xw.T @ X)
        b = Ai @ (Xw.T @ self.yd)
        e = self.yd - X @ b
        G = self.cl.max() + 1
        S = np.column_stack([np.bincount(self.cl, self.w * e * X[:, j], minlength=G) for j in range(5)])
        return b, np.sqrt(np.diag(Ai @ (S.T @ S) @ Ai * G / (G - 1)))



EPI = MI = None


def labels(k):
    return LAB[OFF[EPI] + (MI + k[EPI]) % MM[EPI]].astype(float)


DRAWS_P = int(os.environ.get("P3_TEST_DRAWS", 500))
main_r = pd.read_csv(os.path.join(OUT, "main_result.csv"))
sec_r = pd.read_csv(os.path.join(OUT, "secondary_result.csv"))
MAIN = {"side tau": float(main_r["tau"].iloc[0]),
        "price tau": float(sec_r.loc[sec_r["outcome"].str.startswith("1"), "estimate"].iloc[0]),
        "size tau": float(sec_r.loc[sec_r["outcome"].str.startswith("2"), "estimate"].iloc[0]),
        "at-best jump": float(sec_r.loc[sec_r["outcome"].str.startswith("3"), "estimate"].iloc[0])}
print("PLACEBO LEVELS (pre-registration section 7) + component results at the real boundaries")
print("   main pooled estimates: " + ", ".join(f"{k} {v:+.4f}" for k, v in MAIN.items()))

epall = pd.read_csv(EPISODES, parse_dates=["start", "end"])
epall = epall[(epall["merge_min"] == 60) & epall["good"].astype(bool)]
if not os.environ.get("P3_ALL_DAYS"):              # SOL/XRP: frozen test window only (June = pilot)
    epall = epall[(epall["start"] >= "2026-08-15") & (epall["start"] < "2026-10-26")]
RUNS = [("SOL", 100.0, "REAL"), ("XRP", 1.0, "REAL")] + [("SOL", x, "placebo") for x in (95.0, 105.0, 110.0, 120.0)] + \
       [("XRP", x, "placebo") for x in (1.10, 1.30, 1.40, 1.50) + tuple(float(v) for v in os.environ.get("P3_TEST_LEVELS", "").split() if v)]
RUNS += [(x.split(":")[0].upper(), float(x.split(":")[1]), "placebo")              # e.g. P3_EXTRA="PENGU:0.01"
         for x in os.environ.get("P3_EXTRA", "").split() if x]
RUNS += [(x.split(":")[0].upper(), float(x.split(":")[1]), "REAL")                # replication boundaries
         for x in os.environ.get("P3_EXTRA_REAL", "").split() if x]
ONLY = os.environ.get("P3_ONLY", "")                               # e.g. "XRP" to re-run only XRP levels
if ONLY:
    RUNS = [r for r in RUNS if r[0] == ONLY or ONLY in r[0].split("+")]
CO = os.environ.get("P3_COINS_ONLY", "").split()                  # e.g. the replication coins
if CO:
    RUNS = [r for r in RUNS if r[0] in CO]
LV = [float(x) for x in os.environ.get("P3_LEVELS", "").split()]  # e.g. "1.1 1.3" to re-run only those levels
if LV:
    RUNS = [r for r in RUNS if any(np.isclose(r[1], x) for x in LV)]
SUFFIX = os.environ.get("P3_SUFFIX", f"_{ONLY.lower()}" if ONLY else "")
SEED = int(os.environ.get("P3_SEED", SEED))
rows = []
t00 = time.time()
for coin, lvl, kind in RUNS:
    ep = epall[epall["coin"].isin(coin.split("+")) & np.isclose(epall["level"], lvl)]       # "A+B" = pooled coins
    ep = ep.sort_values(["coin", "start"]).reset_index(drop=True)
    tag = f"{coin} ${lvl:g} ({kind})"
    if ep.empty:
        print(f"\n   {tag}: no good episodes")
        rows.append(dict(level=tag, episodes=0))
        continue
    drop = load(ep)
    if drop is None:
        print(f"\n   {tag}: {len(ep)} episodes but too few usable quotes")
        rows.append(dict(level=tag, episodes=len(ep)))
        continue
    EPI, MI = D["episode"].to_numpy(np.int64), D["mi"].to_numpy(np.int64)
    s = np.where(D["bid"].to_numpy() == 1, -1.0, 1.0)
    qv = D["q"].to_numpy(float)
    me = D["maker"] + "#" + D["episode"].astype(str)
    mse = D["maker"] + "#" + D["bid"].astype(str) + "#" + D["episode"].astype(str)
    outs = {"side tau": (Fit(D["bid"].to_numpy(float), qv, me), 2),
            "price tau": (Fit(((D["price"] - D["mid"]) / D["mid"] * 1e4).to_numpy(float), qv, mse), 2),
            "size tau": (Fit(np.log(D["size"].to_numpy(float)), qv * s, mse), 2),
            "at-best jump": (Fit((D["ticks_behind_best"].round(6) <= 0).to_numpy(float), qv, mse), 1)}
    a0 = labels(np.zeros(len(EP), np.int64))
    est = {n: f.beta(a0)[j] for n, (f, j) in outs.items()}
    rng = np.random.default_rng(SEED)
    dr = np.empty((DRAWS_P, len(outs)))
    for r in range(DRAWS_P):
        if (r + 1) % 100 == 0:
            log(f"   {tag}: shuffle {r + 1}/{DRAWS_P}")
        k = np.where(ROT, rng.integers(1, np.maximum(MM, 2)), 0)
        a = labels(k)
        row = []
        for f, j in outs.values():
            try:
                row.append(f.beta(a)[j])
            except np.linalg.LinAlgError:          # a rotation left no above/below contrast: not estimable
                row.append(np.nan)
        dr[r] = row
    if drop.get("skipped_no_book"):
        print(f"\n   {tag}: {drop['skipped_no_book']} episode(s) skipped - window inside an order-book outage")
    print(f"\n   {tag}: {len(EP)} episodes ({int(ROT.sum())} rotated) on {EP['day'].nunique()} days | "
          f"{len(D):,} quotes, {D['maker'].nunique()} makers | {time.time() - t00:,.0f}s")
    for i, n in enumerate(outs):
        v = dr[:, i][np.isfinite(dr[:, i])]
        sd = v.std() if len(v) else np.nan
        p = (1 + np.sum(np.abs(v) >= abs(est[n]))) / (len(v) + 1)
        if len(v) < DRAWS_P:
            print(f"      ({n}: {DRAWS_P - len(v)} of {DRAWS_P} shuffles could not be estimated and are left out)")
        lim = 0.25 * abs(MAIN[n])
        lo, hi = est[n] - 1.645 * sd, est[n] + 1.645 * sd
        ok = (lo >= -lim) and (hi <= lim)
        rows.append(dict(level=tag, kind=kind, outcome=n, estimate=est[n], p_randomization=p, draw_sd=sd,
                         ci90_lo=lo, ci90_hi=hi, limit_25pct_main=lim,
                         placebo_passes=ok if kind == "placebo" else np.nan, episodes=len(EP), quotes=len(D)))
        verdict = ("PASS (shown < 25% of main)" if ok else "NOT SHOWN small (interval too wide or effect)") \
            if kind == "placebo" else "component result"
        print(f"      {n:<13} {est[n]:+.4f}  p {p:.3f}  90% [{lo:+.4f}, {hi:+.4f}]  limit +-{lim:.4f}  {verdict}")

R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, f"placebo_result{SUFFIX}.csv"), index=False)
P = R[R["kind"] == "placebo"] if "kind" in R else R.iloc[0:0].assign(outcome="", placebo_passes=False, p_randomization=1.0, estimate=0.0)
print("\nSUMMARY OVER THE PLACEBO LEVELS (each outcome)")
for n in MAIN:
    g = P[P["outcome"] == n]
    if len(g):
        print(f"   {n:<13} levels run {len(g)} | pass {int(g['placebo_passes'].sum())} | "
              f"placebo p < 0.05 at {int((g['p_randomization'] < 0.05).sum())} levels | "
              f"mean placebo estimate {g['estimate'].mean():+.4f} vs main {MAIN[n]:+.4f}")
print(f"\nSaved {os.path.join(OUT, f'placebo_result{SUFFIX}.csv')} ({time.time() - t00:,.0f} seconds)")
