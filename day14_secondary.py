"""Day 14 (Nov 13): SECONDARY OUTCOMES - pre-registration prereg-v1, section 6 (Holm-corrected family of 3).

Run:  python E:\\p3\\day14_secondary.py > E:\\p3\\results\\day14_output.txt      (about 30-60 minutes)
Saves E:\\p3\\results\\secondary_result.csv and secondary_draws.csv.

Same sample, episodes, q, vol5, weights (each maker equal) and circular-shift randomization (2,000 draws, seed
20261112, same offsets as the main test) as day13_main_test.py. Only the outcome changes:
  1. PRICE SHADING  y = (quote price - mid) / mid in bp.  Model y = b*q + c*a + tau*(q x a) + d*vol5 + e*(vol5 x a)
     + FE(maker x side x episode). Pre-registered "above" = the quote's OWN price at/above the boundary; that tau is
     printed. The randomization rotates minute labels, so the p-value compares minute-label taus (as in the main test).
     tau > 0 on q x a: with the coarse tick makers shade price LESS per unit of inventory (b is negative: long -> lower prices).
  2. SIZE SKEW      y = log quote size; regressor q*s (s = +1 ask, -1 bid): b > 0 = bigger quotes on the side that
     unwinds. tau on (q*s) x a. FE maker x side x episode.
  3. PRESENCE AT THE BEST  y = 1 if the quote is at (or better than) the market best on its side; estimand = the level
     jump c at the boundary. FE maker x side x episode.
Each outcome: randomization p (decides), then Holm across the 3; clustered t (13 coin-days) shown, not decisive.
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


# ---------------------------------------------------------------- 1. episodes
ep = pd.read_csv(EPISODES, parse_dates=["start", "end"])
ep = ep[ep["coin"].isin(LEVEL) & (ep["level"] == ep["coin"].map(LEVEL)) & ep["good"].astype(bool)]
ep = ep[ep["merge_min"] == 60].sort_values(["coin", "start"]).reset_index(drop=True)
print("SECONDARY OUTCOMES (pre-registration prereg-v1, section 6; Holm over 3)")
print(f"   episodes: {len(ep)} good, merge 60 min | " +
      ", ".join(f"{c}: {len(g)} on {g['day'].nunique()} days" for c, g in ep.groupby("coin")))
if len(ep) != 31 or ep.groupby(["coin", "day"]).ngroups != 14:
    sys.exit("STOP: the frozen design is 31 episodes on 14 days - episodes.csv does not match")

con = duckdb.connect()


def files_for(folder, coin, days):
    return sorted({f for d in days for f in glob.glob(os.path.join(folder, coin, f"{d}*.parquet"))})


def near_days(t0, t1):
    return sorted({(t0 - pd.Timedelta(days=k)).strftime("%Y-%m-%d") for k in (0, 1)} |
                  {t1.strftime("%Y-%m-%d")})


# ---------------------------------------------------------------- 2. sample, episode by episode
parts, ep_rows, lab_all, lab_off = [], [], [], []
drop = dict(no_q=0, no_vol=0, outside_window=0, straddle=0)
t0 = time.time()
for e in ep.itertuples():
    coin, L = e.coin, LEVEL[e.coin]
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
        sys.exit(f"STOP: episode {coin} {e.day}: {no_book} of {M} minutes have no book mid - check {BOOK}")
    # a minute without a snapshot keeps the LAST KNOWN label (past data only, never a later minute)
    mids = mids.ffill().loc[W0:W1]
    if mids.isna().any():
        sys.exit(f"STOP: episode {coin} {e.day}: no book mid in the hour before the window start")
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
    log(f"   episode {k + 1}/31 {coin} {e.day}: {len(q):,} quotes | {time.time() - t0:,.0f}s")

D = pd.concat(parts, ignore_index=True)
if len(D) < 1000 or D["maker"].nunique() < 5 or D["bid"].nunique() < 2 or D["coin"].nunique() < 2:
    sys.exit(f"STOP: analysis sample too small or one-sided: {len(D)} quotes, {D['maker'].nunique()} makers")
EP = pd.DataFrame(ep_rows)
LAB = np.concatenate(lab_all)
OFF = np.array(lab_off, dtype=np.int64)
MM = EP["minutes"].to_numpy(np.int64)
ROT = EP["rotated"].to_numpy(bool)
os.makedirs(OUT, exist_ok=True)

print(f"\n1. SAMPLE")
print(f"   quotes {len(D):,} | makers {D['maker'].nunique()} | coin-days {D.groupby(['coin', 'day']).ngroups} | "
      + " | ".join(f"{c}: {len(g):,} quotes, {g['maker'].nunique()} makers" for c, g in D.groupby("coin")))
print(f"   dropped: straddle {drop['straddle']:,}, outside 0.5% window {drop['outside_window']:,}, "
      f"no q (under 24 h of history or flat) {drop['no_q']:,}, no vol5 {drop['no_vol']:,}")
print(f"   bid share {D['bid'].mean():.3f} | median |q| {D['q'].abs().median():.2f} | capped at 5: "
      f"{(D['q'].abs() >= CLIP).mean():.1%} | minutes without a book snapshot: {int(EP['minutes_without_book'].sum())}")

print(f"\n2. EPISODES IN THE RANDOMIZATION (reported before tau, section 5 step 5)")
print(f"   rotated: {int(ROT.sum())} of {len(EP)} | fixed (< {MIN_MIN} minutes on one side): {int((~ROT).sum())}")
for c, g in EP.groupby("coin"):
    print(f"   {c}: rotated {int(g['rotated'].sum())}, fixed {int((~g['rotated']).sum())} | "
          f"minutes above {int(g['above_min'].sum())}, below {int(g['below_min'].sum())}")


# ---------------------------------------------------------------- 3. the estimator


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


EPI = D["episode"].to_numpy(np.int64)
MI = D["mi"].to_numpy(np.int64)


def labels(k):
    return LAB[OFF[EPI] + (MI + k[EPI]) % MM[EPI]].astype(float)


L_of = D["coin"].map(LEVEL).to_numpy(float)
s = np.where(D["bid"].to_numpy() == 1, -1.0, 1.0)              # s = +1 ask, -1 bid
mse = D["maker"] + "#" + D["bid"].astype(str) + "#" + D["episode"].astype(str)
qv = D["q"].to_numpy(float)
own_price_above = (D["price"].to_numpy(float) >= L_of).astype(float)
OUTS = {
    "1 price shading (bp)": (Fit(((D["price"] - D["mid"]) / D["mid"] * 1e4).to_numpy(float), qv, mse), 2, own_price_above),
    "2 size skew (log size)": (Fit(np.log(D["size"].to_numpy(float)), qv * s, mse), 2, None),
    "3 presence at best (share)": (Fit((D["ticks_behind_best"].round(6) <= 0).to_numpy(float), qv, mse), 1, None),
}
k0 = np.zeros(len(EP), np.int64)
a0 = labels(k0)
obs = {n: f.beta(a0)[j] for n, (f, j, _) in OUTS.items()}

rng = np.random.default_rng(SEED)                                # same seed -> same rotations as the main test
draws = np.empty((DRAWS, len(OUTS)))
t1 = time.time()
for r in range(DRAWS):
    k = np.where(ROT, rng.integers(1, np.maximum(MM, 2)), 0)
    if np.any(k[ROT] < 1) or np.any(k[ROT] > MM[ROT] - 1) or np.any(k[~ROT] != 0):
        sys.exit("STOP: a rotation offset is outside 1..M-1 or a fixed episode moved")
    a = labels(k)
    draws[r] = [f.beta(a)[j] for f, j, _ in OUTS.values()]
    if (r + 1) % 200 == 0:
        log(f"   draw {r + 1}/{DRAWS} | {time.time() - t1:,.0f}s")
pd.DataFrame(draws, columns=list(OUTS)).to_csv(os.path.join(OUT, "secondary_draws.csv"), index=False)

rows = []
for i, (name, (f, j, own)) in enumerate(OUTS.items()):
    b, se = f.clustered(a0)
    p = (1 + np.sum(np.abs(draws[:, i]) >= abs(b[j]))) / (DRAWS + 1)
    est_own = f.beta(own)[j] if own is not None else np.nan
    rows.append(dict(outcome=name, estimand="tau (x above)" if j == 2 else "c (level jump at the boundary)",
                     estimate=b[j], estimate_own_price=est_own, baseline_b=b[0], p_randomization=p,
                     draw_sd=draws[:, i].std(), t_clustered=b[j] / se[j], mean_y=np.average(f.y, weights=f.w)))
R = pd.DataFrame(rows)
order = np.argsort(R["p_randomization"].to_numpy())             # Holm: sorted p times (m - rank), kept monotone
holm, run = np.empty(len(R)), 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (len(R) - rank) * R["p_randomization"].iat[i]))
    holm[i] = run
R["p_holm"] = holm
R.to_csv(os.path.join(OUT, "secondary_result.csv"), index=False)

print("\n3. SECONDARY RESULTS (randomization p decides; Holm across the 3)")
for r in R.itertuples():
    print(f"\n   {r.outcome}  [{r.estimand}]")
    print(f"      estimate {r.estimate:+.4f}" + (f"  (quote's own price as 'above': {r.estimate_own_price:+.4f})"
                                                 if np.isfinite(r.estimate_own_price) else ""))
    print(f"      baseline b (below) {r.baseline_b:+.4f} | mean of y {r.mean_y:.4f}")
    print(f"      randomization p {r.p_randomization:.4f} -> Holm p {r.p_holm:.4f} | draw sd {r.draw_sd:.4f} | "
          f"clustered t {r.t_clustered:+.2f}")
print(f"\nSaved results to {OUT} ({time.time() - t0:,.0f} seconds)")
