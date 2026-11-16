"""Main tick test (prereg-v1): does a 10x tick change the side response?"""
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


# ---------------------------------------------------------------- 1
ep = pd.read_csv(EPISODES, parse_dates=["start", "end"])
ep = ep[ep["coin"].isin(LEVEL) & (ep["level"] == ep["coin"].map(LEVEL)) & ep["good"].astype(bool)]
ep = ep[ep["merge_min"] == 60].sort_values(["coin", "start"]).reset_index(drop=True)
print("MAIN TEST (pre-registration prereg-v1, sections 2, 4, 5)")
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


# ---------------------------------------------------------------- 2
parts, ep_rows, lab_all, lab_off = [], [], [], []
drop = dict(no_q=0, no_vol=0, outside_window=0, straddle=0)
t0 = time.time()
for e in ep.itertuples():
    coin, L = e.coin, LEVEL[e.coin]
    W0 = (e.start - pd.Timedelta(minutes=30)).value // MIN_NS          # first minute of the window
    W1 = (e.end + pd.Timedelta(minutes=30)).value // MIN_NS            # last minute (inclusive)
    M = int(W1 - W0 + 1)
    days = near_days(pd.Timestamp(W0 * MIN_NS, tz="UTC"), pd.Timestamp(W1 * MIN_NS, tz="UTC"))

    # minute labels from the Hyperliquid book
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
    # a minute without a snapshot keeps the LAST KNOWN label
    mids = mids.ffill().loc[W0:W1]
    if mids.isna().any():
        sys.exit(f"STOP: episode {coin} {e.day}: no book mid in the hour before the window start")
    lab = (mids.to_numpy() >= L).astype(np.int8)
    n_up, n_dn = int(lab.sum()), int(M - lab.sum())
    rotate = n_up >= MIN_MIN and n_dn >= MIN_MIN

    # quotes: window + 5 min before
    qf = files_for(QUOTES, coin, days)
    if not qf:
        sys.exit(f"STOP: no quote files for {coin} on {days} in {QUOTES}")
    cols = set(con.execute(f"DESCRIBE SELECT * FROM read_parquet({qf}, union_by_name=true)").df()["column_name"])
    good = "book_quality = 'GOOD'" if "book_quality" in cols else "hour_crossed IS NOT NULL AND NOT poor_book"
    q = con.execute(f"""
        SELECT lower(wallet) AS wallet, ts, side, mid, spread_bp, inventory, best_bid, best_ask,
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
                    "above_own", "mid", "inventory"]])
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
EP.to_csv(os.path.join(OUT, "main_episodes.csv"), index=False)

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


# ---------------------------------------------------------------- 3
class Fit:
    """Weighted LS of bid on [q, a, q*a, v, v*a] with maker x episode FE; only a (the label) changes per draw."""

    def __init__(self, idx):
        d = D.iloc[idx]
        self.idx = idx
        self.y = d["bid"].to_numpy(float)
        self.q = d["q"].to_numpy(float)
        self.v = d["vol5"].to_numpy(float)
        mk = pd.factorize(d["maker"])[0]
        self.w = 1.0 / np.bincount(mk)[mk]                       # every maker counts equally (decision 19)
        g = pd.factorize(d["maker"] + "#" + d["episode"].astype(str))[0]
        self.g, self.sw = g, np.bincount(g, self.w)
        self.cl = pd.factorize(d["coin"] + d["day"])[0]                # cluster = coin-day (14), as in the pilot
        self.ep = d["episode"].to_numpy(np.int64)
        self.mi = d["mi"].to_numpy(np.int64)
        self.yd, self.qd, self.vd = self.dm(self.y), self.dm(self.q), self.dm(self.v)

    def dm(self, x):
        return x - (np.bincount(self.g, self.w * x) / self.sw)[self.g]

    def X(self, a):
        return np.column_stack([self.qd, self.dm(a), self.dm(self.q * a), self.vd, self.dm(self.v * a)])

    def beta(self, a):
        X = self.X(a)
        Xw = X * self.w[:, None]
        return np.linalg.solve(Xw.T @ X, Xw.T @ self.yd)

    def labels(self, k):
        """Label of each quote after rotating episode e by k[e]: minute m gets the label of (m + k) mod M."""
        return LAB[OFF[self.ep] + (self.mi + k[self.ep]) % MM[self.ep]].astype(float)

    def clustered(self, a):
        X = self.X(a)
        Xw = X * self.w[:, None]
        Ai = np.linalg.inv(Xw.T @ X)
        bta = Ai @ (Xw.T @ self.yd)
        e = self.yd - X @ bta
        G = self.cl.max() + 1
        S = np.column_stack([np.bincount(self.cl, self.w * e * X[:, j], minlength=G) for j in range(5)])
        V = Ai @ (S.T @ S) @ Ai * G / (G - 1)
        return bta, np.sqrt(np.diag(V)), G

    def wild(self, a, B, rng):
        """Wild cluster bootstrap-t for tau, null imposed (tau = 0), Webb 6-point weights."""
        X = self.X(a)
        Xr = np.delete(X, 2, axis=1)
        Xrw = Xr * self.w[:, None]
        br = np.linalg.solve(Xrw.T @ Xr, Xrw.T @ self.yd)
        fit_r, er = Xr @ br, self.yd - Xr @ br
        _, se, _ = self.clustered(a)
        t_obs = self.clustered(a)[0][2] / se[2]
        G = self.cl.max() + 1
        webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
        Xw = X * self.w[:, None]
        Ai = np.linalg.inv(Xw.T @ X)
        hits = 0
        for _ in range(B):
            ys = fit_r + er * webb[rng.integers(0, 6, G)][self.cl]
            bs = Ai @ (Xw.T @ ys)
            es = ys - X @ bs
            S = np.column_stack([np.bincount(self.cl, self.w * es * X[:, j], minlength=G) for j in range(5)])
            ses = np.sqrt((Ai @ (S.T @ S) @ Ai)[2, 2] * G / (G - 1))
            hits += abs(bs[2] / ses) >= abs(t_obs)
        return (1 + hits) / (B + 1)


# self-check: planted effect recovered by the same estimator
rs = np.random.default_rng(1)
n = 100_000
Dk = D
D = pd.DataFrame(dict(bid=0, q=rs.normal(size=n), vol5=rs.uniform(0.5, 2, n), maker=rs.integers(0, 300, n).astype(str),
                      episode=rs.integers(0, 5, n), day=rs.integers(0, 14, n).astype(str), coin="X", mi=0))
a_s = rs.integers(0, 2, n).astype(float)
D["bid"] = -0.10 * D["q"] + 0.03 * a_s - 0.05 * D["q"] * a_s + 0.02 * D["vol5"] + rs.normal(size=n)
bb = Fit(np.arange(n)).beta(a_s)
# check 2: the absorbed fixed effects equal explicit dummy-variable WLS (small sample, unequal weights)
sm = np.arange(3000)
fs = Fit(sm)
ds = D.iloc[sm]
dummies = pd.get_dummies(ds["maker"] + "#" + ds["episode"].astype(str)).to_numpy(float)
Xf = np.column_stack([ds["q"], a_s[sm], ds["q"] * a_s[sm], ds["vol5"], ds["vol5"] * a_s[sm], dummies])
sw_ = np.sqrt(fs.w)
bd = np.linalg.lstsq(Xf * sw_[:, None], ds["bid"].to_numpy() * sw_, rcond=None)[0][:5]
fe_err = float(np.max(np.abs(bd - fs.beta(a_s[sm]))))
D = Dk
print(f"\n   self-check: planted tau -0.050, b -0.100 -> recovered tau {bb[2]:+.3f}, b {bb[0]:+.3f} | "
      f"absorbed FE vs explicit dummy WLS: max difference {fe_err:.1e}")
if abs(bb[2] + 0.05) > 0.015 or abs(bb[0] + 0.10) > 0.015 or fe_err > 1e-8:
    sys.exit("STOP: estimator self-check failed")

ALL = Fit(np.arange(len(D)))
SUB = {c: Fit(np.flatnonzero(D["coin"].to_numpy() == c)) for c in LEVEL}
k0 = np.zeros(len(EP), np.int64)
a_obs = {"POOLED": ALL.labels(k0), **{c: f.labels(k0) for c, f in SUB.items()}}
fits = {"POOLED": ALL, **SUB}
obs = {name: fits[name].beta(a_obs[name]) for name in fits}

# ---------------------------------------------------------------- 4
rng = np.random.default_rng(SEED)
draws = np.empty((DRAWS, 3))
t1 = time.time()
for r in range(DRAWS):
    k = np.where(ROT, rng.integers(1, np.maximum(MM, 2)), 0)      # uniform on 1..M-1; fixed episodes keep 0
    if np.any(k[ROT] < 1) or np.any(k[ROT] > MM[ROT] - 1) or np.any(k[~ROT] != 0):
        sys.exit("STOP: a rotation offset is outside 1..M-1 or a fixed episode moved")
    draws[r] = [fits[nm].beta(fits[nm].labels(k))[2] for nm in ("POOLED", "SOL", "XRP")]
    if (r + 1) % 200 == 0:
        log(f"   draw {r + 1}/{DRAWS} | {time.time() - t1:,.0f}s")
pd.DataFrame(draws, columns=["tau_pooled", "tau_SOL", "tau_XRP"]).to_csv(os.path.join(OUT, "main_draws.csv"),
                                                                         index=False)

june = None
if os.path.exists(POWER):
    pc = pd.read_csv(POWER)
    june = pc[pc["outcome"].astype(str).str.startswith("C")]["tau_hat"].to_numpy()

rows = []
wrng = np.random.default_rng(SEED + 1)
print("\n3. RESULT  (tau < 0: makers lean more on quote side to unwind inventory when the tick is 10x coarser)")
for j, name in enumerate(("POOLED", "SOL", "XRP")):
    f, a = fits[name], a_obs[name]
    bt, se, G = f.clustered(a)
    tau = bt[2]
    p_ri = (1 + np.sum(np.abs(draws[:, j]) >= abs(tau))) / (DRAWS + 1)
    p_wild = f.wild(a, WILD, wrng) if G >= 2 else np.nan
    p_june = float(np.mean(np.abs(june) >= abs(tau))) if june is not None and len(june) else np.nan
    tau_own = f.beta(D.iloc[f.idx]["above_own"].to_numpy(float))[2]
    base = bt[0]
    rows.append(dict(estimate="SOL+XRP pooled (CONFIRMATORY)" if name == "POOLED" else f"{name} alone (component)",
                     tau=tau, tau_own_mid=tau_own, b_baseline=base, tau_share_of_b=tau / base if base else np.nan,
                     c_above=bt[1], p_randomization=p_ri, draw_sd=draws[:, j].std(),
                     se_clustered=se[2], t_clustered=tau / se[2], clusters=G, p_wild_cluster=p_wild,
                     p_june_null=p_june, quotes=len(f.y), makers=D.iloc[f.idx]["maker"].nunique()))
    print(f"\n   {rows[-1]['estimate']}")
    print(f"      tau (q x above)         {tau:+.4f}   (quote's own mid: {tau_own:+.4f})")
    print(f"      baseline b (below)      {base:+.4f}   -> above the boundary the side response is "
          f"{(base + tau) / base:.0%} of the below response" if base else "")
    print(f"      RANDOMIZATION p         {p_ri:.4f}   ({DRAWS:,} circular shifts; draw sd {draws[:, j].std():.4f})"
          f"{'   <- DECIDES' if name == 'POOLED' else ''}")
    print(f"      secondary: clustered t {tau / se[2]:+.2f} ({G} coin-days) | wild cluster p {p_wild:.4f} | "
          f"June-null share {p_june:.3f}")
    print(f"      quotes {len(f.y):,} | makers {rows[-1]['makers']}")
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "main_result.csv"), index=False)
D.to_parquet(os.path.join(OUT, "main_sample.parquet"), index=False)

pool = R.iloc[0]
print("\n4. READING (section 9 of the pre-registration)")
if pool["p_randomization"] < 0.05:
    print(f"   tau is significant (p = {pool['p_randomization']:.4f}). This is NOT yet a tick claim: it becomes one only")
    print("   if the 8 placebo levels and the PENGU clean placebo pass (next scripts).")
else:
    print(f"   tau is not significant (p = {pool['p_randomization']:.4f}). Reported with its uncertainty; the June pilot")
    print("   suggested effects of about 14% of the baseline would usually be seen - not a formal proof of no effect.")
print(f"\nSaved results to {OUT} ({time.time() - t0:,.0f} seconds)")
