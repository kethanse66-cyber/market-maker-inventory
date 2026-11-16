"""Power check on the June pilot (fake boundaries). Writes data/power_check.csv."""
import glob
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

QUOTES = os.environ.get("P3_QUOTES", r"E:\p3\data\quotes")
EPISODES = os.environ.get("P3_EPISODES", r"E:\p3\data\episodes.csv")
OUT = os.environ.get("P3_POWER_OUT", r"E:\p3\data\power_check.csv")
LEVEL = {"SOL": 100.0, "XRP": 1.0}
JUNE_GOOD = ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-06",
             "2026-06-10", "2026-06-11", "2026-06-12", "2026-06-13"]     # Jun 7-9, 14: no book
WINDOW = pd.Timedelta(minutes=30)
R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
MIN_SD_MINUTES = 30                       # need 30+ quoting minutes in the past 24 h to scale inventory
CLIP = 5.0                                # q capped at +-5 std
BAND_BP = 5.0                             # outcomes B and C: within 5 bp of the quote's own best price
SAMPLE = 4                                # 1-in-4 quotes by order id (memory)
OUTCOMES = ["A price 2 ticks", "B price 5 bp", "C side 5 bp"]
NS = 1_000_000_000
rng = np.random.default_rng(20261107)


def log(s):
    print(s, file=sys.stderr, flush=True)


# ---------------- 1
ep = pd.read_csv(EPISODES, parse_dates=["start", "end"])
ep = ep[ep["coin"].isin(LEVEL) & (ep["level"] == ep["coin"].map(LEVEL)) & ep["good"].astype(bool)]
merge = 60 if (ep["merge_min"] == 60).any() else ep["merge_min"].max()
ep = ep[ep["merge_min"] == merge]
design = {}
for coin, g in ep.groupby("coin"):
    design[coin] = [[(r.start - r.start.normalize(), r.end - r.start) for r in d.itertuples()]
                    for _, d in g.groupby("day")]
n_eps = sum(len(d) for v in design.values() for d in v)
n_days = sum(len(v) for v in design.values())
print("REAL DESIGN COPIED ONTO JUNE")
for coin, days in design.items():
    print(f"   {coin}: {sum(map(len, days))} good episodes on {len(days)} days "
          f"(episodes per day: {', '.join(str(len(d)) for d in days)})")
print(f"   total {n_eps} episodes, {n_days} day-clusters (merge {merge} min)")
if n_eps != 31 or n_days != 14:
    print("   NOTE: expected 31 episodes on 14 days - check episodes.csv")

# ---------------- 2
con = duckdb.connect()
data = {}
for coin in design:
    files = sorted(f for d in JUNE_GOOD for f in glob.glob(os.path.join(QUOTES, coin, f"{d}*.parquet")))
    allf = sorted(glob.glob(os.path.join(QUOTES, coin, "2026-06-*.parquet")))
    if not files:
        sys.exit(f"STOP: no June quote files for {coin} in {QUOTES}")
    if len(design[coin]) > len({os.path.basename(f)[:10] for f in files}):
        sys.exit(f"STOP: {coin} needs {len(design[coin])} June days, found fewer")
    t0 = time.time()
    cols = set(con.execute(f"DESCRIBE SELECT * FROM read_parquet({files}, union_by_name=true)").df()["column_name"])
    if "book_quality" in cols:
        good = "book_quality = 'GOOD'"
    elif {"hour_crossed", "poor_book"} <= cols:          # older quote files: same rule, built here
        good = "hour_crossed IS NOT NULL AND NOT poor_book"
    else:
        sys.exit(f"STOP: {coin} quote files have no book-quality columns. Columns are: {sorted(cols)}")
    need = {"oid", "ticks_behind_best", "tick", "in_gap", "crosses_book", "mid", "inventory", "best_bid",
            "best_ask", "spread_bp", "rel_mid_bp", "price", "side", "wallet", "ts"}
    if need - cols:
        sys.exit(f"STOP: {coin} quote files miss {sorted(need - cols)}")
    print(f"   {coin}: good-book rule used: {good}")
    df = con.execute(f"""
        WITH m AS (SELECT wallet, ts // {60 * NS} AS minute, arg_max(inventory, ts) AS inv
                   FROM read_parquet({allf}, union_by_name=true)
                   WHERE inventory IS NOT NULL GROUP BY 1, 2),
             s AS (SELECT wallet, minute,
                          stddev_samp(inv) OVER w AS sd, count(inv) OVER w AS n
                   FROM m WINDOW w AS (PARTITION BY wallet ORDER BY minute
                                       RANGE BETWEEN 1440 PRECEDING AND 1 PRECEDING)),
             q AS (SELECT wallet, side, price, ts, mid, best_bid, best_ask, spread_bp, rel_mid_bp, inventory,
                          round(ticks_behind_best, 6) <= 2 AS near2, ts // {60 * NS} AS minute
                   FROM read_parquet({files}, union_by_name=true)
                   WHERE NOT in_gap AND {good} AND NOT crosses_book AND mid IS NOT NULL AND inventory IS NOT NULL
                         AND ticks_behind_best * tick / mid * 1e4 <= {BAND_BP} AND hash(oid) % {SAMPLE} = 0)
        SELECT (dense_rank() OVER (ORDER BY q.wallet) - 1)::INTEGER AS maker, (q.side = 'B')::TINYINT AS bid,
               q.price, q.ts, q.mid, q.best_bid, q.best_ask, q.spread_bp::FLOAT AS spread_bp,
               q.rel_mid_bp::FLOAT AS rel_mid_bp, q.near2,
               greatest(least(q.inventory / s.sd, {CLIP}), -{CLIP})::FLOAT AS q,
               abs(q.inventory / s.sd) >= {CLIP} AS capped
        FROM q JOIN s USING (wallet, minute)
        WHERE s.n >= {MIN_SD_MINUTES} AND s.sd > 0
        ORDER BY q.ts""").df()
    df["day"] = pd.to_datetime(df["ts"], unit="ns", utc=True).dt.strftime("%Y-%m-%d")
    data[coin] = {d: g.reset_index(drop=True) for d, g in df.groupby("day")}
    log(f"   {coin}: {len(df):,} main-sample June quotes with q, {df['maker'].nunique()} makers, "
        f"{time.time() - t0:,.0f}s")
    print(f"   {coin} June: {len(df):,} sampled quotes on {df['day'].nunique()} days, "
          f"{df['maker'].nunique()} makers | within 2 ticks {df['near2'].mean():.0%} | median |q| "
          f"{df['q'].abs().median():.2f} | capped at 5: {df['capped'].mean():.1%}")


# ---------------- 3
def fit(y, X, g, w, cl):
    """Weighted least squares with group fixed effects g; cluster-robust SE by cl."""
    gi = np.unique(g, return_inverse=True)[1]
    sw = np.bincount(gi, w)
    def dm(v):
        return v - (np.bincount(gi, w * v) / sw)[gi]
    yd = dm(y)
    Xd = np.column_stack([dm(X[:, k]) for k in range(X.shape[1])])
    A = (Xd * w[:, None]).T @ Xd
    info = dict(cond=float(np.linalg.cond(A)),
                tau_left=float(np.average(Xd[:, 2] ** 2, weights=w) /
                               max(np.average((X[:, 2] - np.average(X[:, 2], weights=w)) ** 2, weights=w), 1e-300)))
    Ai = np.linalg.inv(A)
    beta = Ai @ ((Xd * w[:, None]).T @ yd)
    e = yd - Xd @ beta
    ci = np.unique(cl, return_inverse=True)[1]
    G = ci.max() + 1
    S = np.column_stack([np.bincount(ci, w * e * Xd[:, k], minlength=G) for k in range(Xd.shape[1])])
    V = Ai @ (S.T @ S) @ Ai * G / (G - 1)
    return beta, np.sqrt(np.diag(V)), G, info


NAMES = ["q (b)", "above", "q x above (tau)", "spread", "spread x above", "above x bid"]


MIN_Q, MIN_SIDE, TRIES = 50, 10, 200     # per fake episode: 50+ quotes, 10+ on each side of the fake level


FAILS = {}


def cut(Q, ts, s, dur, day0):
    """The quotes of one fake episode starting at s, or None (reason counted) if it is not usable."""
    if s - WINDOW < day0 or s + dur + WINDOW > day0 + pd.Timedelta(days=1):
        FAILS["window outside the day"] = FAILS.get("window outside the day", 0) + 1
        return None
    a = np.searchsorted(ts, (s - WINDOW).value)
    z = np.searchsorted(ts, (s + dur + WINDOW).value)
    k0 = np.searchsorted(ts, s.value)
    if z - a < MIN_Q or k0 >= len(ts):
        FAILS["too few good-book quotes"] = FAILS.get("too few good-book quotes", 0) + 1
        return None
    L = float(Q["mid"].iat[k0])          # fake level = the market mid at the fake episode start
    W = Q.iloc[a:z]
    W = W[~((W["best_bid"] < L) & (W["best_ask"] >= L))]               # straddle excluded
    ab = W["price"].to_numpy() >= L
    am = W["mid"].to_numpy() >= L
    n2 = W["near2"].to_numpy()
    if len(W) < MIN_Q or min(ab.sum(), (~ab).sum(), am.sum(), (~am).sum(),
                             (ab & n2).sum(), (~ab & n2).sum()) < MIN_SIDE:
        FAILS["too few quotes on one side of the level"] = FAILS.get("too few quotes on one side of the level", 0) + 1
        return None
    return W, ab.astype(float), am.astype(float)


def place_day(Q, eps, day0):
    """First choice: one time shift for the whole day's schedule (spacing kept exactly; shifts that push a window out of the day are rejected, never clipped)."""
    ts = Q["ts"].to_numpy()
    for _ in range(TRIES):
        shift = pd.Timedelta(minutes=int(rng.integers(-1440, 1441)))   # one shift moves the whole schedule
        out = []
        for tod, dur in eps:
            c = cut(Q, ts, day0 + tod + shift, dur, day0)
            if c is None:
                break
            out.append(c)
        else:
            return out, False
    for _ in range(TRIES):
        out, taken = [], []
        for tod, dur in eps:
            for _ in range(TRIES):
                s = day0 + WINDOW + pd.Timedelta(minutes=int(rng.integers(0, max(1, 1440 - 60 - dur.total_seconds() // 60))))
                lo, hi = s - WINDOW, s + dur + WINDOW
                if any(lo < h and l < hi for l, h in taken):
                    continue
                c = cut(Q, ts, s, dur, day0)
                if c is not None:
                    out.append(c)
                    taken.append((lo, hi))
                    break
            else:
                break
        else:
            return out, True
    return None


def one_design():
    """A full copy of the real design (every episode and day-cluster present, or the draw is redone); returns the three outcome fits on the same fake episodes."""
    for attempt in range(TRIES):
        parts = []
        e_id, ok, n_fb = 0, True, 0
        for ci, coin in enumerate(design):
            days = [str(x) for x in rng.choice(sorted(data[coin]), size=len(design[coin]), replace=False)]
            for di, (eps, d) in enumerate(zip(design[coin], days)):
                placed = place_day(data[coin][d], eps, pd.Timestamp(d, tz="UTC"))
                if placed is None:
                    ok = False
                    break
                placed, fb = placed
                n_fb += fb
                for W, ab, am in placed:
                    parts.append(dict(px=W["rel_mid_bp"].to_numpy(np.float64), bid=W["bid"].to_numpy(np.float64),
                                      q=W["q"].to_numpy(np.float64), sp=W["spread_bp"].to_numpy(np.float64),
                                      ab=ab, am=am, n2=W["near2"].to_numpy(bool),
                                      mk=W["maker"].to_numpy(np.int64) + 100_000 * ci,
                                      ep=np.full(len(W), e_id), cl=np.full(len(W), ci * 100 + di)))
                    e_id += 1
            if not ok:
                break
        if not ok:
            continue
        D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        if len(np.unique(D["cl"])) != n_days or e_id != n_eps:
            FAILS["episode/cluster count"] = FAILS.get("episode/cluster count", 0) + 1
            continue
        fits, good = {}, True
        for name in OUTCOMES:
            sel = D["n2"] if name.startswith("A") else np.ones(len(D["q"]), bool)
            d = {k: v[sel] for k, v in D.items()}
            mi = np.unique(d["mk"], return_inverse=True)[1]
            w = 1.0 / np.bincount(mi)[mi]        # each maker counts equally across the design (decision 19)
            if name.startswith("C"):
                y, a_ = d["bid"], d["am"]
                X = np.column_stack([d["q"], a_, d["q"] * a_, d["sp"], d["sp"] * a_])
                g = d["ep"] * 10_000_000 + d["mk"]
            else:
                y, a_ = d["px"], d["ab"]
                X = np.column_stack([d["q"], a_, d["q"] * a_, d["sp"], d["sp"] * a_, a_ * d["bid"]])
                g = (d["ep"] * 10_000_000 + d["mk"]) * 2 + d["bid"].astype(np.int64)
            beta, se, G, info = fit(y, X, g, w, d["cl"])
            if G != n_days or info["tau_left"] < 0.05 or info["cond"] > 1e12:
                FAILS[f"{name}: weak q x above variation or unstable fit"] = \
                    FAILS.get(f"{name}: weak q x above variation or unstable fit", 0) + 1
                good = False
                break
            fits[name] = (beta, se, len(y), (y, X, g, w, d["cl"]))
        if good:
            return fits, attempt, n_fb
    sys.exit("STOP: could not build a full copy of the design on June data. Reasons: " +
             "; ".join(f"{k}: {v}" for k, v in sorted(FAILS.items(), key=lambda kv: -kv[1])[:8]))


# self-check of the regression on simulated data with a planted tau
sim_n = 200_000
sg = rng.integers(0, 400, sim_n)
sq = rng.normal(size=sim_n)
sa = rng.integers(0, 2, sim_n).astype(float)
ssp = rng.uniform(0.5, 2, sim_n)
sb = rng.integers(0, 2, sim_n).astype(float)
sy = -1.0 * sq + 0.30 * sq * sa + 0.2 * ssp + rng.normal(size=400)[sg] + rng.normal(size=sim_n)
bb, _, _, _ = fit(sy, np.column_stack([sq, sa, sq * sa, ssp, ssp * sa, sa * sb]), sg,
               np.ones(sim_n), sg % 14)
print(f"\nSELF-CHECK on simulated data: planted b = -1.00, tau = +0.30 -> recovered b = {bb[0]:+.2f}, "
      f"tau = {bb[2]:+.2f}")
if abs(bb[2] - 0.30) > 0.03 or abs(bb[0] + 1) > 0.03:
    sys.exit("STOP: regression self-check failed")

# ---------------- 4. repeat many fake designs ----------------
EFFECT_SHARES = (0.25, 0.5, 0.75, 1.0)
rows, saved = [], []
t0 = time.time()
for r in range(R):
    fits, redraws, n_fb = one_design()
    for name, (beta, se, n, keep) in fits.items():
        rows.append(dict(design=r, outcome=name, tau_hat=beta[2], tau_se=se[2], b_hat=beta[0], b_se=se[0],
                         quotes=n, redraws=redraws, fallback_days=n_fb))
    if r < 10:
        saved.append(fits)
    if (r + 1) % 25 == 0:
        log(f"   {r + 1}/{R} fake designs | {time.time() - t0:,.0f}s")
res = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
res.to_csv(OUT, index=False)

# ---------------- 5. verdict per outcome ----------------
G = n_days
t_crit = stats.t.ppf(0.975, G - 1)
one = res[res["outcome"] == OUTCOMES[0]]
print(f"\nPOWER CHECK - {R} fake designs on June (no tick change anywhere, so the true tau is 0)")
print(f"   every design: exactly {n_eps} episodes and {G} day-clusters | redraws: median "
      f"{one['redraws'].median():.0f}, max {one['redraws'].max()} | days with the exact schedule "
      f"{1 - one['fallback_days'].mean() / n_days:.0%}, random non-overlapping times "
      f"{one['fallback_days'].mean() / n_days:.0%}")
summary = {}
for name in OUTCOMES:
    o = res[res["outcome"] == name].reset_index(drop=True)
    unit = "bid-share points" if name.startswith("C") else "bp"
    b = o["b_hat"].median()
    b_clear = (np.abs(o["b_hat"] / o["b_se"]) > 3).mean()
    t_null = np.abs(o["tau_hat"] / o["tau_se"])
    c_cal = float(np.quantile(t_null, 0.95))
    c_ri = float(np.quantile(np.abs(o["tau_hat"]), 0.95))
    noise = o["tau_hat"].std()

    def power(f, rule):
        th = o["tau_hat"] + f * abs(b)   # exact for OLS (checked by the refits below)
        if rule == "t":
            return (np.abs(th / o["tau_se"]) > t_crit).mean()
        if rule == "cal":
            return (np.abs(th / o["tau_se"]) > c_cal).mean()
        return (np.abs(th) > c_ri).mean()

    def mde(rule):
        for f in np.linspace(0, 3, 3001):
            if power(f, rule) >= 0.80:
                return f
        return np.inf

    err, se_chg = 0.0, 0.0
    for fits in saved:
        beta, se, n, (y, X, g, w, cl) = fits[name]
        for f in EFFECT_SHARES:
            bt, st, _, _ = fit(y + f * abs(b) * X[:, 2], X, g, w, cl)
            err = max(err, abs(bt[2] - beta[2] - f * abs(b)))
            se_chg = max(se_chg, abs(st[2] - se[2]))
    m_cal, m_ri = mde("cal"), mde("ri")
    summary[name] = (b, b_clear, m_ri)
    print(f"\n   {name}  (median {o['quotes'].median():,.0f} quotes per design)")
    print(f"      baseline b per 1 std of inventory: median {b:+.4f} {unit} (middle 90% "
          f"{o['b_hat'].quantile(.05):+.4f} to {o['b_hat'].quantile(.95):+.4f}); |b/SE| > 3 in {b_clear:.0%} of designs")
    print(f"      tau_hat with no effect: mean {o['tau_hat'].mean():+.4f}, noise {noise:.4f} | clustered SE / noise "
          f"{o['tau_se'].median() / noise:.2f} | false alarms, plain t {(t_null > t_crit).mean():.0%} "
          f"| calibrated |t| > {c_cal:.2f}")
    print(f"      refit check ({len(saved)} designs): tau_hat moved by exactly the injected tau, max error {err:.1e}; "
          f"SE change max {se_chg:.1e}")
    print("      coarse tick cuts the response by   plain t   calibrated t   randomization-style")
    for f in EFFECT_SHARES:
        print(f"      {f:>33.0%}   {power(f, 't'):>7.0%}   {power(f, 'cal'):>12.0%}   {power(f, 'ri'):>19.0%}")
    print(f"      smallest effect detected 80% of the time: calibrated t {m_cal:.0%} of |b|, randomization-style "
          f"{m_ri:.0%} of |b| ({abs(b) * m_ri:.4f} {unit})")

ok = {k: v for k, v in summary.items() if v[1] >= 0.5}
print("\nCHOICE OF PRIMARY OUTCOME (rule fixed in the header)")
if not ok:
    print("   no outcome has a clearly non-zero baseline b -> none can carry the tick test; replication coins needed")
else:
    best = min(ok.items(), key=lambda kv: (round(kv[1][2], 2), 0 if kv[0][0] in "AB" else 1))
    for k, (b, bc, m) in summary.items():
        print(f"   {k:<16} b clear in {bc:>4.0%} of designs | detects {m:>5.0%} of |b|"
              f"{'   <- PRIMARY' if k == best[0] else ''}")
    m = best[1][2]
    verdict = ("ENOUGH: the 31 episodes can detect a halving of the inventory response" if m <= 0.5 else
               "BORDERLINE: only a large effect is detectable; the replication coins carry the result"
               if m <= 1.0 else "TOO WEAK for a realistic effect; the replication coins are required")
    print(f"\n   VERDICT: {verdict}")
print("   (A weak result means June cannot see effects of that size - not that the effect is absent.)")
print("   Plain clustered t-tests with 14 clusters are not trusted when false alarms exceed ~5%: significance "
      "comes from randomization inference (the plan's primary method).")
print(f"\nSaved {OUT}  ({R} designs, seed 20261107, {time.time() - t0:,.0f} seconds)")
