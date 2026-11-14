"""Day 25 (Nov 14): IS THE DEFENCE DRIVEN BY TOXIC FILLS? Pre-registration addendum v6 (tag prereg-v6).

Run:  python -W ignore E:\\p3\\day25_toxicity.py > E:\\p3\\results\\day25_output.txt               (5 coins, confirmatory)
      python -W ignore E:\\p3\\day25_toxicity.py AAVE DOGE ... > E:\\p3\\results\\day25_replication.txt  (descriptive)
About 20-40 minutes. Window: both cut points are searched on the sorted fill times
(fill time >= tau - 300 s, and fill time <= tau - (H + 1) s so the markout is complete). Saves E:\\p3\\results\\part1_toxicity.csv (or _replication).

Toxicity T of a quote: the maker's own fills in the last 300 s whose markout is already complete (fill second
t + H + 1 s <= quote time); markout = d x (log mid(t+H) - log mid(t-1)) x 1e4 (d = +1 maker bought);
T = -(notional-weighted mean markout) / (coin-day median vol5 x sqrt(H)), capped +-5. T > 0 = toxic.
Side model: bid = b q + c q*T + e T + d vol5 + f r10 + f r60 + f r300 + h q*vz + FE(maker x day). H6: c < 0.
"""
import glob
import os
import sys
import time
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PART1 = os.environ.get("P3_PART1", r"E:\p3\part1")
QUOTES = os.environ.get("P3_QUOTES", r"E:\p3\data\quotes")
INV = os.environ.get("P3_INV", r"E:\p3\data\inventory")
OUT = os.environ.get("P3_RESULTS", r"E:\p3\results")
COINS = [c.upper() for c in sys.argv[1:]] or ["SOL", "XRP", "BTC", "ETH", "HYPE"]
SUFFIX = "" if not sys.argv[1:] else "_replication"
MIN_QUOTES, WIN_S, CAP, STALE_S = 1000, 300, 5.0, 5     # STALE_S: frozen 5 s staleness rule (vol5)
HS = (10, 5, 30)                      # main horizon first
NS = 1_000_000_000
t00 = time.time()
con = duckdb.connect()


def log(s):
    print(s, file=sys.stderr, flush=True)


def prev_day(d):
    return (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


files = [(c, f) for c in COINS for f in sorted(glob.glob(os.path.join(PART1, c, "*.parquet")))
         if pq.ParquetFile(f).metadata.num_rows > 0]
if not files:
    sys.exit(f"STOP: no Part 1 files for {COINS} in {PART1}")

# pass 0: same makers as day21, vol5 moments for vz
cnt, tick_med = defaultdict(int), defaultdict(list)
vs = defaultdict(lambda: [0, 0.0, 0.0])
for c, f in files:
    d = pd.read_parquet(f, columns=["wallet", "tick_bp", "vol5"])
    for w, n in d["wallet"].value_counts().items():
        cnt[(c, w)] += int(n)
    tick_med[c].append(float(d["tick_bp"].median()))
    fv = d["vol5"].to_numpy(float)
    fv = fv[np.isfinite(fv)]
    vs[c][0] += len(fv); vs[c][1] += float(fv.sum()); vs[c][2] += float((fv ** 2).sum())
TICK = {c: float(np.median(x)) for c, x in tick_med.items()}
VMEAN = {c: v[1] / v[0] for c, v in vs.items() if v[0]}
VSD = {c: np.sqrt(max(v[2] / v[0] - (v[1] / v[0]) ** 2, 1e-12)) for c, v in vs.items() if v[0]}
keep = {k for k, n in cnt.items() if n >= MIN_QUOTES}
log(f"pass 0 done: {len(keep)} makers | {time.time() - t00:,.0f}s")


def log_mid(coin, d):
    """Per-second log mid (forward-filled) from the quote tables, from 15 min before the day."""
    qf = sorted(glob.glob(os.path.join(QUOTES, coin, f"{prev_day(d)}*.parquet"))) + \
         sorted(glob.glob(os.path.join(QUOTES, coin, f"{d}*.parquet")))
    D0 = pd.Timestamp(d, tz="UTC").value // NS
    ms = con.execute(f"""SELECT ts // {NS} AS sec, arg_max(mid, ts) AS mid FROM read_parquet({qf}, union_by_name=true)
                         WHERE mid > 0 AND isfinite(mid) AND ts >= {(D0 - 900) * NS} GROUP BY 1 ORDER BY 1""").df()
    if ms.empty:
        return None
    full = np.arange(ms["sec"].min(), ms["sec"].max() + 1)
    lm = pd.Series(np.log(ms["mid"].to_numpy(float)), index=ms["sec"].to_numpy()).reindex(full)
    seen = pd.Series(np.where(lm.notna(), full, np.nan), index=full).ffill()
    lm = lm.ffill()
    return lm.where(full - seen.to_numpy() <= STALE_S)          # a mid older than 5 s = unknown (NaN), never filled


def fills(coin, d, wallets):
    fs = [f for f in (os.path.join(INV, coin, f"{x}.parquet") for x in (prev_day(d), d)) if os.path.exists(f)]
    if not fs:
        return None
    tcol = con.execute(f"DESCRIBE SELECT ts FROM read_parquet({fs}, union_by_name=true)").fetchone()[1]
    T = "epoch_ns(ts)" if "TIMESTAMP" in tcol.upper() else "ts"
    D0 = pd.Timestamp(d, tz="UTC").value // NS
    f = con.execute(f"""SELECT lower(wallet) AS wallet, CAST({T} AS BIGINT) AS ts, pos_after - pos_before AS dpos,
                               price * size AS notional
                        FROM read_parquet({fs}, union_by_name=true)
                        WHERE {T} >= {(D0 - 2 * WIN_S) * NS}""").df()
    f = f[f["wallet"].isin(wallets) & (f["dpos"] != 0) & (f["notional"] > 0)]
    return f.sort_values("ts")


def demean(x, g):
    G = g.max() + 1
    n = np.maximum(np.bincount(g, minlength=G), 1).astype(float)
    return x - np.column_stack([np.bincount(g, x[:, j], minlength=G) / n for j in range(x.shape[1])])[g] \
        if x.ndim > 1 else x - (np.bincount(g, x, minlength=G) / n)[g]


stats = defaultdict(dict)
cover = defaultdict(lambda: [0, 0, 0])   # coin: quotes, quotes with toxicity (H = 10), quotes in the H6 regression
nofill_days = 0
for i, (c, f) in enumerate(files):
    d = pd.read_parquet(f)
    d = d[d["wallet"].isin({w for cc, w in keep if cc == c})]
    d = d[np.isfinite(d[["q", "vol5", "price_bp", "tick_bp", "log_size"]].to_numpy(float)).all(axis=1)]
    if d.empty:
        continue
    if d["day"].astype(str).str[:10].nunique() != 1:
        sys.exit(f"STOP: {f} holds more than one day")
    day = str(d["day"].iat[0])[:10]
    lm = log_mid(c, day)
    fl = fills(c, day, set(d["wallet"]))
    if lm is None or fl is None or fl.empty:
        nofill_days += 1
        continue
    d = d.sort_values("ts").reset_index(drop=True)
    sec = d["ts"].to_numpy("int64") // NS
    get = lambda s: lm.reindex(s).to_numpy(float)
    for k_, h in (("r10", 10), ("r60", 60), ("r300", 300)):
        d[k_] = (get(sec - 1) - get(sec - 1 - h)) * 1e4
    sig_cd = float(np.nanmedian(d["vol5"]))
    if not (np.isfinite(sig_cd) and sig_cd > 0):
        nofill_days += 1
        continue
    fsec = fl["ts"].to_numpy("int64") // NS
    dirn = np.sign(fl["dpos"].to_numpy(float))
    notl = fl["notional"].to_numpy(float)
    fw = fl["wallet"].to_numpy()
    by_w = pd.Series(np.arange(len(fw))).groupby(fw).apply(lambda x: x.to_numpy()).to_dict()
    for H in HS:
        mk = dirn * (get(fsec + H) - get(fsec - 1)) * 1e4
        ok_f = np.isfinite(mk)
        T = np.full(len(d), np.nan)
        for w, ix in d.groupby("wallet", sort=False).indices.items():
            if w not in by_w:
                continue
            sel = by_w[w][ok_f[by_w[w]]]                                   # this maker's fills, time order
            if not len(sel):
                continue
            ft = fsec[sel] * NS                                            # fill times, sorted
            cn = np.r_[0.0, np.cumsum(notl[sel])]
            cm = np.r_[0.0, np.cumsum(notl[sel] * mk[sel])]
            tq = d["ts"].to_numpy("int64")[ix]
            hi = np.searchsorted(ft, tq - (H + 1) * NS, side="right")     # markout complete by tau
            lo = np.searchsorted(ft, tq - WIN_S * NS, side="left")
            n_ = np.where(hi > lo, cn[hi] - cn[lo], 0.0)
            m_ = np.where(hi > lo, cm[hi] - cm[lo], 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                T[ix] = np.where(n_ > 0, -(m_ / n_) / (sig_cd * np.sqrt(H)), np.nan)
        d[f"T{H}"] = np.clip(T, -CAP, CAP)
    cover[c][0] += len(d)
    cover[c][1] += int(np.isfinite(d["T10"]).sum())
    s_ = np.where(d["bid"].to_numpy() == 1, -1.0, 1.0)
    d["vz"] = (d["vol5"] - VMEAN[c]) / VSD[c]
    d["u"] = d["q"] * s_
    d["aggr"] = -s_ * d["price_bp"] / TICK[c]
    for w, m in d.groupby("wallet", sort=False):
        q, v, vz = m["q"].to_numpy(float), m["vol5"].to_numpy(float), m["vz"].to_numpy(float)
        R = m[["r10", "r60", "r300"]].to_numpy(float)
        g_side = (m["bid"].to_numpy() > 0).astype(int)
        for H in HS:
            T = m[f"T{H}"].to_numpy(float)
            ok = np.isfinite(T) & np.isfinite(R).all(axis=1)
            if ok.sum() < 3:
                continue
            if H == HS[0]:
                cover[c][2] += int(ok.sum())
            X = np.column_stack([q, q * T, T, v, R, q * vz])[ok]
            y = m["bid"].to_numpy(float)[ok]
            z = np.zeros(ok.sum(), int)
            Xd, yd = demean(X, z), demean(y, z)
            stats[("side", H)][(c, w, day)] = (Xd.T @ Xd, Xd.T @ yd, len(yd))
            if H == HS[0]:
                u = m["u"].to_numpy(float)
                up, dn = np.maximum(u, 0), np.minimum(u, 0)
                X2 = np.column_stack([up, dn, up * T, dn * T, T, v, R])[ok]
                y2 = m["aggr"].to_numpy(float)[ok]
                g2 = g_side[ok]
                X2d, y2d = demean(X2, g2), demean(y2, g2)
                stats[("aggr", H)][(c, w, day)] = (X2d.T @ X2d, X2d.T @ y2d, len(y2d))
    if (i + 1) % 25 == 0:
        log(f"   pass 1: {i + 1}/{len(files)} files | {time.time() - t00:,.0f}s")
log(f"pass 1 done | {time.time() - t00:,.0f}s")


def pooled(key, keep_fn=lambda c, w, day: True, cluster="coin_day"):
    S = {k: v for k, v in stats[key].items() if keep_fn(*k)}
    if not S:
        return None
    K = next(iter(S.values()))[0].shape[0]
    A, b = np.zeros((K, K)), np.zeros(K)
    n_in = defaultdict(int)
    for k, v in S.items():
        n_in[k[:2]] += v[2]
    W = {mk: 1.0 / n for mk, n in n_in.items()}
    for (c, w, day), (xx, xy, n) in S.items():
        A += W[(c, w)] * xx
        b += W[(c, w)] * xy
    try:
        beta, Ai = np.linalg.solve(A, b), np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None
    sc = defaultdict(lambda: np.zeros(K))
    for (c, w, day), (xx, xy, n) in S.items():
        sc[{"coin_day": (c, day), "maker": (c, w), "day": day}[cluster]] += W[(c, w)] * (xy - xx @ beta)
    G = len(sc)
    V = Ai @ sum(np.outer(v, v) for v in sc.values()) @ Ai * G / max(G - 1, 1)
    return beta, np.sqrt(np.diag(V)), G, len(n_in), sum(v[2] for v in S.values())


def pval(t):
    from math import erf, sqrt
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


rows = []


def line(lab, r, j):
    beta, se, G, nm, nq = r
    return (f"   {lab:<34} {beta[j]:+.5f} (se {se[j]:.5f}, 95% [{beta[j] - 1.96 * se[j]:+.5f}, "
            f"{beta[j] + 1.96 * se[j]:+.5f}], p {pval(beta[j] / se[j]):.4f})")


print(f"TOXICITY TEST (prereg-v6) - coins {', '.join(COINS)}"
      + ("  [CONFIRMATORY]" if not SUFFIX else "  [descriptive replication]"))
print("   T > 0 = the maker's fills in the last 5 min were followed by a price move against it (toxic)")
print("   share of the T1 sample - with toxicity (H = 10 s) / entering the H6 regression (all returns known): "
      + ", ".join(f"{c} {v[1] / v[0]:.0%} / {v[2] / v[0]:.0%}" for c, v in cover.items() if v[0])
      + f" | coin-days skipped (no fills or prices): {nofill_days}")
for H in HS:
    r = pooled(("side", H))
    if r is None:
        print(f"\nH = {H} s: no data")
        continue
    tag = "MAIN (H = 10 s)" if H == HS[0] else f"robustness H = {H} s"
    print(f"\nSIDE MODEL, {tag} | {r[3]} makers, {r[4]:,} quotes, {r[2]} coin-days")
    print(line("q (side response at T = 0)", r, 0))
    print(line("q x T  <- H6 (predicted < 0)", r, 1))
    print(line("T", r, 2))
    rows.append(dict(model="side", H=H, term="q x T", value=r[0][1], se=r[1][1], p=pval(r[0][1] / r[1][1]),
                     makers=r[3], quotes=r[4], coin_days=r[2]))
    rows.append(dict(model="side", H=H, term="q", value=r[0][0], se=r[1][0], p=pval(r[0][0] / r[1][0]),
                     makers=r[3], quotes=r[4], coin_days=r[2]))
    if H == HS[0] and not SUFFIX:
        c_, p_ = r[0][1], pval(r[0][1] / r[1][1])
        print(f"\n   H6 VERDICT (c < 0, p < .05): "
              f"{'SUPPORTED - stronger defence after toxic fills (adverse selection, not in GLFT)' if c_ < 0 and p_ < 0.05 else 'NOT SUPPORTED'}")
        for cl in ("maker", "day"):
            rr = pooled(("side", H), cluster=cl)
            if rr is not None:
                print(f"   q x T clustered by {cl:<5}: se {rr[1][1]:.5f}, p {pval(rr[0][1] / rr[1][1]):.4f} ({rr[2]} clusters)")
        b_ = r[0][0]
        print(f"   size: at T = 0 the side response is {b_:+.4f} per sd of q; at T = +1 (fills 1 sigma toxic) "
              f"{b_ + c_:+.4f}")
print("\n   each coin (H = 10 s), q x T:")
for c in COINS:
    r = pooled(("side", HS[0]), (lambda cc: lambda c, w, d: c == cc)(c))
    if r is not None:
        print(f"      {c:<9} {r[0][1]:+.5f} (p {pval(r[0][1] / r[1][1]):.3f}) | q {r[0][0]:+.4f} | {r[3]} makers")
        rows.append(dict(model="side by coin", H=HS[0], term=f"{c} q x T", value=r[0][1], se=r[1][1],
                         p=pval(r[0][1] / r[1][1]), makers=r[3], quotes=r[4], coin_days=r[2]))
r = pooled(("aggr", HS[0]))
if r is not None:
    print(f"\nSECONDARY - aggressiveness in ticks (prereg-v5 model + toxicity), H = 10 s")
    for j, lab in ((0, "u+ (unwinding)"), (1, "u- (accumulating)"), (2, "u+ x T"), (3, "u- x T  (predicted > 0)")):
        print(line(lab, r, j))
        rows.append(dict(model="aggr", H=HS[0], term=lab, value=r[0][j], se=r[1][j], p=pval(r[0][j] / r[1][j]),
                         makers=r[3], quotes=r[4], coin_days=r[2]))
os.makedirs(OUT, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, f"part1_toxicity{SUFFIX}.csv"), index=False)
print(f"\nSaved part1_toxicity{SUFFIX}.csv in {OUT} | {(time.time() - t00) / 60:.0f} min")
