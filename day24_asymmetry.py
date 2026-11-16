"""Asymmetry test (prereg-v5): queue priority vs GLFT."""
import glob
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PART1 = os.environ.get("P3_PART1", r"E:\p3\part1")
OUT = os.environ.get("P3_RESULTS", r"E:\p3\results")
COINS = [c.upper() for c in sys.argv[1:]] or ["SOL", "XRP", "BTC", "ETH", "HYPE"]
SUFFIX = "" if not sys.argv[1:] else "_replication"
MIN_QUOTES = 1000
t00 = time.time()


def log(s):
    print(s, file=sys.stderr, flush=True)


files = [(c, f) for c in COINS for f in sorted(glob.glob(os.path.join(PART1, c, "*.parquet")))
         if pq.ParquetFile(f).metadata.num_rows > 0]
if not files:
    sys.exit(f"STOP: no Part 1 files for {COINS} in {PART1}")

# pass 0: same makers and coin median tick as day21
cnt, tick_med = defaultdict(int), defaultdict(list)
for c, f in files:
    d = pd.read_parquet(f, columns=["wallet", "tick_bp"])
    for w, n in d["wallet"].value_counts().items():
        cnt[(c, w)] += int(n)
    tick_med[c].append(float(d["tick_bp"].median()))
TICK = {c: float(np.median(x)) for c, x in tick_med.items()}
keep = {k for k, n in cnt.items() if n >= MIN_QUOTES}
log(f"pass 0 done: {len(keep)} makers | {time.time() - t00:,.0f}s")


def demean(x, g):
    G = g.max() + 1
    n = np.maximum(np.bincount(g, minlength=G), 1).astype(float)
    return x - np.column_stack([np.bincount(g, x[:, j], minlength=G) / n for j in range(x.shape[1])])[g] \
        if x.ndim > 1 else x - (np.bincount(g, x, minlength=G) / n)[g]


stats = defaultdict(dict)        # outcome -> (coin, wallet, day) -> (XtX, Xty, n)
for i, (c, f) in enumerate(files):
    d = pd.read_parquet(f)
    d = d[d["wallet"].isin({w for cc, w in keep if cc == c})]
    d = d[np.isfinite(d[["q", "vol5", "price_bp", "tick_bp", "log_size", "at_best"]].to_numpy(float)).all(axis=1)]
    if d.empty:
        continue
    if d["day"].astype(str).str[:10].nunique() != 1:
        sys.exit(f"STOP: {f} holds more than one day")
    day = str(d["day"].iat[0])[:10]
    s = np.where(d["bid"].to_numpy() == 1, -1.0, 1.0)
    u = d["q"].to_numpy(float) * s
    aggr = -s * d["price_bp"].to_numpy(float) / TICK[c]           # bid: price - mid; ask: mid - price
    d = d.assign(up=np.maximum(u, 0), dn=np.minimum(u, 0), aggr=aggr, best=d["at_best"].astype(float))
    for w, m in d.groupby("wallet", sort=False):
        g = (m["bid"].to_numpy() > 0).astype(int)
        X = np.column_stack([m["up"].to_numpy(), m["dn"].to_numpy(), m["vol5"].to_numpy(float)])
        Xd = demean(X, g)
        for outcome in ("aggr", "best"):
            yd = demean(m[outcome].to_numpy(float), g)
            stats[outcome][(c, w, day)] = (Xd.T @ Xd, Xd.T @ yd, len(yd))
    if (i + 1) % 25 == 0:
        log(f"   pass 1: {i + 1}/{len(files)} files | {time.time() - t00:,.0f}s")
log(f"pass 1 done | {time.time() - t00:,.0f}s")


def pooled(outcome, keep_fn=lambda c, w, day: True):
    S = {k: v for k, v in stats[outcome].items() if keep_fn(*k)}
    if not S:
        return None
    A, b = np.zeros((3, 3)), np.zeros(3)
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
    sc = defaultdict(lambda: np.zeros(3))
    for (c, w, day), (xx, xy, n) in S.items():
        sc[(c, day)] += W[(c, w)] * (xy - xx @ beta)
    G = len(sc)
    V = Ai @ sum(np.outer(v, v) for v in sc.values()) @ Ai * G / max(G - 1, 1)
    return beta, V, G, len(n_in), sum(v[2] for v in S.values())


def pval(t):
    from math import erf, sqrt
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


rows = []


def report(lab, outcome, fn=lambda c, w, day: True, main=False):
    r = pooled(outcome, fn)
    if r is None:
        print(f"   {lab:<24} (no data)")
        return None
    beta, V, G, nm, nq = r
    se = np.sqrt(np.diag(V))
    diff = beta[0] - beta[1]
    sed = np.sqrt(V[0, 0] + V[1, 1] - 2 * V[0, 1])
    p = pval(diff / sed)
    k = 100 if outcome == "best" else 1                          # at_best: percentage points of P(at best)
    print(f"   {lab:<24} k_plus {k * beta[0]:+.4f} (se {k * se[0]:.4f}) | k_minus {k * beta[1]:+.4f} (se {k * se[1]:.4f}) | "
          f"difference {k * diff:+.4f} [{k * (diff - 1.96 * sed):+.4f}, {k * (diff + 1.96 * sed):+.4f}] p {p:.4f} "
          f"| {nm} makers, {nq:,} quotes, {G} coin-days")
    rows.append(dict(outcome=outcome, sample=lab, k_plus=beta[0], se_plus=se[0], k_minus=beta[1], se_minus=se[1],
                     difference=diff, se_difference=sed, p=p, makers=nm, coin_days=G, quotes=nq))
    return diff, p


print(f"ASYMMETRY TEST (prereg-v5) - coins {', '.join(COINS)}"
      + ("  [CONFIRMATORY]" if not SUFFIX else "  [descriptive replication]"))
print("   k_plus  = gain in aggressiveness per unit of u on the UNWINDING side (u = q x s > 0)")
print("   k_minus = the same on the ACCUMULATING side (k_minus > 0: that quote moves away from the mid)")
print("   GLFT: k_plus = k_minus.  Queue priority: k_plus > k_minus (accumulating side pulled, not shaded).\n")
print("MAIN: aggressiveness in TICKS per unit of u (signed slopes; difference = k_plus - k_minus)")
print("   sample check: day22 'all' row had 230 makers, 165,257,159 quotes, 386 coin-days (5 coins)")
res = report("all coins pooled", "aggr", main=True)
if not SUFFIX and rows:
    r0 = rows[-1]
    same = (r0["makers"], r0["quotes"], r0["coin_days"]) == (230, 165_257_159, 386)
    print(f"   SAMPLE CHECK vs day22 'all': {'IDENTICAL' if same else 'DIFFERENT - stop and send this output before reading H5'}"
          f" ({r0['makers']} makers, {r0['quotes']:,} quotes, {r0['coin_days']} coin-days)")
if res and not SUFFIX:
    diff, p = res
    print(f"\n   H5 VERDICT (k_plus - k_minus > 0, p < .05): "
          f"{'SUPPORTED - asymmetric, fits queue priority, not GLFT' if diff > 0 and p < 0.05 else 'NOT SUPPORTED'}")
print("\n   each coin:")
for c in COINS:
    report(f"coin {c}", "aggr", (lambda cc: lambda c, w, d: c == cc)(c))
big = [c for c in ("BTC", "HYPE") if c in COINS]
if big:
    print("\n   truncation guard (5 bp band = ~40 ticks, so dropped far quotes cannot flatten k_minus):")
    report("BTC + HYPE", "aggr", lambda c, w, d: c in big)
print("\nSECONDARY: at_best - PERCENTAGE POINTS of P(quote at or better than the best) per unit of u, not ticks")
report("all coins pooled", "best")
pd.DataFrame(rows).to_csv(os.path.join(OUT, f"part1_asymmetry{SUFFIX}.csv"), index=False)
print(f"\nSaved part1_asymmetry{SUFFIX}.csv in {OUT} | {(time.time() - t00) / 60:.0f} min")
