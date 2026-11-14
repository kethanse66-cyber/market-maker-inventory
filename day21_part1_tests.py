"""Day 21 (Nov 14): PART 1 TESTS - pre-registration addendum v3 (tag prereg-v3), sections 3-5.

Run (after day20_part1_build.py finished):  python E:\\p3\\day21_part1_tests.py > E:\\p3\\results\\day21_output.txt
About 20-60 minutes (reads every E:\\p3\\part1\\<COIN>\\<day>.parquet twice). Results also in E:\\p3\\results\\part1_*.csv

How it is computed (exact, not approximate): every fixed-effect group (maker x day, or maker x side x day) lies
inside one coin-day file, so each file is demeaned on its own and only small sums (X'X, X'y per maker per file)
are kept. Pooled estimates weight every maker equally within its coin (weight 1 / its number of quotes);
standard errors are clustered by coin-day (each file is one cluster). Per-maker estimates use only that maker's
quotes, clustered by day.

Tests (frozen):
  T1  which tool: side (bid on q), price (offset from mid in TICKS on q), size (log size on q x s)
      "side, not price" if (a) side < 0, p < .05; (b) the 95% interval of the price response lies inside
      (-1, +1) tick per sd of q; (c) among makers with a significant response in either tool, a majority have
      a significant side response and a price response below one tick.
  T2  price and side models with q x (vol5 standardized within coin): GLFT predicts price coefficient < 0.
  T3  side on a spline of q (slopes |q| < 1, 1-3, 3-5): GLFT predicts slope ~0 for |q| < 1.
  T4  Spearman( per-maker side response, inventory half-life from natives.csv ), permutation p (10,000).
Robustness: each coin; hedgers vs natives; q capped at 3 and uncapped; June vs Aug-Oct.
"""
import glob
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

PART1 = os.environ.get("P3_PART1", r"E:\p3\part1")
DATA = os.environ.get("P3_DATA", r"E:\p3\data")
OUT = os.environ.get("P3_RESULTS", r"E:\p3\results")
COINS = ["SOL", "XRP", "BTC", "ETH", "HYPE"]
MIN_QUOTES = 1000
t00 = time.time()


def log(s):
    print(s, file=sys.stderr, flush=True)


import pyarrow.parquet as pq

os.makedirs(OUT, exist_ok=True)
files = [(c, f) for c in COINS for f in sorted(glob.glob(os.path.join(PART1, c, "*.parquet")))
         if pq.ParquetFile(f).metadata.num_rows > 0]              # empty days (no market price / no history) skipped
if not files:
    sys.exit(f"STOP: no Part 1 files in {PART1} - run day20_part1_build.py first")

# ---------------------------------------------------------------- pass 0: maker counts, vol5 moments, tick size
cnt = defaultdict(int)
vs = defaultdict(lambda: [0, 0.0, 0.0])
tick_med = defaultdict(list)
for c, f in files:
    d = pd.read_parquet(f, columns=["wallet", "vol5", "tick_bp"])
    if d.empty:
        continue
    for w, n in d["wallet"].value_counts().items():
        cnt[(c, w)] += int(n)
    v = vs[c]
    v[0] += int(d["vol5"].notna().sum())                           # NaN vol5 rows do not count in the moments
    v[1] += float(d["vol5"].sum())
    v[2] += float((d["vol5"] ** 2).sum())
    tick_med[c].append(float(d["tick_bp"].median()))
VMEAN = {c: v[1] / v[0] for c, v in vs.items()}
VSD = {c: np.sqrt(max(v[2] / v[0] - (v[1] / v[0]) ** 2, 1e-12)) for c, v in vs.items()}
TICK = {c: float(np.median(x)) for c, x in tick_med.items()}
keep = {k for k, n in cnt.items() if n >= MIN_QUOTES}
log(f"pass 0 done: {len(cnt)} maker accounts, {len(keep)} with >= {MIN_QUOTES} quotes | {time.time() - t00:,.0f}s")

# ---------------------------------------------------------------- the model specs
#   name: (outcome, regressors builder, fixed effects within maker: "day" or "side_day")


def spline(q):
    a = np.abs(q)
    s = np.sign(q)
    return np.clip(q, -1, 1), s * np.clip(a - 1, 0, 2), s * np.clip(a - 3, 0, 2)


SPECS = {
    "side":        ("bid",   lambda q, s, v, vz: [q, v],                "day"),
    "price_ticks": ("price", lambda q, s, v, vz: [q, v],                "side_day"),
    "price_own":   ("price_own", lambda q, s, v, vz: [q, v],            "side_day"),
    "size":        ("size",  lambda q, s, v, vz: [q * s, v],            "side_day"),
    "side_T2":     ("bid",   lambda q, s, v, vz: [q, v, q * vz],        "day"),
    "price_T2":    ("price", lambda q, s, v, vz: [q, v, q * vz],        "side_day"),
    "side_T3":     ("bid",   lambda q, s, v, vz: [*spline(q), v],       "day"),
}
ROBUST_Q = {"cap3": lambda d: d["q_raw"].clip(-3, 3), "uncapped": lambda d: d["q_raw"]}
NAMES = {"side": ["q", "vol5"], "price_ticks": ["q", "vol5"], "price_own": ["q", "vol5"], "size": ["q x s", "vol5"],
         "side_T2": ["q", "vol5", "q x vol"], "price_T2": ["q", "vol5", "q x vol"],
         "side_T3": ["q |q|<1", "q 1-3", "q 3-5", "vol5"]}

# stats[(spec, variant)][(coin, wallet, day)] = (XtX, Xty, n)
stats = defaultdict(dict)


def demean(x, g):
    """x minus its mean within group codes g (unweighted - weights are constant within a maker)."""
    G = g.max() + 1
    n = np.bincount(g, minlength=G).astype(float)
    if x.ndim == 1:
        return x - (np.bincount(g, x, minlength=G) / n)[g]
    return x - np.column_stack([np.bincount(g, x[:, j], minlength=G) / n for j in range(x.shape[1])])[g]


# ---------------------------------------------------------------- pass 1: per maker per file sums
for i, (c, f) in enumerate(files):
    d = pd.read_parquet(f)
    if d.empty:
        continue
    d = d[d["wallet"].isin({w for cc, w in keep if cc == c})]
    if d.empty:
        continue
    day = str(d["day"].iat[0])[:10]                                # always "YYYY-MM-DD" text (date filters below)
    d = d[np.isfinite(d[["vol5", "price_bp", "tick_bp", "log_size"]].to_numpy(float)).all(axis=1)]
    if d.empty:
        continue
    d = d.assign(s=np.where(d["bid"] == 1, -1.0, 1.0), vz=(d["vol5"] - VMEAN[c]) / VSD[c],
                 price=d["price_bp"] / TICK[c], price_own=d["price_bp"] / d["tick_bp"], size=d["log_size"],
                 bid=d["bid"].astype(float))
    for w, m in d.groupby("wallet", sort=False):
        side_g = (m["bid"].to_numpy() > 0).astype(int)
        for variant, qfun in (("main", lambda x: x["q"]), *ROBUST_Q.items()):
            specs = SPECS if variant == "main" else ("side", "price_ticks")
            qv = qfun(m).to_numpy(float)
            ok = np.isfinite(qv)
            if not ok.any():
                continue
            for name in specs:
                yname, xf, fe = SPECS[name]
                X = np.column_stack(xf(qv, m["s"].to_numpy(), m["vol5"].to_numpy(), m["vz"].to_numpy()))[ok]
                y = m[yname].to_numpy(float)[ok]
                g = side_g[ok] if fe == "side_day" else np.zeros(ok.sum(), int)
                Xd, yd = demean(X, g), demean(y, g)
                stats[(name, variant)][(c, w, day)] = (Xd.T @ Xd, Xd.T @ yd, len(yd))
    if (i + 1) % 25 == 0:
        log(f"   pass 1: {i + 1}/{len(files)} files | {time.time() - t00:,.0f}s")
log(f"pass 1 done | {time.time() - t00:,.0f}s")


# ---------------------------------------------------------------- estimation from the sums
def pooled(name, variant="main", keep_fn=lambda c, w, day: True, cluster="coin_day"):
    """Maker-equal-weighted pooled estimate, clustered by coin-day (frozen); "maker" / "day" for the extra SEs."""
    ck = {"coin_day": lambda c, w, day: (c, day), "maker": lambda c, w, day: (c, w), "day": lambda c, w, day: day}[cluster]
    S = {k: v for k, v in stats[(name, variant)].items() if keep_fn(*k)}
    if not S:
        return None
    k = len(NAMES[name])
    A, b = np.zeros((k, k)), np.zeros(k)
    n_in = defaultdict(int)
    for key, v in S.items():
        n_in[key[:2]] += v[2]
    W = {mk: 1.0 / n for mk, n in n_in.items()}         # every maker equal within the selected sample
    for (c, w, day), (xx, xy, n) in S.items():
        A += W[(c, w)] * xx
        b += W[(c, w)] * xy
    beta = np.linalg.solve(A, b)
    sc = defaultdict(lambda: np.zeros(k))
    for (c, w, day), (xx, xy, n) in S.items():
        sc[ck(c, w, day)] += W[(c, w)] * (xy - xx @ beta)
    Ai = np.linalg.inv(A)
    G = len(sc)
    M = sum(np.outer(v, v) for v in sc.values())
    se = np.sqrt(np.diag(Ai @ M @ Ai) * G / max(G - 1, 1))
    return beta, se, G, len({key[:2] for key in S}), sum(v[2] for v in S.values())


def per_maker(name, variant="main"):
    """One regression per maker account, clustered by day."""
    by = defaultdict(list)
    for (c, w, day), v in stats[(name, variant)].items():
        by[(c, w)].append(v)
    out = {}
    for key, vals in by.items():
        if len(vals) < 3:
            continue
        A = sum(v[0] for v in vals)
        b = sum(v[1] for v in vals)
        try:
            beta = np.linalg.solve(A, b)
            Ai = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            continue
        M = sum(np.outer(v[1] - v[0] @ beta, v[1] - v[0] @ beta) for v in vals)
        G = len(vals)
        se = np.sqrt(np.diag(Ai @ M @ Ai) * G / (G - 1))
        out[key] = (beta[0], se[0], G, sum(v[2] for v in vals))
    return out


def pval(t):
    from math import erf, sqrt
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def line(lab, r, j=0, unit=""):
    if r is None:
        return f"   {lab:<34} (no data)"
    beta, se, G, nm, nq = r
    t = beta[j] / se[j]
    return (f"   {lab:<34} {beta[j]:+.4f}{unit} (se {se[j]:.4f}, 95% [{beta[j] - 1.96 * se[j]:+.4f}, "
            f"{beta[j] + 1.96 * se[j]:+.4f}], p {pval(t):.4f}) | {nm} makers, {nq:,} quotes, {G} coin-days")


rows = []


def rec(test, lab, r, j=0):
    if r is not None:
        beta, se, G, nm, nq = r
        rows.append(dict(test=test, estimate=lab, value=beta[j], se=se[j], p=pval(beta[j] / se[j]),
                         makers=nm, quotes=nq, coin_days=G))


print("PART 1 - DO REAL MAKERS MANAGE INVENTORY THE WAY GLFT SAYS?  (pre-registration addendum v3)")
print(f"   coins {', '.join(COINS)} | maker accounts with >= {MIN_QUOTES} quotes: {len(keep)} | "
      f"median tick (bp): " + ", ".join(f"{c} {TICK[c]:.3f}" for c in COINS if c in TICK))

# ---- T1
print("\nT1  WHICH TOOL? (per 1 sd of inventory q; long inventory -> fewer bids, lower prices, bigger asks)")
r_side, r_price, r_size = pooled("side"), pooled("price_ticks"), pooled("size")
print(line("side: change in P(bid)", r_side))
print(line("price: shift in TICKS", r_price))
print(line("size: log size on the unwinding side", r_size))
r_own = pooled("price_own")
print(line("(robustness) price in each quote's own ticks", r_own))
for lab, r in (("side", r_side), ("price_ticks", r_price), ("size", r_size)):
    rec("T1 pooled", lab, r)
print("   same estimates, other clustering (reported, not part of the frozen verdict):")
for cl in ("maker", "day"):
    for lab, nm in (("side", "side"), ("price", "price_ticks")):
        r = pooled(nm, cluster=cl)
        if r is not None:
            print(f"      clustered by {cl:<5} {lab:<5} se {r[1][0]:.4f}, 95% [{r[0][0] - 1.96 * r[1][0]:+.4f}, "
                  f"{r[0][0] + 1.96 * r[1][0]:+.4f}], p {pval(r[0][0] / r[1][0]):.4f} ({r[2]} clusters)")
            rec(f"T1 pooled, clustered by {cl}", nm, r)
a_ok = r_side is not None and r_side[0][0] < 0 and pval(r_side[0][0] / r_side[1][0]) < 0.05
lo, hi = (r_price[0][0] - 1.96 * r_price[1][0], r_price[0][0] + 1.96 * r_price[1][0]) if r_price else (np.nan, np.nan)
b_ok = r_price is not None and lo > -1 and hi < 1
pm_side, pm_price = per_maker("side"), per_maker("price_ticks")
both = sorted(set(pm_side) & set(pm_price))
sig_any = [k for k in both if pval(pm_side[k][0] / pm_side[k][1]) < 0.05 or pval(pm_price[k][0] / pm_price[k][1]) < 0.05]
side_not_price = [k for k in sig_any if pval(pm_side[k][0] / pm_side[k][1]) < 0.05 and pm_side[k][0] < 0
                  and abs(pm_price[k][0]) < 1]
c_ok = len(sig_any) > 0 and len(side_not_price) > len(sig_any) / 2
print(f"\n   (a) side response < 0, p < .05:                 {'YES' if a_ok else 'NO'}")
print(f"   (b) price response inside (-1, +1) tick (95%):   {'YES' if b_ok else 'NO'}  [{lo:+.3f}, {hi:+.3f}] ticks")
print(f"   (c) makers with a significant response: {len(sig_any)}; side-significant with price < 1 tick: "
      f"{len(side_not_price)} ({len(side_not_price) / max(len(sig_any), 1):.0%})  ->  {'YES' if c_ok else 'NO'}")
print(f"\n   T1 VERDICT: {'SUPPORTED - makers manage inventory mainly by quote side, not by price (departure from GLFT)' if a_ok and b_ok and c_ok else 'NOT SUPPORTED as pre-registered'}")
rows.append(dict(test="T1 verdict", estimate="side not price", value=float(a_ok and b_ok and c_ok),
                 se=np.nan, p=np.nan, makers=len(sig_any), quotes=len(side_not_price), coin_days=np.nan))
pmt = pd.DataFrame([dict(coin=k[0], wallet=k[1], b_side=pm_side[k][0], se_side=pm_side[k][1],
                         b_price_ticks=pm_price[k][0], se_price=pm_price[k][1], days=pm_side[k][2],
                         quotes=pm_side[k][3]) for k in both])
pmt.assign(wallet=pmt["wallet"].str[:8]).to_csv(os.path.join(OUT, "part1_per_maker.csv"), index=False)  # 8 chars only
if len(pmt):
    print(f"   days per maker: median {pmt['days'].median():.0f}, makers with < 10 days {int((pmt['days'] < 10).sum())} "
          f"(their individual p-values are rough - few day clusters)")
if len(pmt):
    print(f"   per maker (median over {len(pmt)}): side {pmt['b_side'].median():+.4f}, "
          f"price {pmt['b_price_ticks'].median():+.4f} ticks | side < 0 for {(pmt['b_side'] < 0).mean():.0%}, "
          f"|price| < 1 tick for {(pmt['b_price_ticks'].abs() < 1).mean():.0%}")

# ---- T2
print("\nT2  DOES THE RESPONSE GROW WITH VOLATILITY?  (coefficient on q x standardized vol5)")
r2p, r2s = pooled("price_T2"), pooled("side_T2")
print(line("price (ticks): q x vol", r2p, 2))
print(line("side: q x vol", r2s, 2))
rec("T2", "price q x vol", r2p, 2)
rec("T2", "side q x vol", r2s, 2)
if r2p is not None:
    ok2 = r2p[0][2] < 0 and pval(r2p[0][2] / r2p[1][2]) < 0.05
    print(f"   T2 VERDICT (GLFT: price coefficient < 0): {'CONSISTENT with GLFT' if ok2 else 'NOT as GLFT predicts'}")

# ---- T3
print("\nT3  WHERE ALONG INVENTORY DOES SIDE RESPOND?  (slopes of P(bid) per unit q)")
r3 = pooled("side_T3")
for j, lab in enumerate(["|q| < 1", "1 <= |q| < 3", "3 <= |q| <= 5"]):
    print(line(f"slope {lab}", r3, j))
    rec("T3", f"slope {lab}", r3, j)
if r3 is not None:
    ok3 = pval(r3[0][0] / r3[1][0]) < 0.05
    print(f"   T3 VERDICT (GLFT: ~0 for |q| < 1): {'side responds already at small inventory - NOT as GLFT' if ok3 else 'no significant response at small inventory (not proof that it is zero)'}")

# ---- T4
print("\nT4  DO MAKERS WITH STRONGER SIDE RESPONSE UNWIND FASTER?")
nat_f = os.path.join(DATA, "natives.csv")
if os.path.exists(nat_f) and len(pmt):
    nat = pd.read_csv(nat_f)
    low = {x.lower(): x for x in nat.columns}
    wc = next((low[x] for x in ("wallet", "account", "address", "user_address") if x in low), None)
    hc = next((low[x] for x in low if "half" in x), None)
    cc = low.get("coin")
    if wc and hc:
        nat = nat.assign(wfull=nat[wc].astype(str).str.lower().str.strip())
        key = ["wfull", cc] if cc else ["wfull"]
        dup = int(nat.duplicated(key).sum())
        conflict = int(nat.groupby(key)[hc].nunique(dropna=False).gt(1).sum()) if dup else 0
        if dup:
            print(f"   note: {dup} duplicate maker rows in natives.csv; {conflict} with DIFFERENT half-lives")
        if conflict:
            print("   T4 NOT RUN: conflicting half-lives for the same maker - resolve natives.csv first")
            nat = nat.iloc[0:0]
        m = pmt.assign(wfull=pmt["wallet"].astype(str).str.lower().str.strip()).merge(nat[key + [hc]].drop_duplicates(key),
                                                   left_on=["wfull", "coin"] if cc else ["wfull"], right_on=key)
        m = m[np.isfinite(m[hc]) & (m[hc] > 0)]
        if len(m) >= 8:
            rk = lambda x: pd.Series(x).rank().to_numpy()
            rho = np.corrcoef(rk(m["b_side"]), rk(m[hc]))[0, 1]
            rng = np.random.default_rng(20261114)
            perm = np.array([np.corrcoef(rk(m["b_side"]), rk(rng.permutation(m[hc].to_numpy())))[0, 1]
                             for _ in range(10000)])
            p4 = (1 + np.sum(np.abs(perm) >= abs(rho))) / 10001
            hv, cv = m[hc].to_numpy(), m["coin"].to_numpy()       # within-coin shuffles (coins differ in level)
            perm_c = []
            for _ in range(10000):
                h2 = hv.copy()
                for cc_ in np.unique(cv):
                    ix = np.flatnonzero(cv == cc_)
                    h2[ix] = rng.permutation(h2[ix])
                perm_c.append(np.corrcoef(rk(m["b_side"]), rk(h2))[0, 1])
            p4c = (1 + np.sum(np.abs(np.array(perm_c)) >= abs(rho))) / 10001
            print(f"   Spearman(side response, half-life) = {rho:+.3f}, permutation p {p4:.4f} "
                  f"(shuffling within each coin: p {p4c:.4f}), {len(m)} makers")
            print(f"   T4 VERDICT (predicted > 0; frozen plan: plain permutation): "
                  f"{'SUPPORTED' if rho > 0 and p4 < 0.05 else 'NOT SUPPORTED'} | diagnostic (within-coin shuffle): "
                  f"{'SUPPORTED' if rho > 0 and p4c < 0.05 else 'NOT SUPPORTED'}")
            rows.append(dict(test="T4", estimate="spearman side vs half-life", value=rho, se=np.nan, p=p4,
                             makers=len(m), quotes=np.nan, coin_days=np.nan))
            rows.append(dict(test="T4", estimate="spearman, within-coin shuffle (diagnostic only)", value=rho, se=np.nan, p=p4c,
                             makers=len(m), quotes=np.nan, coin_days=np.nan))
        else:
            print(f"   only {len(m)} makers matched to a half-life - not tested")
    else:
        print(f"   natives.csv has no wallet / half-life column ({list(nat.columns)[:12]}) - not tested")
else:
    print("   natives.csv not found - not tested")

# ---- robustness
print("\nROBUSTNESS (reported, not decisive): side change in P(bid) | price shift in ticks, per 1 sd of q")
nat_lab = {}
if os.path.exists(nat_f):
    nat = pd.read_csv(nat_f)
    low = {x.lower(): x for x in nat.columns}
    wc = next((low[x] for x in ("wallet", "account", "address", "user_address") if x in low), None)
    lc = next((low[x] for x in ("label_cal", "label", "native") if x in low), None)
    if wc and lc:
        nat_lab = {str(a).lower().strip(): str(b) for a, b in zip(nat[wc], nat[lc])}
groups = [(f"coin {c}", "main", (lambda cc: lambda c, w, d: c == cc)(c)) for c in COINS] + [
    ("June (Jun 1-14)", "main", lambda c, w, d: d < "2026-07-01"),
    ("Aug 15 - Oct 25", "main", lambda c, w, d: d >= "2026-07-01"),
    ("q capped at 3", "cap3", lambda c, w, d: True),
    ("q uncapped", "uncapped", lambda c, w, d: True)]
if nat_lab:
    for lab in sorted(set(nat_lab.values())):
        groups.append((f"natives label = {lab}", "main", (lambda L: lambda c, w, d: nat_lab.get(str(w).lower().strip()) == L)(lab)))
for lab, var, fn in groups:
    rs, rp = pooled("side", var, fn), pooled("price_ticks", var, fn)
    if rs is None or rp is None:
        print(f"   {lab:<26} (no data)")
        continue
    print(f"   {lab:<26} side {rs[0][0]:+.4f} (p {pval(rs[0][0] / rs[1][0]):.3f}) | price {rp[0][0]:+.4f} ticks "
          f"(p {pval(rp[0][0] / rp[1][0]):.3f}) | {rs[3]} makers")
    rec("robust", f"{lab} side", rs)
    rec("robust", f"{lab} price_ticks", rp)

pd.DataFrame(rows).to_csv(os.path.join(OUT, "part1_results.csv"), index=False)
print(f"\nSaved part1_results.csv and part1_per_maker.csv (8-character wallets only) in {OUT} | "
      f"{(time.time() - t00) / 60:.0f} min")