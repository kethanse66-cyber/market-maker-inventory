"""T4 and the robustness checks for Part 1."""
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
DATA = os.environ.get("P3_DATA", r"E:\p3\data")
OUT = os.environ.get("P3_RESULTS", r"E:\p3\results")
COINS = ["SOL", "XRP", "BTC", "ETH", "HYPE"]
MIN_QUOTES = 1000
NS = 1_000_000_000
t00 = time.time()
con = duckdb.connect()


def log(s):
    print(s, file=sys.stderr, flush=True)


def prev_day(d):
    return (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


os.makedirs(OUT, exist_ok=True)
files = [(c, f) for c in COINS for f in sorted(glob.glob(os.path.join(PART1, c, "*.parquet")))
         if pq.ParquetFile(f).metadata.num_rows > 0]
if not files:
    sys.exit(f"STOP: no Part 1 files in {PART1}")

# ---------------------------------------------------------------- pass 0: the same makers and ticks as day21
cnt = defaultdict(int)
tick_med = defaultdict(list)
vs = defaultdict(lambda: [0, 0.0, 0.0])
big = defaultdict(lambda: [0, 0, 0, 0.0])      # coin: quotes, |q_raw| > 5, |q_raw| > 50, max |q_raw|
big_m = defaultdict(int)                       # (coin, wallet): quotes with |q_raw| > 5
for c, f in files:
    d = pd.read_parquet(f, columns=["wallet", "tick_bp", "vol5", "q_raw"])
    for w, n in d["wallet"].value_counts().items():
        cnt[(c, w)] += int(n)
    tick_med[c].append(float(d["tick_bp"].median()))
    v = vs[c]
    v[0] += int(d["vol5"].notna().sum()); v[1] += float(d["vol5"].sum()); v[2] += float((d["vol5"] ** 2).sum())
    a = d["q_raw"].abs()
    b = big[c]
    b[0] += len(d); b[1] += int((a > 5).sum()); b[2] += int((a > 50).sum()); b[3] = max(b[3], float(a.max()))
    for w, n in d.loc[a > 5, "wallet"].value_counts().items():
        big_m[(c, w)] += int(n)
VMEAN = {c: v[1] / v[0] for c, v in vs.items()}
VSD = {c: np.sqrt(max(v[2] / v[0] - (v[1] / v[0]) ** 2, 1e-12)) for c, v in vs.items()}
TICK = {c: float(np.median(x)) for c, x in tick_med.items()}
keep = {k for k, n in cnt.items() if n >= MIN_QUOTES}
log(f"pass 0 done: {len(keep)} maker accounts with >= {MIN_QUOTES} quotes | {time.time() - t00:,.0f}s")


def demean(x, g):
    G = g.max() + 1
    n = np.maximum(np.bincount(g, minlength=G), 1).astype(float)   # empty groups never indexed
    if x.ndim == 1:
        return x - (np.bincount(g, x, minlength=G) / n)[g]
    return x - np.column_stack([np.bincount(g, x[:, j], minlength=G) / n for j in range(x.shape[1])])[g]


def mid_series(coin, d):
    """Per-second market mid from the quote tables (same source as vol5), forward-filled."""
    qf = sorted(glob.glob(os.path.join(QUOTES, coin, f"{prev_day(d)}*.parquet"))) + \
         sorted(glob.glob(os.path.join(QUOTES, coin, f"{d}*.parquet")))
    D0s = pd.Timestamp(d, tz="UTC").value // NS
    ms = con.execute(f"""SELECT ts // {NS} AS sec, arg_max(mid, ts) AS mid FROM read_parquet({qf}, union_by_name=true)
                         WHERE mid IS NOT NULL AND ts >= {(D0s - 120) * NS} GROUP BY 1 ORDER BY 1""").df()
    if ms.empty:
        return None
    full = np.arange(ms["sec"].min(), ms["sec"].max() + 1)
    return pd.Series(np.log(ms["mid"].to_numpy(float)), index=ms["sec"].to_numpy()).reindex(full).ffill()


def fill_times(coin, d):
    invf = [f for f in (os.path.join(INV, coin, f"{x}.parquet") for x in (prev_day(d), d)) if os.path.exists(f)]
    if not invf:
        return pd.DataFrame({"wallet": pd.Series(dtype=object), "fts": pd.Series(dtype="int64")})
    tcol = con.execute(f"DESCRIBE SELECT ts FROM read_parquet({invf}, union_by_name=true)").fetchone()[1]
    T = "epoch_ns(ts)" if "TIMESTAMP" in tcol.upper() else "ts"
    return con.execute(f"SELECT lower(wallet) AS wallet, CAST({T} AS BIGINT) AS fts "
                       f"FROM read_parquet({invf}, union_by_name=true)").df()


# ---------------------------------------------------------------- pass 1: sums per maker per file, per version
# version: (rows to keep, extra controls)
VERSIONS = {
    "all":            (lambda m: np.ones(len(m), bool),            False),   # = day21 (sanity check)
    "same sample":    (lambda m: m["has_ret"],                      False),   # rows where returns exist
    "no fill 1s":     (lambda m: m["has_ret"] & (m["since"] > 1),   False),
    "no fill 5s":     (lambda m: m["has_ret"] & (m["since"] > 5),   False),
    "no fill 10s":    (lambda m: m["has_ret"] & (m["since"] > 10),  False),
    "no fill 30s":    (lambda m: m["has_ret"] & (m["since"] > 30),  False),
    "no fill 60s":    (lambda m: m["has_ret"] & (m["since"] > 60),  False),
    "return ctrl":    (lambda m: m["has_ret"],                      True),
    "both (60s+ret)": (lambda m: m["has_ret"] & (m["since"] > 60),  True),
}
stats = defaultdict(dict)       # stats[(outcome, version)][(coin, wallet, day)] = (XtX, Xty, n)
share = defaultdict(lambda: [0, 0, 0, 0])   # coin: quotes, has_ret, >10s, >60s
for i, (c, f) in enumerate(files):
    d = pd.read_parquet(f)
    d = d[d["wallet"].isin({w for cc, w in keep if cc == c})]
    d = d[np.isfinite(d[["q", "vol5", "price_bp", "tick_bp", "log_size"]].to_numpy(float)).all(axis=1)]  # = day21
    if d.empty:
        continue
    day = str(d["day"].iat[0])[:10]
    d = d.sort_values("ts").reset_index(drop=True)
    # seconds since the same maker's last fill
    ft = fill_times(c, day).sort_values("fts")
    ft = ft[ft["wallet"].isin(set(d["wallet"]))]
    if len(ft):
        j = pd.merge_asof(d[["ts", "wallet"]].assign(ts=d["ts"].astype("int64")), ft, left_on="ts", right_on="fts",
                          by="wallet", direction="backward")
        since = ((j["ts"] - j["fts"]) / NS).to_numpy(float)
    else:
        since = np.full(len(d), np.nan)
    d["since"] = np.where(np.isnan(since), np.inf, since)
    # market return over the 10 s and 60 s up to the second before the quote (bp)
    lm = mid_series(c, day)
    sec = (d["ts"].to_numpy("int64") // NS)
    if lm is not None:
        get = lambda s: lm.reindex(s).to_numpy(float)
        d["r10"] = (get(sec - 1) - get(sec - 11)) * 1e4
        d["r60"] = (get(sec - 1) - get(sec - 61)) * 1e4
    else:
        d["r10"] = d["r60"] = np.nan
    d["has_ret"] = np.isfinite(d["r10"]) & np.isfinite(d["r60"])
    d["price"] = d["price_bp"] / TICK[c]
    d["vz"] = (d["vol5"] - VMEAN[c]) / VSD[c]
    sh = share[c]
    sh[0] += len(d); sh[1] += int(d["has_ret"].sum())
    sh[2] += int((d["has_ret"] & (d["since"] > 5)).sum()); sh[3] += int((d["has_ret"] & (d["since"] > 60)).sum())
    for w, m in d.groupby("wallet", sort=False):
        side_g = (m["bid"].to_numpy() > 0).astype(int)
        for ver, (rows_fn, ctrl) in VERSIONS.items():
            ok = np.asarray(rows_fn(m), bool)
            if ok.sum() < 2:
                continue
            cols = [m["q"].to_numpy(float), m["vol5"].to_numpy(float)]
            if ctrl:
                cols += [m["r10"].to_numpy(float), m["r60"].to_numpy(float)]
            X = np.column_stack(cols)[ok]
            todo = [("side", "bid", "day", X), ("price", "price", "side_day", X)]
            if ver == "all":                                        # T2 model: q, vol5, q x vol (as day21)
                X2 = np.column_stack([X, m["q"].to_numpy(float) * m["vz"].to_numpy(float)])
                todo += [("side_T2", "bid", "day", X2), ("price_T2", "price", "side_day", X2)]
            for outcome, yname, fe, XX in todo:
                y = m[yname].to_numpy(float)[ok]
                g = side_g[ok] if fe == "side_day" else np.zeros(ok.sum(), int)
                Xd, yd = demean(XX, g), demean(y, g)
                stats[(outcome, ver)][(c, w, day)] = (Xd.T @ Xd, Xd.T @ yd, len(yd))
    if (i + 1) % 25 == 0:
        log(f"   pass 1: {i + 1}/{len(files)} files | {time.time() - t00:,.0f}s")
log(f"pass 1 done | {time.time() - t00:,.0f}s")


def pooled(key, keep_fn=lambda c, w, day: True, cluster="coin_day"):
    S = {k: v for k, v in stats[key].items() if keep_fn(*k)}
    if not S:
        return None
    k = next(iter(S.values()))[0].shape[0]
    A, b = np.zeros((k, k)), np.zeros(k)
    n_in = defaultdict(int)
    for kk, v in S.items():
        n_in[kk[:2]] += v[2]
    W = {mk: 1.0 / n for mk, n in n_in.items()}
    for (c, w, day), (xx, xy, n) in S.items():
        A += W[(c, w)] * xx
        b += W[(c, w)] * xy
    try:
        beta = np.linalg.solve(A, b)
        Ai = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None
    sc = defaultdict(lambda: np.zeros(k))
    for (c, w, day), (xx, xy, n) in S.items():
        ck = {"coin_day": (c, day), "maker": (c, w), "day": day}[cluster]
        sc[ck] += W[(c, w)] * (xy - xx @ beta)
    G = len(sc)
    M = sum(np.outer(v, v) for v in sc.values())
    se = np.sqrt(np.diag(Ai @ M @ Ai) * G / max(G - 1, 1))
    return beta, se, G, len(n_in), sum(v[2] for v in S.values())


def per_maker(key):
    by = defaultdict(list)
    for (c, w, day), v in stats[key].items():
        by[(c, w)].append(v)
    out = {}
    for mk, vals in by.items():
        if len(vals) < 3:
            continue
        A, b = sum(v[0] for v in vals), sum(v[1] for v in vals)
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        out[mk] = beta[0]
    return out


def pval(t):
    from math import erf, sqrt
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


rows = []
print("PART 1 CHECKS (day22) - T4 (frozen) and two robustness checks of T1")
print("   quotes kept per check (share of all T1 quotes):")
for c in COINS:
    if c in share:
        a, r, t10, t60 = share[c]
        print(f"      {c:<5} {a:>12,} quotes | with past returns {r / a:.1%} | >5 s after a fill {t10 / a:.1%} "
              f"| >60 s after a fill {t60 / a:.1%}")

print("\nCHECKS A and B: T1 per 1 sd of q, re-estimated (all coins pooled)")
print(f"   {'version':<16} {'side: change in P(bid)':<36} {'price: shift in ticks':<36} makers  quotes")
for ver in VERSIONS:
    rs, rp = pooled(("side", ver)), pooled(("price", ver))
    if rs is None or rp is None:
        print(f"   {ver:<16} (no data)")
        continue
    f_ = lambda r: f"{r[0][0]:+.4f} [{r[0][0] - 1.96 * r[1][0]:+.4f}, {r[0][0] + 1.96 * r[1][0]:+.4f}] p {pval(r[0][0] / r[1][0]):.3f}"
    print(f"   {ver:<16} {f_(rs):<36} {f_(rp):<36} {rs[3]:>5}  {rs[4]:>12,}")
    for nm, r in (("side", rs), ("price_ticks", rp)):
        rows.append(dict(test="check", version=ver, estimate=nm, value=r[0][0], se=r[1][0], p=pval(r[0][0] / r[1][0]),
                         makers=r[3], quotes=r[4], coin_days=r[2]))
    if ver.startswith("return") or ver.startswith("both"):
        rr = pooled(("side", ver))
        print(f"   {'':<16} (controls in the side model: past 10 s return {rr[0][2]:+.5f} per bp, "
              f"past 60 s return {rr[0][3]:+.5f} per bp)")
print("   By coin, strictest version (both):")
for c in COINS:
    rs = pooled(("side", "both (60s+ret)"), lambda cc, w, d, c=c: cc == c)
    rp = pooled(("price", "both (60s+ret)"), lambda cc, w, d, c=c: cc == c)
    if rs is not None and rp is not None:
        print(f"      {c:<5} side {rs[0][0]:+.4f} (p {pval(rs[0][0] / rs[1][0]):.3f}) | price {rp[0][0]:+.4f} ticks "
              f"(p {pval(rp[0][0] / rp[1][0]):.3f}) | {rs[3]} makers")
        rows.append(dict(test="check by coin", version="both (60s+ret)", estimate=f"{c} side", value=rs[0][0],
                         se=rs[1][0], p=pval(rs[0][0] / rs[1][0]), makers=rs[3], quotes=rs[4], coin_days=rs[2]))

# ---------------------------------------------------------------- T2 with other clustering
print("\nT2 AGAIN: coefficient on q x standardized vol5, three clusterings (day21 = coin-day)")
for nm, lab in (("price_T2", "price (ticks)"), ("side_T2", "side")):
    for cl in ("coin_day", "maker", "day"):
        r = pooled((nm, "all"), cluster=cl)
        if r is None:
            continue
        print(f"   {lab:<14} clustered by {cl:<9} {r[0][2]:+.4f} (se {r[1][2]:.4f}, 95% [{r[0][2] - 1.96 * r[1][2]:+.4f}, "
              f"{r[0][2] + 1.96 * r[1][2]:+.4f}], p {pval(r[0][2] / r[1][2]):.4f}) | {r[2]} clusters")
        rows.append(dict(test="T2 clustering", version=cl, estimate=f"{nm} q x vol", value=r[0][2], se=r[1][2],
                         p=pval(r[0][2] / r[1][2]), makers=r[3], quotes=r[4], coin_days=r[2]))

# ---------------------------------------------------------------- uncapped q
print("\nWHY 'q uncapped' IS ~0: quotes with |q_raw| above the cap (all T1 quotes, before the maker filter)")
for c in COINS:
    if c in big:
        n, b5, b50, mx = big[c]
        mk = sorted(((v, w) for (cc, w), v in big_m.items() if cc == c), reverse=True)
        tot = sum(v for v, _ in mk)
        top = ", ".join(f"{w[:8]} {v / tot:.0%}" for v, w in mk[:3]) if tot else "-"
        print(f"   {c:<5} |q_raw| > 5: {b5:>10,} ({b5 / n:.2%}) | > 50: {b50:>9,} ({b50 / n:.3%}) | max {mx:,.0f} "
              f"| from {len(mk)} makers, top 3: {top}")

# ---------------------------------------------------------------- T4 and natives labels
print("\nT4  DO MAKERS WITH STRONGER SIDE RESPONSE UNWIND FASTER?  (frozen test, prereg-v3)")
nat_f = os.path.join(DATA, "natives.csv")
pm = per_maker(("side", "all"))
nat_lab = {}
if not os.path.exists(nat_f):
    print("   natives.csv not found - not tested")
else:
    nat = pd.read_csv(nat_f)
    nat = nat[nat["level"].astype(str).str.lower() == "account"].copy()
    nat["wfull"] = nat["unit"].astype(str).str.lower().str.strip()
    nat["coin"] = nat["coin"].astype(str).str.upper()
    lab_col = next((x for x in ("label_cal", "label") if x in nat.columns), None)
    if lab_col:
        nat_lab = {(r.coin, r.wfull): str(getattr(r, lab_col)) for r in nat.itertuples()}
    dup = int(nat.duplicated(["coin", "wfull"]).sum())
    if dup:
        print(f"   STOP for T4: {dup} duplicate account rows in natives.csv")
    else:
        pmt = pd.DataFrame([dict(coin=k[0], wfull=str(k[1]).lower().strip(), b_side=v) for k, v in pm.items()])
        m = pmt.merge(nat[["coin", "wfull", "half_life_min"]], on=["coin", "wfull"])
        print(f"   makers with a per-maker side response: {len(pmt)} | with an account half-life row: {len(m)} "
              f"(natives.csv covers accounts with 500+ fills on its 11 test days)")
        hl = m["half_life_min"].astype(float)
        print(f"   half-life: finite {int(np.isfinite(hl).sum())}, never reverts (inf) {int(np.isinf(hl).sum())}, "
              f"missing {int(hl.isna().sum())}")
        rk = lambda x: pd.Series(x).rank().to_numpy()
        rng = np.random.default_rng(20261114)
        for lab, mm in (("finite half-lives (frozen code rule)", m[np.isfinite(hl) & (hl > 0)]),
                        ("incl. 'never reverts' as slowest (robustness)", m[hl.notna() & (hl > 0)])):
            if len(mm) < 8:
                print(f"   {lab}: only {len(mm)} makers - not tested")
                continue
            a, h = mm["b_side"].to_numpy(), mm["half_life_min"].to_numpy(float)
            rho = np.corrcoef(rk(a), rk(h))[0, 1]
            perm = np.array([np.corrcoef(rk(a), rk(rng.permutation(h)))[0, 1] for _ in range(10000)])
            p4 = (1 + np.sum(np.abs(perm) >= abs(rho))) / 10001
            print(f"   {lab}: Spearman {rho:+.3f}, permutation p {p4:.4f}, {len(mm)} makers")
            rows.append(dict(test="T4", version=lab, estimate="spearman side vs half-life", value=rho, se=np.nan,
                             p=p4, makers=len(mm), quotes=np.nan, coin_days=np.nan))
            if lab.startswith("finite"):
                print(f"   T4 VERDICT (predicted > 0, p < .05): {'SUPPORTED' if rho > 0 and p4 < 0.05 else 'NOT SUPPORTED'}")
        print("   (half-life measured on Aug-Oct test days only; side response on all 86 days)")

print("\nROBUSTNESS: hedgers vs natives (account labels from natives.csv), main T1 sample")
if nat_lab:
    for lab in sorted(set(nat_lab.values())):
        fn = (lambda L: lambda c, w, d: nat_lab.get((c, str(w).lower().strip())) == L)(lab)
        rs, rp = pooled(("side", "all"), fn), pooled(("price", "all"), fn)
        if rs is None or rp is None:
            print(f"   {lab:<12} (no data)")
            continue
        print(f"   {lab:<12} side {rs[0][0]:+.4f} (p {pval(rs[0][0] / rs[1][0]):.3f}) | price {rp[0][0]:+.4f} ticks "
              f"(p {pval(rp[0][0] / rp[1][0]):.3f}) | {rs[3]} makers")
        rows.append(dict(test="robust natives", version=lab, estimate="side", value=rs[0][0], se=rs[1][0],
                         p=pval(rs[0][0] / rs[1][0]), makers=rs[3], quotes=rs[4], coin_days=rs[2]))
        rows.append(dict(test="robust natives", version=lab, estimate="price_ticks", value=rp[0][0], se=rp[1][0],
                         p=pval(rp[0][0] / rp[1][0]), makers=rp[3], quotes=rp[4], coin_days=rp[2]))
    no = sum(1 for (c, w) in keep if (c, str(w).lower().strip()) not in nat_lab)
    print(f"   ({no} of {len(keep)} makers have no account label - too few fills on the natives test days)")
else:
    print("   no labels found in natives.csv")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "part1_checks.csv"), index=False)
print(f"\nSaved part1_checks.csv in {OUT} | {(time.time() - t00) / 60:.0f} min")
