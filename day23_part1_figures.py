"""Part 1 figures and tables, and the q-scaling check."""
import glob
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("STOP: matplotlib missing - run:  pip install matplotlib")

PART1 = os.environ.get("P3_PART1", r"E:\p3\part1")
DATA = os.environ.get("P3_DATA", r"E:\p3\data")
RES = os.environ.get("P3_RESULTS", r"E:\p3\results")
FIG = os.environ.get("P3_FIGURES", r"E:\p3\figures")
COINS = ["SOL", "XRP", "BTC", "ETH", "HYPE"]
MIN_QUOTES, CAP, HEAVY = 1000, 5.0, 0.20
EDGES = np.arange(-5, 5.01, 0.5)              # 20 bins of q
MIN_BIN = 30                                  # a maker counts in a bin with 30+ quotes there
t00 = time.time()

for f in ("part1_results.csv", "part1_per_maker.csv", "part1_checks.csv"):
    if not os.path.exists(os.path.join(RES, f)):
        sys.exit(f"STOP: {f} missing in {RES} - run day21 and day22 first")
os.makedirs(FIG, exist_ok=True)


def log(s):
    print(s, file=sys.stderr, flush=True)


files = [(c, f) for c in COINS for f in sorted(glob.glob(os.path.join(PART1, c, "*.parquet")))
         if pq.ParquetFile(f).metadata.num_rows > 0]

# ---------------------------------------------------------------- pass 0: makers, ticks, cap share, vol terciles
cnt, beyond = defaultdict(int), defaultdict(int)
tick_med, vsample = defaultdict(list), defaultdict(list)
for c, f in files:
    d = pd.read_parquet(f, columns=["wallet", "tick_bp", "vol5", "q_raw"])
    for w, n in d["wallet"].value_counts().items():
        cnt[(c, w)] += int(n)
    for w, n in d.loc[d["q_raw"].abs() > CAP, "wallet"].value_counts().items():
        beyond[(c, w)] += int(n)
    tick_med[c].append(float(d["tick_bp"].median()))
    v = d["vol5"].to_numpy(float)
    vsample[c].append(v[np.isfinite(v)][::50])
TICK = {c: float(np.median(x)) for c, x in tick_med.items()}
keep = {k for k, n in cnt.items() if n >= MIN_QUOTES}
heavy = {k for k in keep if beyond[k] / cnt[k] > HEAVY}
VT = {}
for c, x in vsample.items():
    x = np.concatenate(x) if x else np.array([])
    VT[c] = np.quantile(x, [1 / 3, 2 / 3]) if len(x) else np.array([np.nan, np.nan])   # no vol -> not in fig2
log(f"pass 0 done: {len(keep)} makers, {len(heavy)} with > {HEAVY:.0%} of quotes beyond the cap | {time.time() - t00:,.0f}s")


def demean(x, g):
    G = g.max() + 1
    n = np.maximum(np.bincount(g, minlength=G), 1).astype(float)
    if x.ndim == 1:
        return x - (np.bincount(g, x, minlength=G) / n)[g]
    return x - np.column_stack([np.bincount(g, x[:, j], minlength=G) / n for j in range(x.shape[1])])[g]


# ---------------------------------------------------------------- pass 1
stats = defaultdict(dict)                     # (outcome, version) -> (coin, wallet, day) -> (XtX, Xty, n)
bins = defaultdict(lambda: np.zeros((20, 3)))  # (coin, wallet) -> per bin: n, sum side, sum price
vbins = defaultdict(lambda: np.zeros((2, 20, 2)))  # (coin, wallet) -> [low/high vol] per bin: n, sum price
for i, (c, f) in enumerate(files):
    d = pd.read_parquet(f)
    d = d[d["wallet"].isin({w for cc, w in keep if cc == c})]
    d = d[np.isfinite(d[["q", "vol5", "price_bp", "tick_bp", "log_size"]].to_numpy(float)).all(axis=1)]
    if d.empty:
        continue
    day = str(d["day"].iat[0])[:10]
    d = d.assign(price=d["price_bp"] / TICK[c])
    lo, hi = VT[c]
    for w, m in d.groupby("wallet", sort=False):
        bid = m["bid"].to_numpy(float)
        side_g = (bid > 0).astype(int)
        z0 = np.zeros(len(m), int)
        q = m["q"].to_numpy(float)
        qr = m["q_raw"].to_numpy(float)
        v = m["vol5"].to_numpy(float)
        price = m["price"].to_numpy(float)
        # binned shapes (change vs the maker's own average that day
        b = np.clip(np.searchsorted(EDGES, q, side="right") - 1, 0, 19)
        s_dm, p_dm = demean(bid, z0), demean(price, side_g)
        B = bins[(c, w)]
        np.add.at(B, (b, 0), 1.0)
        np.add.at(B[:, 1], b, s_dm)
        np.add.at(B[:, 2], b, p_dm)
        V = vbins[(c, w)]
        for k, mask in ((0, v <= lo), (1, v >= hi)):
            np.add.at(V[k, :, 0], b[mask], 1.0)
            np.add.at(V[k, :, 1], b[mask], p_dm[mask])
        # regressions for the q-scaling check
        n = len(qr)
        rank_z = norm.ppf((pd.Series(qr).rank(method="average").to_numpy() - 0.5) / n) if n > 1 else np.zeros(n)
        for ver, x in (("main", q), ("rank q", rank_z), ("sign q", np.sign(qr))):
            X = np.column_stack([x, v])
            for outcome, y, g in (("side", bid, z0), ("price", price, side_g)):
                Xd, yd = demean(X, g), demean(y, g)
                stats[(outcome, ver)][(c, w, day)] = (Xd.T @ Xd, Xd.T @ yd, n)
    if (i + 1) % 25 == 0:
        log(f"   pass 1: {i + 1}/{len(files)} files | {time.time() - t00:,.0f}s")
log(f"pass 1 done | {time.time() - t00:,.0f}s")


def pooled(key, keep_fn=lambda c, w, day: True):
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
        beta, Ai = np.linalg.solve(A, b), np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None
    sc = defaultdict(lambda: np.zeros(k))
    for (c, w, day), (xx, xy, n) in S.items():
        sc[(c, day)] += W[(c, w)] * (xy - xx @ beta)
    G = len(sc)
    M = sum(np.outer(v, v) for v in sc.values())
    se = np.sqrt(np.diag(Ai @ M @ Ai) * G / max(G - 1, 1))
    return beta[0], se[0], len(n_in), sum(v[2] for v in S.values())


def pval(t):
    from math import erf, sqrt
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


# ---------------------------------------------------------------- q-scaling check
print("q-SCALING CHECK (robustness): T1 per unit of q, coin-day clusters, every maker weighted equally")
print(f"   {'version':<34} {'side: change in P(bid)':<38} {'price: shift in ticks':<38} makers")
qrows = []
for lab, ver, fn in (("main q (= day21)", "main", lambda c, w, d: True),
                     ("rank q (normal score in maker-day)", "rank q", lambda c, w, d: True),
                     ("sign q (long +1 / short -1)", "sign q", lambda c, w, d: True),
                     (f"main q, without {len(heavy)} cap-heavy makers", "main", lambda c, w, d: (c, w) not in heavy)):
    rs, rp = pooled(("side", ver), fn), pooled(("price", ver), fn)
    if rs is None or rp is None:
        print(f"   {lab:<34} (no data)")
        continue
    f_ = lambda r: f"{r[0]:+.4f} [{r[0] - 1.96 * r[1]:+.4f}, {r[0] + 1.96 * r[1]:+.4f}] p {pval(r[0] / r[1]):.3f}"
    print(f"   {lab:<34} {f_(rs):<38} {f_(rp):<38} {rs[2]:>5}")
    for nm, r in (("side", rs), ("price_ticks", rp)):
        qrows.append(dict(version=lab, estimate=nm, value=r[0], se=r[1], p=pval(r[0] / r[1]), makers=r[2], quotes=r[3]))
print("   (sign q: the model-implied long-minus-short contrast is 2 x the coefficient)")
Q = pd.DataFrame(qrows)
Q.to_csv(os.path.join(RES, "part1_qscale.csv"), index=False)

# ---------------------------------------------------------------- figure style
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
COIN_C = dict(zip(COINS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]))
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False})


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print(f"   saved {name}.png / .pdf")


def maker_curve(arr_n, arr_s):
    """Equal-weight mean across makers per bin (makers with MIN_BIN+ quotes in the bin) and its 95% band."""
    mean, lo, hi, nm = (np.full(20, np.nan) for _ in range(4))
    for j in range(20):
        ok = arr_n[:, j] >= MIN_BIN
        if ok.sum() >= 5:
            x = arr_s[ok, j] / arr_n[ok, j]
            mean[j] = x.mean()
            se = x.std(ddof=1) / np.sqrt(ok.sum())
            lo[j], hi[j], nm[j] = mean[j] - 1.96 * se, mean[j] + 1.96 * se, ok.sum()
    return mean, lo, hi, nm


mid = (EDGES[:-1] + EDGES[1:]) / 2
print("\nFIGURES")
# fig1 (each coin-day file is one day, grouped by wallet -> every average is within maker-day; price also within side)
Bn = np.array([v[:, 0] for v in bins.values()])
Bs = np.array([v[:, 1] for v in bins.values()])
Bp = np.array([v[:, 2] for v in bins.values()])
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, arr, col, title, yl in ((ax[0], Bs, C1, "Side: share of quotes that are bids", "change in P(bid), percentage points"),
                               (ax[1], Bp, C2, "Price: quote offset from mid", "change in offset, ticks (same side, same day)")):
    m_, l_, h_, _ = maker_curve(Bn, arr)
    scale = 100 if a is ax[0] else 1
    a.fill_between(mid, l_ * scale, h_ * scale, color=col, alpha=0.18, linewidth=0)
    a.plot(mid, m_ * scale, color=col, linewidth=2, marker="o", markersize=4)
    a.axhline(0, color=INK2, linewidth=0.8)
    a.axvline(0, color=INK2, linewidth=0.8)
    a.set_title(title, loc="left")
    a.set_xlabel("inventory q (sd of the maker's position; + = long)")
    a.set_ylabel(yl)
for yv in (-1, 1):
    ax[1].axhline(yv, color=INK2, linewidth=1, linestyle="--")
ax[1].text(4.9, 1.05, "one tick", ha="right", va="bottom", color=INK2, fontsize=9)
save(fig, "fig1_side_vs_price_along_inventory")

# fig2
Vn = np.array([v[:, :, 0] for v in vbins.values()])
Vs = np.array([v[:, :, 1] for v in vbins.values()])
fig, a = plt.subplots(figsize=(6, 4))
for k, col, lab in ((0, C1, "low volatility (bottom third)"), (1, C2, "high volatility (top third)")):
    m_, l_, h_, _ = maker_curve(Vn[:, k, :], Vs[:, k, :])
    a.fill_between(mid, l_, h_, color=col, alpha=0.15, linewidth=0)
    a.plot(mid, m_, color=col, linewidth=2, marker="o", markersize=4, label=lab)
    if np.isfinite(m_).any():                                  # label at the last populated bin
        j = int(np.flatnonzero(np.isfinite(m_))[-1])
        a.text(mid[j] + 0.15, m_[j], lab.split(" (")[0], color=INK, fontsize=9, va="center")
a.axhline(0, color=INK2, linewidth=0.8)
a.set_title("Price shading is weaker when volatility is high (T2)", loc="left")
a.set_xlabel("inventory q (sd; + = long)")
a.set_ylabel("change in quote offset, ticks")
a.set_xlim(-5.2, 6.6)
a.legend(loc="lower left", fontsize=9)
save(fig, "fig2_price_response_by_volatility")

# fig3
def per_maker(key):
    by = defaultdict(list)
    for (c, w, day), v in stats[key].items():
        by[(c, w)].append(v)
    out = {}
    for mk, vals in by.items():
        if len(vals) < 3:                                       # same rule as day21 (3+ days)
            continue
        try:
            out[mk] = np.linalg.solve(sum(v[0] for v in vals), sum(v[1] for v in vals))[0]
        except np.linalg.LinAlgError:
            pass
    return out


ps, pp = per_maker(("side", "main")), per_maker(("price", "main"))
pm = pd.DataFrame([dict(coin=k[0], wallet=k[1], b_side=ps[k], b_price_ticks=pp[k]) for k in sorted(set(ps) & set(pp))])
print(f"   per-maker estimates (full wallets, 3+ days): {len(pm)} of {len(keep)} makers")
fig, a = plt.subplots(figsize=(6.5, 4.8))
a.axvspan(-1, 1, color=GRID, alpha=0.6, linewidth=0)
for c in COINS:
    s = pm[pm["coin"] == c]
    a.scatter(s["b_price_ticks"], s["b_side"] * 100, s=22, color=COIN_C[c], edgecolor=SURF, linewidth=0.8,
              label=f"{c} ({len(s)})", zorder=3)
a.axhline(0, color=INK2, linewidth=0.8)
a.axvline(0, color=INK2, linewidth=0.8)
xl = np.nanpercentile(pm["b_price_ticks"].abs(), 98) * 1.15
a.set_xlim(-max(xl, 1.5), max(xl, 1.5))
a.text(-0.97, a.get_ylim()[1], " within one tick", ha="left", va="top", color=INK2, fontsize=9)
inside = (pm["b_price_ticks"].abs() < 1).mean()
neg = (pm["b_side"] < 0).mean()
a.set_title(f"Each maker with 3+ days ({len(pm)}): side vs price response (T1)\n{neg:.0%} pull bids when long; "
            f"{inside:.0%} shade price by less than one tick", loc="left", fontsize=10)
a.set_xlabel("price response, ticks per sd of q")
a.set_ylabel("side response, pp of P(bid) per sd of q")
a.legend(fontsize=8, loc="lower right", title="coin (makers)", title_fontsize=8)
save(fig, "fig3_per_maker_side_vs_price")

# fig4
nat_f = os.path.join(DATA, "natives.csv")
if os.path.exists(nat_f):
    nat = pd.read_csv(nat_f)
    nat = nat[nat["level"].astype(str).str.lower() == "account"].copy()
    nat["wfull"] = nat["unit"].astype(str).str.lower().str.strip()
    nat["coin"] = nat["coin"].astype(str).str.upper()
    nat = nat[~nat.duplicated(["coin", "wfull"], keep=False)]
    m = pm.assign(wfull=pm["wallet"].astype(str).str.lower().str.strip()).merge(
        nat[["coin", "wfull", "half_life_min"]], on=["coin", "wfull"])        # full-wallet match, as day22
    m = m[np.isfinite(m["half_life_min"]) & (m["half_life_min"] > 0)]
    if len(m) >= 8:
        rho = np.corrcoef(m["b_side"].rank(), m["half_life_min"].rank())[0, 1]
        fig, a = plt.subplots(figsize=(6.5, 4.6))
        for c in COINS:
            s = m[m["coin"] == c]
            a.scatter(s["half_life_min"], s["b_side"] * 100, s=22, color=COIN_C[c], edgecolor=SURF, linewidth=0.8,
                      label=c, zorder=3)
        a.set_xscale("log")
        a.axhline(0, color=INK2, linewidth=0.8)
        a.set_title(f"Makers that pull bids harder unwind faster (T4)\nSpearman {rho:+.2f}, {len(m)} makers",
                    loc="left", fontsize=10)
        a.set_xlabel("inventory half-life, minutes (log scale; Aug-Oct test days)")
        a.set_ylabel("side response, pp of P(bid) per sd of q")
        a.legend(fontsize=8, loc="lower right")
        save(fig, "fig4_side_response_vs_half_life")
        print(f"      fig4 uses {len(m)} makers (full-wallet match; should equal day22's T4 count)")
else:
    print("   natives.csv not found - fig4 skipped")

# fig5 - robustness forest
R = pd.read_csv(os.path.join(RES, "part1_results.csv"))
K = pd.read_csv(os.path.join(RES, "part1_checks.csv"))
rows = []


def add(label, group, side, price):
    if side is not None and price is not None:
        rows.append((label, group, side, price))


def pick(df, **kw):
    s = df
    for k_, v_ in kw.items():
        s = s[s[k_] == v_]
    return (float(s["value"].iat[0]), float(s["se"].iat[0])) if len(s) else None


add("Main estimate (T1)", "main", pick(R, test="T1 pooled", estimate="side"), pick(R, test="T1 pooled", estimate="price_ticks"))
for c in COINS:
    add(f"{c} only", "coin", pick(R, test="robust", estimate=f"coin {c} side"), pick(R, test="robust", estimate=f"coin {c} price_ticks"))
add("June only", "period", pick(R, test="robust", estimate="June (Jun 1-14) side"), pick(R, test="robust", estimate="June (Jun 1-14) price_ticks"))
add("Aug-Oct only", "period", pick(R, test="robust", estimate="Aug 15 - Oct 25 side"), pick(R, test="robust", estimate="Aug 15 - Oct 25 price_ticks"))
for v in ("no fill 1s", "no fill 5s", "no fill 10s", "no fill 30s", "no fill 60s", "return ctrl", "both (60s+ret)"):
    add({"return ctrl": "control past returns", "both (60s+ret)": "no fill 60s + returns"}.get(v, v.replace("no fill", "no fill within")),
        "checks", pick(K, test="check", version=v, estimate="side"), pick(K, test="check", version=v, estimate="price_ticks"))
for lab in ("NATIVE", "HEDGER"):
    add(f"{lab.lower()}s only", "type", pick(K, test="robust natives", version=lab, estimate="side"),
        pick(K, test="robust natives", version=lab, estimate="price_ticks"))
add("q capped at 3", "q", pick(R, test="robust", estimate="q capped at 3 side"), pick(R, test="robust", estimate="q capped at 3 price_ticks"))
for lab in ("rank q (normal score in maker-day)", f"main q, without {len(heavy)} cap-heavy makers"):
    add(lab.split(" (")[0] if lab.startswith("rank") else f"without cap-heavy makers", "q",
        pick(Q, version=lab, estimate="side"), pick(Q, version=lab, estimate="price_ticks"))
if rows:
    n = len(rows)
    fig, ax = plt.subplots(1, 2, figsize=(10, 0.32 * n + 1.4), sharey=True)
    y = np.arange(n)[::-1]
    for a, j, col, title, scale in ((ax[0], 2, C1, "Side: pp of P(bid) per sd of q", 100),
                                    (ax[1], 3, C2, "Price: ticks per sd of q", 1)):
        est = np.array([r[j][0] for r in rows]) * scale
        se = np.array([r[j][1] for r in rows]) * scale
        a.errorbar(est, y, xerr=1.96 * se, fmt="o", color=col, markersize=5, elinewidth=1.5, capsize=0)
        a.axvline(0, color=INK2, linewidth=0.8)
        a.set_title(title, loc="left")
        a.grid(axis="y", visible=False)
    for v_ in (-1, 1):
        ax[1].axvline(v_, color=INK2, linewidth=1, linestyle="--")
    ax[1].set_xlim(-1.3, 1.3)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([r[0] for r in rows])
    save(fig, "fig5_robustness")

# ---------------------------------------------------------------- tables (markdown, for the paper)
t1 = R[(R["test"] == "T1 pooled") & (R["estimate"] == "side")]
nm_ = f"{int(t1['makers'].iat[0])} makers, {int(t1['coin_days'].iat[0])} coin-days" if len(t1) else ""
L = ["# Part 1 tables", "", f"## Table 1. Which tool? (T1, per 1 sd of inventory q; {nm_})", "",
     "| Response | Estimate | 95% interval | p |", "|---|---|---|---|"]
for lab, est in (("Side: change in P(bid)", "side"), ("Price: shift in ticks", "price_ticks"),
                 ("Size: log size on the unwinding side", "size")):
    r = pick(R, test="T1 pooled", estimate=est)
    if r:
        L.append(f"| {lab} | {r[0]:+.4f} | [{r[0] - 1.96 * r[1]:+.4f}, {r[0] + 1.96 * r[1]:+.4f}] | {pval(r[0] / r[1]):.4f} |")
L += ["", "## Table 2. GLFT's other predictions (T2-T4)", "", "| Test | GLFT predicts | Estimate | p | Verdict |",
      "|---|---|---|---|---|"]
r = pick(R, test="T2", estimate="price q x vol")
if r:
    L.append(f"| T2 price x volatility | < 0 | {r[0]:+.4f} | {pval(r[0] / r[1]):.4f} | {'consistent' if r[0] < 0 and pval(r[0] / r[1]) < .05 else 'not as GLFT'} |")
r = pick(R, test="T3", estimate="slope |q| < 1")
if r:
    L.append(f"| T3 side slope for small inventory | ~0 | {r[0]:+.4f} | {pval(r[0] / r[1]):.4f} | {'not as GLFT' if pval(r[0] / r[1]) < .05 else 'consistent'} |")
t4 = K[(K["test"] == "T4") & K["version"].astype(str).str.startswith("finite")]
if len(t4):
    rr = t4.iloc[0]
    L.append(f"| T4 Spearman(side response, half-life) | > 0 | {rr['value']:+.3f} | {rr['p']:.4f} | "
             f"{'supported' if rr['value'] > 0 and rr['p'] < .05 else 'not supported'} ({int(rr['makers'])} makers) |")
L += ["", "## Table 3. Robustness (side | price, per 1 sd of q)", "", "| Version | Side | Price (ticks) |", "|---|---|---|"]
for lab, g, s, p_ in rows:
    L.append(f"| {lab} | {s[0]:+.4f} (se {s[1]:.4f}) | {p_[0]:+.4f} (se {p_[1]:.4f}) |")
with open(os.path.join(RES, "part1_tables.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print(f"\nSaved part1_tables.md and part1_qscale.csv in {RES}, figures in {FIG} | {(time.time() - t00) / 60:.0f} min")
