"""Day 9: group each firm's accounts into SLEEVES - accounts that trade as one unit. About 20-40 minutes.

Run:  python E:\\p3\\day9_sleeves.py > E:\\p3\\day9_output.txt          (all 5 coins)
Saves: E:\\p3\\data\\sleeves.csv   (coin, wallet, firm, sleeve, fills, quotes) - the 3 clustering rules:
       account = each wallet alone | sleeve = this file | firm = the on-chain firm label

Why: one firm can run several accounts. If two accounts move together (same engine, same signals) they
are one decision-maker, and counting them as two makers double-counts it. If they mirror each other
(one long when the other is short), each account's inventory alone is misleading.
Two accounts of the SAME firm are linked when either:
  1. their minute-by-minute trading flows are strongly related, |corr| at or above the threshold
     (positive = trade together, negative = mirror each other), or
  2. they place quotes in the same blocks far more than unrelated accounts do (same engine).
Thresholds come from accounts of DIFFERENT firms (which cannot share an engine): the 99th percentile of
cross-firm values, and at least |corr| 0.5. Linked accounts form one sleeve (chains count: A-B, B-C ->
A, B, C). Window Aug 15 - Oct 25 (the test window); accounts with at least 500 fills there.
Reported per sleeve: net_position_vol_ratio = std of the sleeve's net position CHANGE since the window
start / sum of the accounts' stds of position change (starting positions set to 0, so it measures how the
accounts' movements offset, not the volatility of their true combined inventory) (near 0 = strongly offsetting; near 1 = little offset - same direction OR independent, it cannot tell
those two apart), plus the weakest and median FLOW correlation among all pairs inside the sleeve (low when
some pairs are joined only through others or by same-block links) and how many of the sleeve's actual
links came from flow and from same blocks.
Same-block overlap is computed two ways: shared / blocks of the smaller account (the link rule) and
Jaccard = shared / all blocks of both (reported). Blocks are keyed by day + block number, and the block
sample is drawn twice (blocks ending in 0 and in 10 out of 20): a disjoint-sample stability check, not an
independent replication; each half gets its own cross-firm threshold and both are reported.
Sleeves are INFERRED coordination units, not verified legal entities. All later results are shown at three
levels: account (each wallet), sleeve (this file), firm (on-chain label) - sleeves never replace the others.
"""
import glob
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd

INV = os.environ.get("P3_INV", r"E:\p3\data\inventory")
QUOTES = os.environ.get("P3_QUOTES", r"E:\p3\data\quotes")
OUT = os.environ.get("P3_SLEEVES", r"E:\p3\data\sleeves.csv")
COINS = [c.upper() for c in sys.argv[1:]] or ["SOL", "XRP", "BTC", "ETH", "HYPE"]
DAYS = list(pd.date_range("2026-08-15", "2026-10-25").strftime("%Y-%m-%d"))
BLOCK_DAYS = ["2026-09-10", "2026-10-08"]            # same-block check on two ordinary days
BLOCK_SAMPLE = 20                                     # every 20th block (memory)
MIN_FILLS = 500
MIN_CORR = 0.5
NS = 1_000_000_000


def log(s):
    print(s, file=sys.stderr, flush=True)


def components(nodes, edges):
    parent = {n: n for n in nodes}

    def root(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    for a, b in edges:
        parent[root(a)] = root(b)
    return {n: root(n) for n in nodes}


con = duckdb.connect()
rows, thr_log = [], []
t0 = time.time()
print("SLEEVES: ACCOUNTS OF ONE FIRM THAT TRADE AS ONE UNIT (Aug 15 - Oct 25)")
for coin in COINS:
    inv = [f for d in DAYS for f in glob.glob(os.path.join(INV, coin, f"{d}.parquet"))]
    if not inv:
        print(f"\n   {coin}: no inventory files")
        continue
    # ---- minute flow per account: change in position within each minute ----
    nf = con.execute(f"""SELECT count(*) FROM (SELECT lower(wallet) FROM read_parquet({inv}, union_by_name=true)
                           GROUP BY 1 HAVING count(DISTINCT lower(firm)) > 1)""").fetchone()[0]
    if nf:
        sys.exit(f"STOP: {nf} {coin} wallets map to more than one firm")
    tcol = con.execute(f"DESCRIBE SELECT ts FROM read_parquet({inv}, union_by_name=true)").fetchone()[1]
    if "TIMESTAMP" in tcol.upper():
        tns = "epoch_ns(ts)"
    else:                                               # integer time: find its unit from its size
        mx = con.execute(f"SELECT max(ts) FROM read_parquet({inv}, union_by_name=true)").fetchone()[0]
        mult = {19: 1, 16: 1_000, 13: 1_000_000, 10: 1_000_000_000}.get(len(str(int(mx))))
        if mult is None:
            sys.exit(f"STOP: cannot tell the time unit of ts in {coin} inventory (max {mx})")
        tns = f"(ts * {mult})"
    print(f"   {coin}: inventory time column {tcol} -> {tns}")
    fl = con.execute(f"""
        SELECT lower(wallet) AS wallet, any_value(lower(firm)) AS firm,
               {tns} // {60 * NS} AS minute, sum(pos_after - pos_before) AS flow, count(*) AS n
        FROM read_parquet({inv}, union_by_name=true) GROUP BY 1, 3""").df()
    if not np.isfinite(fl["flow"]).all():
        sys.exit(f"STOP: non-finite flows in {coin}")
    fills = fl.groupby("wallet")["n"].sum()
    firm_of = fl.groupby("wallet")["firm"].first()
    act = sorted(fills[fills >= MIN_FILLS].index)
    fl = fl[fl["wallet"].isin(act)]
    grid = np.arange(pd.Timestamp(DAYS[0], tz="UTC").value // (60 * NS),
                     (pd.Timestamp(DAYS[-1], tz="UTC") + pd.Timedelta(days=1)).value // (60 * NS))
    M = fl.pivot_table(index="minute", columns="wallet", values="flow", aggfunc="sum")
    M = M.reindex(index=grid, columns=act).fillna(0.0).to_numpy(np.float64)   # no fill in a minute = 0 flow
    sd = M.std(0)
    zero_var = [w for w, v in zip(act, sd) if v == 0]
    Z = (M - M.mean(0)) / np.where(sd > 0, sd, 1)
    C = Z.T @ Z / len(Z)
    if not np.isfinite(C).all():
        sys.exit(f"STOP: non-finite correlations in {coin}")
    # ---- same-block quoting: overlap = shared blocks / blocks of the smaller account ----
    qf = [f for d in BLOCK_DAYS for f in glob.glob(os.path.join(QUOTES, coin, f"{d}*.parquet"))]

    def overlap(half):
        """shared/smaller and Jaccard for one half of the block sample (blocks keyed by day + number)."""
        O, J = np.full((len(act), len(act)), np.nan), np.full((len(act), len(act)), np.nan)
        con.execute(f"""CREATE OR REPLACE TEMP TABLE b AS
            SELECT DISTINCT aw.i, q.ts // {86400 * NS} AS day,
                   q.block_number AS blk
            FROM read_parquet({qf}, union_by_name=true) q JOIN aw ON lower(q.wallet) = aw.wallet
            WHERE q.block_number % {BLOCK_SAMPLE} = {half}""")
        nb = con.execute("SELECT i, count(*) FROM b GROUP BY 1").df().set_index("i").iloc[:, 0]
        pair = con.execute("""SELECT x.i AS a, y.i AS c, count(*) AS n FROM b x JOIN b y
                              ON x.day = y.day AND x.blk = y.blk AND x.i < y.i GROUP BY 1, 2""").df()
        have = list(nb.index)
        for i in have:
            for j in have:
                if i != j:
                    O[i, j] = J[i, j] = 0.0
        for r in pair.itertuples():
            na, nc = nb[r.a], nb[r.c]
            O[r.a, r.c] = O[r.c, r.a] = r.n / min(na, nc)
            J[r.a, r.c] = J[r.c, r.a] = r.n / (na + nc - r.n)
        return O, J

    if qf:
        con.register("aw", pd.DataFrame({"wallet": act, "i": range(len(act))}))
        O, J = overlap(0)
        O2, _ = overlap(BLOCK_SAMPLE // 2)               # the other, disjoint half: stability check
    else:
        O = J = O2 = np.full((len(act), len(act)), np.nan)
    # ---- thresholds from cross-firm pairs ----
    fa = np.array([firm_of[w] for w in act])
    iu = np.triu_indices(len(act), 1)
    same = fa[iu[0]] == fa[iu[1]]
    c_thr = max(MIN_CORR, float(np.quantile(np.abs(C[iu][~same]), 0.99))) if (~same).any() else MIN_CORR
    ov = O[iu]
    o_cross = ov[~same & ~np.isnan(ov)]
    o_thr = float(np.quantile(o_cross, 0.99)) if len(o_cross) else np.inf
    by_flow = same & (np.abs(C[iu]) >= c_thr)
    by_blk = same & (np.nan_to_num(ov, nan=-1) > o_thr)
    o_cross2 = O2[iu][~same & ~np.isnan(O2[iu])]
    o_thr2 = float(np.quantile(o_cross2, 0.99)) if len(o_cross2) else np.inf
    by_blk2 = same & (np.nan_to_num(O2[iu], nan=-1) > o_thr2)
    link = by_flow | by_blk
    edges = [(act[a], act[b]) for a, b in zip(iu[0][link], iu[1][link])]
    comp = components(act, edges)
    sl = pd.DataFrame({"wallet": act, "firm": fa, "root": [comp[w] for w in act],
                       "fills": [int(fills[w]) for w in act]})
    sl["sleeve"] = coin + "_s" + sl.groupby("root").ngroup().astype(str).str.zfill(3)
    # how much the accounts in a sleeve cancel each other
    pos = M.cumsum(0)                                    # position change since window start (per minute)
    idx = {w: i for i, w in enumerate(act)}
    cancel = {}
    for s, g in sl.groupby("sleeve"):
        if len(g) > 1:
            cols = [idx[w] for w in g["wallet"]]
            cancel[s] = pos[:, cols].sum(1).std() / max(pos[:, cols].std(0).sum(), 1e-12)
    sl["net_position_vol_ratio"] = sl["sleeve"].map(cancel)
    # weakest / median direct link inside each multi-account sleeve (flow |corr|)
    wk, md = {}, {}
    for s_, g in sl.groupby("sleeve"):
        if len(g) > 1:
            ii = [idx[w] for w in g["wallet"]]
            v = np.abs(C[np.ix_(ii, ii)][np.triu_indices(len(ii), 1)])
            wk[s_], md[s_] = float(v.min()), float(np.median(v))
    sl["min_pair_flow_corr"], sl["median_pair_flow_corr"] = sl["sleeve"].map(wk), sl["sleeve"].map(md)
    ef, eb = {}, {}
    for a_, b_, f_, k_ in zip(iu[0][link], iu[1][link], by_flow[link], by_blk[link]):
        s_ = sl.at[a_, "sleeve"]
        ef[s_], eb[s_] = ef.get(s_, 0) + int(f_), eb.get(s_, 0) + int(k_)
    sl["links_by_flow"], sl["links_by_blocks"] = sl["sleeve"].map(ef), sl["sleeve"].map(eb)
    rows.append(sl.assign(coin=coin).drop(columns="root"))
    thr_log.append(dict(coin=coin, start=DAYS[0], end=DAYS[-1], accounts=len(act), corr_threshold=c_thr,
                        overlap_threshold=o_thr, overlap_threshold_half2=o_thr2, block_days="|".join(BLOCK_DAYS), block_sample=BLOCK_SAMPLE,
                        zero_variance_accounts=len(zero_var)))
    multi = sl.groupby("sleeve").filter(lambda g: len(g) > 1)
    print(f"\n   {coin}: {len(act)} active accounts (>= {MIN_FILLS} fills) in {sl['firm'].nunique()} firms "
          f"-> {sl['sleeve'].nunique()} sleeves | {time.time() - t0:,.0f}s")
    per_firm = sl.groupby("firm").size()
    print(f"      active accounts per firm: median {per_firm.median():.0f}, max {per_firm.max()} | "
          f"zero-flow accounts: {len(zero_var)}")
    print(f"      flow |corr|: same-firm median {np.median(np.abs(C[iu][same])) if same.any() else np.nan:.2f}, "
          f"cross-firm 99th pct {np.quantile(np.abs(C[iu][~same]), .99) if (~same).any() else np.nan:.2f} "
          f"-> threshold {c_thr:.2f}")
    print(f"      same-block overlap (shared/smaller): cross-firm 99th pct -> threshold {o_thr:.2f} "
          f"({len(o_cross):,} cross-firm pairs) | Jaccard of linked pairs: median "
          f"{np.nanmedian(J[iu][by_blk]) if by_blk.any() else np.nan:.2f}")
    print(f"      same-firm pairs: {int(same.sum()):,} | linked {len(edges)}: flow only "
          f"{int((by_flow & ~by_blk).sum())}, blocks only {int((by_blk & ~by_flow).sum())}, both "
          f"{int((by_flow & by_blk).sum())}")
    print(f"      disjoint-sample stability: half 1 threshold {o_thr:.2f} -> {int(by_blk.sum())} block links | "
          f"half 2 threshold {o_thr2:.2f} -> {int(by_blk2.sum())} | found in both {int((by_blk & by_blk2).sum())}")
    print(f"      multi-account sleeves: {multi['sleeve'].nunique()} holding {len(multi)} accounts "
          f"({multi['fills'].sum() / sl['fills'].sum():.0%} of fills)")
    top = multi.groupby("sleeve").agg(n=("wallet", "size"), firm=("firm", "first"), fills=("fills", "sum"),
                                      ratio=("net_position_vol_ratio", "first"), wk=("min_pair_flow_corr", "first"),
                                      md=("median_pair_flow_corr", "first"), lf=("links_by_flow", "first"),
                                      lb=("links_by_blocks", "first")).sort_values("fills", ascending=False)
    for s_, r in top.head(8).iterrows():
        kind = "strongly offsetting" if r.ratio < 0.3 else ("little offset" if r.ratio > 0.7 else "partly offsetting")
        print(f"         {s_}: {r.n:>2} accounts, firm {r.firm[:8]}…, {r.fills:>10,} fills | net position vol ratio "
              f"{r.ratio:.2f} ({kind}) | flow |corr| over all pairs min {r.wk:.2f}, median {r.md:.2f} | links: "
              f"flow {int(r.lf)}, blocks {int(r.lb)}")
res = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
if not len(res):
    sys.exit("STOP: no inventory files for any coin - nothing written")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
res[["coin", "wallet", "firm", "sleeve", "fills", "net_position_vol_ratio", "min_pair_flow_corr",
     "median_pair_flow_corr", "links_by_flow", "links_by_blocks"]].to_csv(OUT, index=False)
pd.DataFrame(thr_log).to_csv(OUT.replace(".csv", "_thresholds.csv"), index=False)
print(f"\nSaved {OUT} and {OUT.replace('.csv', '_thresholds.csv')}  ({time.time() - t0:,.0f} seconds)")
print("Firm and wallet names stay private: only the first 8 characters are printed.")
