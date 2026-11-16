"""Resting quotes: rebuild each maker's resting orders every 10 s and relate the bid share to inventory.
Not pre-registered."""
import glob
import os
import sys
import time
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

EXPORT = [p for p in os.environ.get("P3_EXPORT", "E:/hl_export;E:/hl_export/delivery-2").split(";") if p]
BOOK = os.environ.get("P3_BOOK", r"E:\hl_data\book")
INV = os.environ.get("P3_INV", r"E:\p3\data\inventory")
PART1 = os.environ.get("P3_PART1", r"E:\p3\part1")
OUT = os.environ.get("P3_RESULTS", r"E:\p3\results")
CACHE = os.path.join(OUT, "resting_cache_v6")      # new folder: nothing from an earlier version is reused
COINS = os.environ.get("P3_COINS", "SOL,XRP,BTC,ETH,HYPE").split(",")
DAYS = list(pd.date_range("2026-06-01", "2026-06-14").strftime("%Y-%m-%d")) + \
       list(pd.date_range("2026-08-15", "2026-10-25").strftime("%Y-%m-%d"))
DAYS = os.environ.get("P3_DAYS", ",".join(DAYS)).split(",")
BOOK_GAP_DAYS = {"2026-06-07", "2026-06-08", "2026-06-09", "2026-06-14"}
GAPS = [("2026-06-05 22:00:10", "2026-06-05 22:24:16"), ("2026-08-29 20:21:00", "2026-08-29 20:32:00"),
        ("2026-10-25 09:32:20", "2026-10-25 10:28:57")]
STEP_S, BAND_BP, CAP, MIN_QUOTES, PRE_S, MID_AGE_S = 10, 5.0, 5.0, 1000, 86400, 1.0   # PRE_S: whole previous day
NS = 1_000_000_000
MIN_NS = 60 * NS
t00 = time.time()
con = duckdb.connect()
con.execute("SET TimeZone = 'UTC'")
os.makedirs(os.path.join(OUT, "duck_tmp"), exist_ok=True)
con.execute(f"SET temp_directory = '{os.path.join(OUT, 'duck_tmp')}'")   # spill to disk instead of swapping
con.execute(f"SET memory_limit = '{os.environ.get('P3_MEM', '3GB')}'")
con.execute("SET preserve_insertion_order = false")
os.makedirs(CACHE, exist_ok=True)


def log(s):
    print(s, file=sys.stderr, flush=True)


def prev_day(d):
    return (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


gap_ns = [(pd.Timestamp(a, tz="UTC").value - 600 * NS, pd.Timestamp(b, tz="UTC").value + 600 * NS) for a, b in GAPS]

# ---------------------------------------------------------------- the T1 makers (same rule as day21)
keep = defaultdict(set)
for c in COINS:
    cnt = defaultdict(int)
    for f in glob.glob(os.path.join(PART1, c, "*.parquet")):
        if pq.ParquetFile(f).metadata.num_rows == 0:
            continue
        for w, n in pd.read_parquet(f, columns=["wallet"])["wallet"].value_counts().items():
            cnt[w] += int(n)
    keep[c] = {w for w, n in cnt.items() if n >= MIN_QUOTES}
log("makers: " + ", ".join(f"{c} {len(keep[c])}" for c in COINS))


def mids(coin, d, grid):
    f = os.path.join(BOOK, coin, f"{coin}_{d}.parquet")
    if d in BOOK_GAP_DAYS or not os.path.exists(f):
        return None
    b = pd.read_parquet(f)
    tc = next(c for c in ("time", "timestamp", "ts", "local_timestamp") if c in b.columns)
    t = (pd.to_datetime(b[tc], utc=True, format="ISO8601") - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(nanoseconds=1)
    m = pd.DataFrame({"bt": t.astype("int64"), "mid": (b["bid_px"].astype(float) + b["ask_px"].astype(float)) / 2,
                      "ok": (b["ask_px"].astype(float) > b["bid_px"].astype(float)) & (b["bid_px"].astype(float) > 0)
                      & np.isfinite(b["bid_px"].astype(float)) & np.isfinite(b["ask_px"].astype(float))})
    m = m[m["ok"]].sort_values("bt")
    g = pd.merge_asof(pd.DataFrame({"t": grid}), m[["bt", "mid"]], left_on="t", right_on="bt", direction="backward")
    g = g[(g["t"] - g["bt"]) <= MID_AGE_S * NS]
    return g[["t", "mid"]]


COLS_SHOWN = False


def order_day(coin, d, wallets, pend):
    """Read ONE day's order file once."""
    of = [f"{r}/orders/{coin}/{d}.parquet" for r in EXPORT if os.path.exists(f"{r}/orders/{coin}/{d}.parquet")]
    if not of:
        return None
    cols = {c[0] for c in con.execute(f"DESCRIBE SELECT * FROM read_parquet({of}, union_by_name=true)").fetchall()}
    trig = "AND NOT COALESCE(CAST(is_trigger AS BOOLEAN), FALSE)" if "is_trigger" in cols else ""
    wl = sorted(wallets)
    con.register("wk", pd.DataFrame({"w": wl, "wid": np.arange(len(wl), dtype=np.int16)}))
    con.register("pend", pend)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ev AS
            SELECT oid, st, ts, wid, side, price, size FROM (
                SELECT CAST(oid AS BIGINT) AS oid, epoch_ns(timestamp) AS ts,
                       CAST(CASE WHEN CAST(status AS VARCHAR) = 'open' THEN 0
                                 WHEN lower(CAST(status AS VARCHAR)) LIKE '%fill%' THEN 1 ELSE 2 END AS TINYINT) AS st,
                       CASE WHEN CAST(status AS VARCHAR) = 'open' THEN wk.wid END AS wid,
                       CASE WHEN CAST(status AS VARCHAR) = 'open' THEN CAST(side AS VARCHAR) END AS side,
                       CASE WHEN CAST(status AS VARCHAR) = 'open' THEN CAST(limit_price AS DOUBLE) END AS price,
                       CASE WHEN CAST(status AS VARCHAR) = 'open' THEN CAST(orig_size AS DOUBLE) END AS size,
                       (CAST(status AS VARCHAR) <> 'open') OR (CAST(tif AS VARCHAR) = 'Alo' AND CAST(order_type AS VARCHAR) = 'Limit'
                        AND NOT COALESCE(CAST(reduce_only AS BOOLEAN), FALSE) {trig}) AS keep_row
                FROM read_parquet({of}, union_by_name=true) JOIN wk ON lower(user_address) = wk.w)
            WHERE keep_row""")
    # compact in memory: status as a small code
    # only the columns needed, and opens that are not post-only quotes dropped while reading
    global COLS_SHOWN
    if not COLS_SHOWN:
        log("   order-file columns: " + ", ".join(sorted(cols)))
        st = con.execute("SELECT CASE st WHEN 0 THEN 'open' WHEN 1 THEN 'filled' ELSE 'other end' END, count(*) "
                         "FROM ev GROUP BY 1 ORDER BY 2 DESC").fetchall()
        log("   order statuses (T1 makers' post-only quotes and all end events, first file): " + ", ".join(f"{a} {b:,}" for a, b in st))
        COLS_SHOWN = True
    con.execute("""
        CREATE OR REPLACE TEMP TABLE iv AS
        WITH op0 AS (SELECT oid, arg_min(wid, ts) AS wid, arg_min(side, ts) AS side, arg_min(price, ts) AS price,
                           arg_min(size, ts) AS size, min(ts) AS t_open, count(DISTINCT wid) AS n_w
                    FROM ev WHERE st = 0
                      AND oid NOT IN (SELECT oid FROM pend)
                    GROUP BY oid),
             op AS (SELECT op0.*, wk.w FROM op0 JOIN wk ON op0.wid = wk.wid),
             al AS (SELECT oid, w, side, price, size, t_open, t_end0, ef0 FROM pend
                    UNION ALL
                    SELECT oid, w, side, price, size, t_open, NULL::BIGINT AS t_end0, FALSE AS ef0 FROM op
                    WHERE n_w = 1 AND price > 0 AND size > 0 AND side IN ('B', 'A')),
             en AS (SELECT e.oid, min(e.ts) AS t_end, arg_min(e.st, e.ts) AS end_st
                    FROM ev e JOIN al a ON e.oid = a.oid
                    WHERE e.st <> 0 AND e.ts >= a.t_open GROUP BY e.oid)
        SELECT al.oid, al.w, al.side, al.price, al.size, al.t_open,
               COALESCE(al.t_end0, en.t_end) AS t_end,
               CASE WHEN al.t_end0 IS NOT NULL THEN al.ef0
                    ELSE COALESCE(en.end_st = 1, FALSE) END AS ends_filled
        FROM al LEFT JOIN en ON al.oid = en.oid""")
    return True


def next_pend(d):
    """Orders still resting at the next midnight (their end, if already seen in this file, is kept)."""
    E = pd.Timestamp(d, tz="UTC").value + 86400 * NS
    return con.execute(f"""SELECT oid, w, side, price, size, t_open, t_end AS t_end0, ends_filled AS ef0 FROM iv
                           WHERE t_end IS NULL OR t_end > {E}""").df()


def resting(d, G):
    """Per maker and snapshot: resting bid / ask notional within the band, from table 'iv'."""
    D0 = pd.Timestamp(d, tz="UTC").value
    S, E = STEP_S * NS, D0 + 86400 * NS
    g = G.assign(k=(G["t"] - D0) // S)
    g["b"] = g["k"] // 6                                          # 1-minute blocks of six snapshots
    eps = BAND_BP / 1e4
    blk = g.groupby("b")["mid"].agg(["min", "max"]).reset_index()
    blk["lo"], blk["hi"] = blk["min"] * (1 - eps) * (1 - 1e-9), blk["max"] * (1 + eps) * (1 + 1e-9)
    con.register("g", g)
    con.register("blk", blk[["b", "lo", "hi"]])
    # An order can be in the band in a minute only if its price is within the band of that minute's lowest or
    # highest mid; other minutes are skipped before the per-snapshot check (the result is exactly the same,
    # it just avoids expanding deep orders that rest far from the mid for hours).
    return con.execute(f"""
        WITH x AS (SELECT w, side, price, price * size AS notl, t_open, ends_filled,
                          GREATEST(0, CAST(ceil((t_open - {D0}) / {S}::DOUBLE) AS BIGINT)) AS k_lo,
                          LEAST(8639, CAST(floor((LEAST(COALESCE(t_end, {E}), {E}) - 1 - {D0}) / {S}::DOUBLE) AS BIGINT)) AS k_hi
                   FROM iv),
             xb AS (SELECT *, unnest(range(k_lo // 6, k_hi // 6 + 1)) AS b FROM x WHERE k_hi >= k_lo),
             xm AS (SELECT xb.* FROM xb JOIN blk ON xb.b = blk.b WHERE xb.price BETWEEN blk.lo AND blk.hi),
             ex AS (SELECT w, side, price, notl, t_open, ends_filled,
                           unnest(range(GREATEST(k_lo, b * 6), LEAST(k_hi, b * 6 + 5) + 1)) AS k
                    FROM xm)
        SELECT ex.w AS wallet, g.t AS t,
               sum(CASE WHEN ex.side = 'B' THEN ex.notl ELSE 0 END) AS bid_not,
               sum(CASE WHEN ex.side = 'A' THEN ex.notl ELSE 0 END) AS ask_not,
               sum(CASE WHEN ex.side = 'B' THEN 1 ELSE 0 END) AS bid_n,
               sum(CASE WHEN ex.side = 'A' THEN 1 ELSE 0 END) AS ask_n,
               sum(CASE WHEN g.t - ex.t_open > {3600 * NS} THEN 1 ELSE 0 END) AS age1h_n,
               sum(CASE WHEN g.t - ex.t_open > {6 * 3600 * NS} THEN 1 ELSE 0 END) AS age6h_n,
               sum(CASE WHEN g.t - ex.t_open > {12 * 3600 * NS} THEN 1 ELSE 0 END) AS age12h_n,
               sum(CASE WHEN g.t - ex.t_open > {24 * 3600 * NS} THEN 1 ELSE 0 END) AS age24h_n,
               sum(CASE WHEN ex.ends_filled THEN 1 ELSE 0 END) AS endfill_n
        FROM ex JOIN g ON ex.k = g.k
        WHERE abs(ex.price - g.mid) / g.mid * 1e4 <= {BAND_BP}
        GROUP BY 1, 2""").df()



def full_grid(R, wallets, G):
    """Every T1 maker at every valid snapshot; snapshots without a resting order in the band get zeros."""
    base = pd.MultiIndex.from_product([sorted(wallets), G["t"].to_numpy()], names=["wallet", "t"]).to_frame(index=False)
    out = base.merge(R, on=["wallet", "t"], how="left")
    for c in ("bid_not", "ask_not", "bid_n", "ask_n"):
        out[c] = out[c].fillna(0.0)
    return out


def q_at(coin, d, R):
    """Frozen q at each (wallet, snapshot): position / std of minute position over the previous 1,440 minutes."""
    invf = [f for f in (os.path.join(INV, coin, f"{x}.parquet") for x in (prev_day(d), d)) if os.path.exists(f)]
    if not invf:
        return None
    tcol = con.execute(f"DESCRIBE SELECT ts FROM read_parquet({invf}, union_by_name=true)").fetchone()[1]
    T = "epoch_ns(ts)" if "TIMESTAMP" in tcol.upper() else "ts"
    fl = con.execute(f"""SELECT lower(wallet) AS wallet, CAST({T} AS BIGINT) AS ts, pos_after
                         FROM read_parquet({invf}, union_by_name=true)""").df()
    fl = fl[fl["wallet"].isin(set(R["wallet"]))]
    if fl.empty:
        return None
    D0 = pd.Timestamp(d, tz="UTC").value // MIN_NS
    pm = fl.assign(minute=fl["ts"] // MIN_NS).sort_values("ts").groupby(["minute", "wallet"])["pos_after"].last()
    P = pm.unstack("wallet")
    P = P.reindex(np.arange(min(P.index.min(), D0 - 1440), D0 + 1440)).ffill()
    S = P.rolling(1440, min_periods=1440).std().shift(1).loc[D0:D0 + 1439]
    sd = S.stack().rename("sd").reset_index().rename(columns={"level_0": "minute"})
    sd = sd[sd["sd"] > 0]
    pos = pd.merge_asof(R.sort_values("t"), fl.sort_values("ts")[["ts", "wallet", "pos_after"]],
                        left_on="t", right_on="ts", by="wallet", direction="backward")
    pos = pos.assign(minute=pos["t"] // MIN_NS).merge(sd, on=["minute", "wallet"], how="left")
    pos["q"] = (pos["pos_after"] / pos["sd"]).clip(-CAP, CAP)
    return pos.dropna(subset=["q"])[["wallet", "t", "q", "bid_not", "ask_not", "bid_n", "ask_n"]]


# ---------------------------------------------------------------- build (cached per coin-day)
# Order files are read in calendar order, each ONCE
# day, so an order is seen however long ago it was opened
# reset (logged)
EMPTY = pd.DataFrame({"oid": pd.Series(dtype="int64"), "w": pd.Series(dtype="str"), "side": pd.Series(dtype="str"),
                      "price": pd.Series(dtype="float64"), "size": pd.Series(dtype="float64"),
                      "t_open": pd.Series(dtype="int64"), "t_end0": pd.Series(dtype="Int64"),
                      "ef0": pd.Series(dtype="bool")})
CAL = list(pd.date_range(prev_day(min(DAYS)), max(DAYS)).strftime("%Y-%m-%d"))
for c in COINS:
    if not keep[c]:
        continue
    pend = EMPTY
    for d in CAL:
        pf = os.path.join(CACHE, f"pend_{c}_{d}.parquet")
        cf = os.path.join(CACHE, f"{c}_{d}.parquet")
        if os.path.exists(pf):                                    # this day is finished: resume from its state
            pend = pd.read_parquet(pf)
            continue
        t0 = time.time()
        if order_day(c, d, keep[c], pend) is None:
            if len(pend):
                log(f"   {c} {d}: no order file - {len(pend):,} carried orders dropped (state unknown)")
            pend = EMPTY
            if d in DAYS:
                log(f"   {c} {d}: MISSING order file - not cached, will be retried")
            continue
        t_read = time.time() - t0
        done = True
        if d in DAYS and d not in BOOK_GAP_DAYS and not os.path.exists(cf):
            D0 = pd.Timestamp(d, tz="UTC").value
            grid = D0 + np.arange(0, 86400, STEP_S, dtype=np.int64) * NS
            grid = grid[~np.any([(grid >= a) & (grid <= b) for a, b in gap_ns], axis=0)]
            G = mids(c, d, grid)
            if not os.path.exists(os.path.join(INV, c, f"{prev_day(d)}.parquet")):
                log(f"   {c} {d}: no previous-day inventory, so q is undefined (same as T1) - skipped")
            elif G is None or not len(G):
                log(f"   {c} {d}: MISSING book file or no usable mid - not cached, will be retried")
                done = False
            else:
                R = resting(d, G)
                out = q_at(c, d, full_grid(R, keep[c], G))
                if out is None:
                    log(f"   {c} {d}: MISSING inventory file - not cached, will be retried")
                    done = False
                else:
                    dg = pd.DataFrame([{k: float(R[k].sum()) for k in
                                        ("bid_n", "ask_n", "age1h_n", "age6h_n", "age12h_n", "age24h_n", "endfill_n")}])
                    dg.to_csv(os.path.join(CACHE, f"diag_{c}_{d}.csv"), index=False)
                    out.to_parquet(cf + ".part", index=False)
                    os.replace(cf + ".part", cf)
                    log(f"   {c} {d}: {len(out):,} maker-snapshots | {time.time() - t0:,.0f}s (reading {t_read:,.0f}s) | "
                        f"total {(time.time() - t00) / 60:,.0f} min")
        pend = next_pend(d)
        if done:
            pend.to_parquet(pf + ".part", index=False)
            os.replace(pf + ".part", pf)


# ---------------------------------------------------------------- estimate
stats = defaultdict(dict)
bins = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
EDGES = [-5.01, -2, -1, -0.25, 0.25, 1, 2, 5.01]
n_rows = defaultdict(int)
for c in COINS:
    for f in sorted(glob.glob(os.path.join(CACHE, f"{c}_*.parquet"))):
        R = pd.read_parquet(f)
        if R.empty:
            continue
        day = os.path.basename(f)[len(c) + 1:len(c) + 11]
        if day not in DAYS:                                       # P3_DAYS defines the sample
            continue
        R["Y2"] = (R["bid_not"] > 0).astype(float)
        R["Y3"] = (R["ask_not"] > 0).astype(float)
        tot = R["bid_not"] + R["ask_not"]
        R["Y1"] = np.where(tot > 0, R["bid_not"] / tot.where(tot > 0), np.nan)
        totn = R["bid_n"] + R["ask_n"]
        R["Y1c"] = np.where(totn > 0, R["bid_n"] / totn.where(totn > 0), np.nan)
        n_rows[c] += int((tot > 0).sum())
        for w, m in R.groupby("wallet", sort=False):
            for y in ("Y1", "Y1c", "Y2", "Y3"):
                mm = m[np.isfinite(m[y].to_numpy(float))]
                if len(mm) < 2:
                    continue
                x = mm["q"].to_numpy(float)
                xd = x - x.mean()
                v = mm[y].to_numpy(float)
                stats[y][(c, w, day)] = (float(xd @ xd), float(xd @ (v - v.mean())), len(v))
            m = m[np.isfinite(m["Y1"].to_numpy(float))]
            x = m["q"].to_numpy(float)
            b = np.digitize(x, EDGES) - 1
            for k in range(len(EDGES) - 1):
                s = b == k
                if s.any():
                    bb = bins[(c, w)][k]
                    bb[0] += float(m["Y1"].to_numpy()[s].sum()); bb[1] += int(s.sum())


def pooled(y, keep_fn=lambda c, w, d: True, cluster="coin_day"):
    S = {k: v for k, v in stats[y].items() if keep_fn(*k) and v[0] > 0}
    if not S:
        return None
    n_in = defaultdict(int)
    for k, v in S.items():
        n_in[k[:2]] += v[2]
    W = {mk: 1.0 / n for mk, n in n_in.items()}
    A = sum(W[k[:2]] * v[0] for k, v in S.items())
    B = sum(W[k[:2]] * v[1] for k, v in S.items())
    beta = B / A
    sc = defaultdict(float)
    for (c, w, d), (xx, xy, n) in S.items():
        sc[(c, d) if cluster == "coin_day" else d] += W[(c, w)] * (xy - xx * beta)
    G = len(sc)
    se = np.sqrt(sum(v * v for v in sc.values()) / A ** 2 * G / max(G - 1, 1))
    return beta, se, G, len(n_in), sum(v[2] for v in S.values())


def pval(t):
    from math import erf, sqrt
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


rows = []
print("RESTING-QUOTE CHECK - do resting quotes shift away from the accumulating side?  [robustness of T1; reported either way]")
print("   maker-snapshots (10 s) with a resting order within 5 bp: " + ", ".join(f"{c} {n_rows[c]:,}" for c in COINS))
for y, lab, pred in (("Y1", "Y1 resting bid share (PRIMARY)", "< 0"),
                     ("Y1c", "Y1c same, by order count", "< 0"),
                     ("Y2", "Y2 has a resting bid (all snapshots)", "< 0"),
                     ("Y3", "Y3 has a resting ask (all snapshots)", "> 0")):
    r = pooled(y)
    if r is None:
        print(f"   {lab}: no data")
        continue
    b, se, G, nm, nq = r
    p = pval(b / se)
    print(f"   {lab:<34} {b:+.4f} per sd of q (se {se:.4f}, 95% [{b - 1.96 * se:+.4f}, {b + 1.96 * se:+.4f}], p {p:.4f}) "
          f"| predicted {pred} | {nm} makers, {nq:,} snapshots, {G} coin-days")
    rows.append(dict(outcome=y, sample="all", value=b, se=se, p=p, makers=nm, snapshots=nq, coin_days=G))
    if y == "Y1":
        rd = pooled("Y1", cluster="day")
        if rd is not None:
            print(f"   {'':<34} day-clustered: se {rd[1]:.4f}, p {pval(rd[0] / rd[1]):.4f} ({rd[2]} days)")
        print(f"\n   VERDICT (b < 0, p < .05): "
              f"{'SUPPORTED - resting liquidity also shifts away from the accumulating side' if b < 0 and p < 0.05 else 'NOT SUPPORTED'}")
        print(f"   (for comparison: new placements, prereg-v3 T1, -0.0620 per sd of q - a different object)\n")
print("   each coin, Y1:")
for c in COINS:
    r = pooled("Y1", (lambda cc: lambda c, w, d: c == cc)(c))
    if r is not None:
        print(f"      {c:<5} {r[0]:+.4f} (p {pval(r[0] / r[1]):.3f}) | {r[3]} makers")
        rows.append(dict(outcome="Y1", sample=c, value=r[0], se=r[1], p=pval(r[0] / r[1]), makers=r[3],
                         snapshots=r[4], coin_days=r[2]))
print("   resting bid share by inventory (every maker weighted equally):")
labs = ["q < -2", "-2..-1", "-1..-0.25", "about 0", "0.25..1", "1..2", "q > 2"]
for k, lab in enumerate(labs):
    v = [bb[k][0] / bb[k][1] for bb in bins.values() if bb[k][1] >= 30]
    if v:
        print(f"      {lab:<10} {np.mean(v):.3f}  ({len(v)} makers)")
# ---------------------------------------------------------------- order-age and partial-fill diagnostics
D = [pd.read_csv(f) for c in COINS for d in DAYS for f in [os.path.join(CACHE, f"diag_{c}_{d}.csv")] if os.path.exists(f)]
if D:
    D = pd.concat(D).sum()
    n = D["bid_n"] + D["ask_n"]
    print("\n   CHECKS ON THE RECONSTRUCTION (resting order-snapshots within 5 bp):")
    print(f"      order age > 1 h: {D['age1h_n'] / n:.4%} | > 6 h: {D['age6h_n'] / n:.4%} | > 12 h: {D['age12h_n'] / n:.4%}"
          f" | > 24 h: {D['age24h_n'] / n:.4%}")
    print("      (orders are carried from the start of the data, so no order is missed because it is old)")
    print(f"      share whose order ends 'filled': {D['endfill_n'] / n:.2%}  (only these, plus fill-then-cancel, can be partly filled;")
    print("       Y1c counts orders and is unaffected by partial fills)")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "part1_resting.csv"), index=False)
print(f"\nSaved part1_resting.csv in {OUT} | {(time.time() - t00) / 60:.0f} min")
