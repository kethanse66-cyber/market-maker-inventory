"""June only: do makers move quote prices with inventory, and how far from the best?"""
import glob
import os

import duckdb
import numpy as np
import pandas as pd

QUOTES = os.environ.get("P3_QUOTES", r"E:\p3\data\quotes")
JUNE_GOOD = ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-06",
             "2026-06-10", "2026-06-11", "2026-06-12", "2026-06-13"]
BANDS = [("within 2 ticks (current)", None), ("within 2 bp of own best", 2), ("within 5 bp", 5),
         ("within 10 bp", 10), ("within 25 bp", 25)]
NS = 1_000_000_000
CLIP = 5.0                         # q capped at +-5 std (tiny past-24h std -> huge q otherwise)


def fit(y, x, g, w, cl):
    gi = np.unique(g, return_inverse=True)[1]
    sw = np.bincount(gi, w)
    yd = y - (np.bincount(gi, w * y) / sw)[gi]
    xd = x - (np.bincount(gi, w * x) / sw)[gi]
    a = np.sum(w * xd * xd)
    b = np.sum(w * xd * yd) / a
    e = yd - b * xd
    ci = np.unique(cl, return_inverse=True)[1]
    s = np.bincount(ci, w * xd * e)
    G = ci.max() + 1
    return b, np.sqrt(np.sum(s ** 2) * G / (G - 1)) / a


con = duckdb.connect()
print("JUNE: INVENTORY CHANNELS BY DISTANCE FROM THE BEST PRICE")
for coin in ("SOL", "XRP"):
    files = sorted(f for d in JUNE_GOOD for f in glob.glob(os.path.join(QUOTES, coin, f"{d}*.parquet")))
    allf = sorted(glob.glob(os.path.join(QUOTES, coin, "2026-06-*.parquet")))
    cols = set(con.execute(f"DESCRIBE SELECT * FROM read_parquet({files}, union_by_name=true)").df()["column_name"])
    good = "book_quality = 'GOOD'" if "book_quality" in cols else "hour_crossed IS NOT NULL AND NOT poor_book"
    df = con.execute(f"""
        WITH m AS (SELECT wallet, ts // {60 * NS} AS minute, arg_max(inventory, ts) AS inv
                   FROM read_parquet({allf}, union_by_name=true) WHERE inventory IS NOT NULL GROUP BY 1, 2),
             s AS (SELECT wallet, minute, stddev_samp(inv) OVER w AS sd, count(inv) OVER w AS n
                   FROM m WINDOW w AS (PARTITION BY wallet ORDER BY minute
                                       RANGE BETWEEN 1440 PRECEDING AND 1 PRECEDING)),
             q AS (SELECT wallet, side, ts, rel_mid_bp, inventory, ticks_behind_best * tick / mid * 1e4 AS behind_bp,
                          ticks_behind_best, ts // {60 * NS} AS minute
                   FROM read_parquet({files}, union_by_name=true)
                   WHERE NOT in_gap AND {good} AND NOT crosses_book AND mid IS NOT NULL
                         AND inventory IS NOT NULL AND hash(oid) % 4 = 0)
        SELECT dense_rank() OVER (ORDER BY q.wallet) - 1 AS maker, (q.side = 'B')::INT AS bid, q.ts,
               q.rel_mid_bp, q.behind_bp, q.ticks_behind_best, q.inventory / s.sd AS q
        FROM q JOIN s USING (wallet, minute) WHERE s.n >= 30 AND s.sd > 0""").df()
    df["hour"] = df["ts"] // (3600 * NS)
    df["day"] = df["ts"] // (86400 * NS)
    aq = df["q"].abs()
    print(f"\n   {coin}: {len(df):,} sampled quotes, {df['maker'].nunique()} makers | |q| median {aq.median():.2f}, "
          f"99% {aq.quantile(.99):.1f}, 99.9% {aq.quantile(.999):.1f}, max {aq.max():,.0f} | "
          f"|q| > 5 in {(aq > 5).mean():.2%}")
    df["q_raw"] = df["q"]
    df["q"] = df["q"].clip(-CLIP, CLIP)
    print("      band                          quotes   share     b (bp per 1 std)   t      side (bid share)   t")
    for name, bp in BANDS:
        d = df[df["ticks_behind_best"].round(6) <= 2] if bp is None else df[df["behind_bp"] <= bp]
        w = 1.0 / d.groupby("maker")["maker"].transform("size").to_numpy()
        y, x = d["rel_mid_bp"].to_numpy(), d["q"].to_numpy()
        bb, bs = fit(y, x, (d["maker"] * 2 + d["bid"]).to_numpy() * 10**7 + d["hour"].to_numpy(), w,
                     d["day"].to_numpy())
        sb, ss = fit(d["bid"].to_numpy().astype(float), x, d["maker"].to_numpy() * 10**7 + d["hour"].to_numpy(),
                     w, d["day"].to_numpy())
        print(f"      {name:<28} {len(d):>10,} {len(d) / len(df):>6.0%}   {bb:>+10.4f}  {bb / bs:>+7.1f}   "
              f"{sb:>+14.4f}  {sb / ss:>+7.1f}")
    print("      unwinding by side, per maker (q capped at 5; corr of minute bid share with q):")
    mn = df.assign(m=df["ts"] // (60 * NS)).groupby(["maker", "m"]).agg(bs=("bid", "mean"), q=("q", "mean"),
                                                                         n=("bid", "size"))
    mn = mn[mn["n"] >= 5].reset_index()
    cm = mn.groupby("maker").apply(lambda g: g["bs"].corr(g["q"]) if len(g) > 100 else np.nan).dropna()
    print(f"         {len(cm)} makers: median corr {cm.median():+.2f} | below -0.2: {(cm < -0.2).mean():.0%} "
          f"| near 0 (|corr| < 0.05): {(cm.abs() < 0.05).mean():.0%}")
print("\n   q capped at +-5 for all rows above. t uses SEs clustered by day (only 10 days, so treat |t| < 3 as noise).")
