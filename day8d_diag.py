"""June only: check the inventory join behind the near-zero response.
Usage: python day8d_diag.py SOL XRP BTC"""
import glob
import os
import sys

import duckdb
import numpy as np
import pandas as pd

QUOTES = os.environ.get("P3_QUOTES", r"E:\p3\data\quotes")
JUNE_GOOD = ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-06",
             "2026-06-10", "2026-06-11", "2026-06-12", "2026-06-13"]
COINS = [c.upper() for c in sys.argv[1:]] or ["SOL", "XRP", "BTC"]
NS = 1_000_000_000

con = duckdb.connect()
print("JUNE: DOES INVENTORY MOVE QUOTES? (top 10 makers by quotes, raw correlations)")
for coin in COINS:
    files = sorted(f for d in JUNE_GOOD for f in glob.glob(os.path.join(QUOTES, coin, f"{d}*.parquet")))
    if not files:
        print(f"\n   {coin}: no June files")
        continue
    k = 40 if coin == "BTC" else 10
    df = con.execute(f"""
        SELECT wallet, (side = 'B')::INT AS bid, ts, rel_mid_bp, inventory, ticks_behind_best
        FROM read_parquet({files}, union_by_name=true)
        WHERE NOT in_gap AND NOT crosses_book AND mid IS NOT NULL AND inventory IS NOT NULL
              AND hash(oid) % {k} = 0""").df()
    print(f"\n   {coin}: {len(df):,} sampled quotes, {df['wallet'].nunique()} accounts | "
          f"inventory exactly 0 in {(df['inventory'] == 0).mean():.1%} of quotes")
    top = df["wallet"].value_counts().head(10).index
    print("      maker       quotes   inv=0    inv std   inv range            bid%   corr_side  corr_px_bid  corr_min")
    for w in top:
        d = df[df["wallet"] == w]
        b = d[d["bid"] == 1]
        mn = d.assign(m=d["ts"] // (60 * NS)).groupby("m").agg(bs=("bid", "mean"), inv=("inventory", "mean"),
                                                               n=("bid", "size"))
        mn = mn[mn["n"] >= 5]
        c = lambda x, y: np.corrcoef(x, y)[0, 1] if len(x) > 30 and np.std(x) > 0 and np.std(y) > 0 else np.nan
        print(f"      {w[:8]}… {len(d):>9,} {(d['inventory'] == 0).mean():>6.0%} {d['inventory'].std():>10.3g} "
              f"{d['inventory'].min():>9.3g}..{d['inventory'].max():<9.3g} {d['bid'].mean():>5.0%} "
              f"{c(d['bid'], d['inventory']):>+10.2f} {c(b['rel_mid_bp'], b['inventory']):>+12.2f} "
              f"{c(mn['bs'], mn['inv']):>+9.2f}")
