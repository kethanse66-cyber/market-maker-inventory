"""Day 4c: zoom into Aug 29 - when and for which wallets did the order stream lose events? ~3 minutes.

Run:  python E:\\p3\\day4c_aug29.py > E:\\p3\\day4c_output.txt

Two kinds of trouble seen on Aug 29 in the Day 4 rebuild:
  (a) endings of orders we never saw born       -> an 'open' event is missing
  (b) orders born that day that never end         -> an ending event is missing
We find WHEN each happened and WHICH wallets, plus every silence longer than 5 seconds that day.
"""
import os

import duckdb

ROOTS = ["E:/hl_export", "E:/hl_export/delivery-2"]
con = duckdb.connect()
con.execute("SET TimeZone = 'UTC'")


def files(coin, days):
    return [p for p in (f"{r}/orders/{coin}/{d}.parquet" for r in ROOTS for d in days) if os.path.exists(p)]


for coin in ("SOL", "XRP"):
    print(f"\n==================== {coin} ====================")
    day = files(coin, ["2026-08-29"])
    around = files(coin, ["2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31"])
    con.execute(f"""CREATE OR REPLACE TEMP TABLE e AS
        SELECT CAST(oid AS BIGINT) AS oid, user_address AS u, status, timestamp
        FROM read_parquet({around}, union_by_name=true)""")

    print("   silences longer than 5 seconds on Aug 29 (all wallets together):")
    for t0, t1, s in con.execute(f"""
            WITH b AS (SELECT DISTINCT date_trunc('second', timestamp) AS t FROM read_parquet({day}, union_by_name=true)),
                 g AS (SELECT LAG(t) OVER (ORDER BY t) AS t0, t AS t1 FROM b)
            SELECT strftime(t0, '%H:%M:%S'), strftime(t1, '%H:%M:%S'), date_diff('second', t0, t1)
            FROM g WHERE date_diff('second', t0, t1) > 5 ORDER BY t0""").fetchall():
        print(f"      {t0} -> {t1}  ({s} s)")

    # (a) endings on Aug 29 whose order has no 'open' anywhere in Aug 28-31
    print("\n   (a) endings with no birth, by minute (UTC):")
    for m, n, w in con.execute("""
            SELECT strftime(date_trunc('minute', timestamp), '%H:%M'), COUNT(*), COUNT(DISTINCT u)
            FROM e WHERE status NOT IN ('open', 'triggered')
              AND CAST(timestamp AS DATE) = DATE '2026-08-29'
              AND oid NOT IN (SELECT oid FROM e WHERE status = 'open')
            GROUP BY 1 ORDER BY 1""").fetchall():
        print(f"      {m}  {n:>5} endings, {w} wallets")

    # (b) orders born on Aug 29 with no ending anywhere in Aug 29-31
    print("\n   (b) births with no ending (until Aug 31), by minute (UTC):")
    for m, n, w in con.execute("""
            SELECT strftime(date_trunc('minute', timestamp), '%H:%M'), COUNT(*), COUNT(DISTINCT u)
            FROM e WHERE status = 'open' AND CAST(timestamp AS DATE) = DATE '2026-08-29'
              AND oid NOT IN (SELECT oid FROM e WHERE status NOT IN ('open', 'triggered'))
            GROUP BY 1 ORDER BY 1""").fetchall():
        print(f"      {m}  {n:>5} births, {w} wallets")

    print("\n   which wallets (a + b together):")
    for u, n in con.execute("""
            SELECT u, COUNT(*) FROM e
            WHERE CAST(timestamp AS DATE) = DATE '2026-08-29' AND (
                  (status NOT IN ('open', 'triggered') AND oid NOT IN (SELECT oid FROM e WHERE status = 'open'))
               OR (status = 'open' AND oid NOT IN (SELECT oid FROM e WHERE status NOT IN ('open', 'triggered'))))
            GROUP BY 1 ORDER BY 2 DESC LIMIT 10""").fetchall():
        print(f"      {u[:10]}...  {n:,}")
print()
