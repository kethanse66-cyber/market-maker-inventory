"""Day 1: the story of one SOL day, told by the data.

Run:  python E:\\p3\\day1_story.py

We use June 3, a day with NO tick-boundary crossing, on purpose: exploring a crossing day before the
pre-registration is frozen could bias us. All times are shown in UTC.
"""
import duckdb

D = "2026-06-03"
ORD = (f"read_parquet(['E:/hl_export/orders/SOL/{D}.parquet', "
       f"'E:/hl_export/delivery-2/orders/SOL/{D}.parquet'], union_by_name=true)")
TRD = (f"read_parquet(['E:/hl_export/trades/SOL/{D}.parquet', "
       f"'E:/hl_export/delivery-2/trades/SOL/{D}.parquet'], union_by_name=true)")

con = duckdb.connect()
con.execute("SET TimeZone = 'UTC'")

# ---------- 1. the whole day ----------
events, orders, makers = con.execute(
    f"SELECT COUNT(*), COUNT(DISTINCT oid), COUNT(DISTINCT user_address) FROM {ORD}").fetchone()
print(f"\n1. THE DAY ({D}, SOL)")
print(f"   {events:,} events from {orders:,} orders, placed by {makers} maker wallets")
for status, n in con.execute(f"SELECT status, COUNT(*) AS n FROM {ORD} GROUP BY status ORDER BY n DESC").fetchall():
    print(f"   {status:20} {n:>12,}  ({n / events:.1%})")

# ---------- 2. the busiest maker ----------
wallet, placed = con.execute(
    f"SELECT user_address, COUNT(DISTINCT oid) FROM {ORD} WHERE status = 'open' "
    f"GROUP BY 1 ORDER BY 2 DESC LIMIT 1").fetchone()
con.execute(f"""
    CREATE TEMP TABLE life AS
    SELECT oid,
           MIN(CASE WHEN status = 'open'  THEN timestamp END) AS t_open,
           MAX(CASE WHEN status <> 'open' THEN timestamp END) AS t_end,
           MAX(CASE WHEN status <> 'open' THEN status END)    AS how_it_ended,
           ANY_VALUE(side) AS side, ANY_VALUE(limit_price) AS price, ANY_VALUE(orig_size) AS size
    FROM {ORD} WHERE user_address = '{wallet}' GROUP BY oid""")
n_full, med_ms, under_1s, filled = con.execute("""
    SELECT COUNT(*),
           MEDIAN(date_diff('millisecond', t_open, t_end)),
           AVG(CASE WHEN date_diff('millisecond', t_open, t_end) < 1000 THEN 1 ELSE 0 END),
           AVG(CASE WHEN how_it_ended = 'filled' THEN 1 ELSE 0 END)
    FROM life WHERE t_open IS NOT NULL AND t_end IS NOT NULL""").fetchone()
execs, orders_hit, maker_share, usd = con.execute(f"""
    SELECT COUNT(*), COUNT(DISTINCT order_id),
           AVG(CASE WHEN lower(CAST(crossed AS VARCHAR)) = 'true' THEN 0.0
                    WHEN lower(CAST(crossed AS VARCHAR)) = 'false' THEN 1.0 END), SUM(price * size)
    FROM {TRD} WHERE user_address = '{wallet}'""").fetchone()
print(f"\n2. THE BUSIEST MAKER: {wallet[:10]}...")
print(f"   placed {placed:,} orders today")
print(f"   a typical order lives {med_ms / 1000:.2f} seconds; {under_1s:.0%} live less than 1 second")
print(f"   only {filled:.2%} of its orders end filled. How its orders ended:")
for how, n in con.execute("""SELECT COALESCE(how_it_ended, 'no ending today'), COUNT(*) FROM life
                             WHERE t_open IS NOT NULL GROUP BY 1 ORDER BY 2 DESC""").fetchall():
    print(f"      {how:22} {n:>10,}")
print(f"   executions: {execs:,} on {orders_hit:,} different orders, worth ${usd:,.0f}; "
      f"{maker_share:.0%} maker-side (its quote was hit)")

# ---------- 2b. sanity checks: do the columns mean what we think? ----------
print("\n2b. CHECKS (we want: 0, 0, 0, no overlap, and only 'false'/'true' for crossed)")
multi_end = con.execute(f"""
    SELECT COUNT(*) FROM (SELECT oid FROM {ORD} WHERE user_address = '{wallet}' AND status <> 'open'
                          GROUP BY oid HAVING COUNT(*) > 1)""").fetchone()[0]
print(f"   orders with more than one ending event: {multi_end}")
changed = con.execute(f"""
    SELECT COUNT(*) FROM (SELECT oid FROM {ORD} WHERE user_address = '{wallet}' GROUP BY oid
                          HAVING COUNT(DISTINCT side) > 1 OR COUNT(DISTINCT limit_price) > 1
                              OR COUNT(DISTINCT orig_size) > 1)""").fetchone()[0]
print(f"   orders whose side, price or original size changed: {changed}")
both = con.execute(f"""
    SELECT COUNT(*) FROM (
      SELECT DISTINCT user_address FROM read_parquet('E:/hl_export/orders/SOL/{D}.parquet')
      INTERSECT
      SELECT DISTINCT user_address FROM read_parquet('E:/hl_export/delivery-2/orders/SOL/{D}.parquet'))""").fetchone()[0]
print(f"   wallets present in BOTH deliveries: {both}")
dups = con.execute(f"""
    SELECT COUNT(*) FROM (SELECT oid, status, timestamp FROM {ORD}
                          GROUP BY ALL HAVING COUNT(*) > 1)""").fetchone()[0]
print(f"   duplicate events (same order, status and time): {dups}")
for val, n in con.execute(f"""SELECT CAST(crossed AS VARCHAR), COUNT(*) FROM {TRD}
                             WHERE user_address = '{wallet}' GROUP BY 1 ORDER BY 1""").fetchall():
    print(f"   crossed = {str(val):6} -> {n:,} executions")

# ---------- 3. one order's whole life ----------
row = con.execute("""SELECT oid, side, price, size, strftime(t_open, '%H:%M:%S.%f'),
                            date_diff('millisecond', t_open, t_end)
                     FROM life WHERE how_it_ended = 'filled' AND t_open IS NOT NULL
                     ORDER BY t_open LIMIT 1""").fetchone()
if row:
    oid, side, price, size, t_open, life_ms = row
    word = "SELL (ask)" if side == "A" else "BUY (bid)"
    print(f"\n3. ONE ORDER'S LIFE: order {oid}")
    print(f"   {t_open}: placed, {word} {size} SOL at ${price}")
    for ts, px, sz, fee, taker in con.execute(f"""
            SELECT strftime(timestamp, '%H:%M:%S.%f'), price, size, fee,
                   lower(CAST(crossed AS VARCHAR)) = 'true'
            FROM {TRD} WHERE order_id = {oid} ORDER BY timestamp""").fetchall():
        print(f"   {ts}: FILLED {sz} SOL at ${px}  (fee {fee}; "
              f"{'it took liquidity' if taker else 'a trader hit our quote'})")
    done = con.execute(f"SELECT COALESCE(SUM(size), 0) FROM {TRD} WHERE order_id = {oid}").fetchone()[0]
    print(f"   original size {size} SOL, executed {done} SOL in total, remaining {size - done:g} SOL "
          f"-> {'PASS' if abs(size - done) < 1e-8 else 'CHECK'}")
    print(f"   it lived {life_ms / 1000:.3f} seconds")
print()
