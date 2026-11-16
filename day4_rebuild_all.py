"""Day 4: rebuild EVERY wallet's open orders, day by day, and check against the next snapshot."""
import os
import sys
import time

import duckdb
import pandas as pd

ROOTS = ["E:/hl_export", "E:/hl_export/delivery-2"]
COINS = ["SOL", "XRP"]
EXPECTED = ["2026-05-31", "2026-06-05", "2026-08-14", "2026-10-25"]   # the 4 snapshot dates
WINDOWS = [  # (name, start snapshot #, end snapshot #, first day, last day, outage start, outage end) UTC
    ("June", 0, 1, "2026-05-31", "2026-06-05", "2026-06-05 22:00:10", "2026-06-05 22:24:16"),
    ("Aug-Oct", 2, 3, "2026-08-14", "2026-10-25", "2026-10-25 09:32:20", "2026-10-25 10:28:57"),
]
ENDS = "status NOT IN ('open', 'triggered')"
K = 1_000_000                        # order key = block * K + seq (checked: seq must stay below K)
KEY = f"block_number * {K} + seq"


def existing(paths):
    return [p for p in paths if os.path.exists(p)]


def rp(paths):
    return "read_parquet([" + ", ".join(f"'{p}'" for p in paths) + "], union_by_name=true)"


def log(s):
    print(s, file=sys.stderr, flush=True)


con = duckdb.connect()
con.execute("SET TimeZone = 'UTC'")
os.makedirs("E:/duckdb_tmp", exist_ok=True)
con.execute("SET temp_directory = 'E:/duckdb_tmp'")


def q1(sql):
    return con.execute(sql).fetchone()


for coin in COINS:
    snap_files = existing([f"{r}/snapshots/{coin}_resting_orders.parquet" for r in ROOTS])
    SNAP = rp(snap_files)
    ts_type = dict((c[0], c[1]) for c in con.execute(f"DESCRIBE SELECT * FROM {SNAP}").fetchall())["order_timestamp"]
    if "INT" in ts_type.upper() or "DOUBLE" in ts_type.upper():
        born = ("CASE WHEN order_timestamp > 1e17 THEN to_timestamp(order_timestamp / 1e9) "
                "WHEN order_timestamp > 1e14 THEN to_timestamp(order_timestamp / 1e6) "
                "WHEN order_timestamp > 1e11 THEN to_timestamp(order_timestamp / 1e3) "
                "ELSE to_timestamp(order_timestamp) END")
    else:
        born = "CAST(order_timestamp AS TIMESTAMPTZ)"

    # ---------- the snapshots
    snaps = con.execute(f"""SELECT CAST(snapshot_time AS VARCHAR) AS t, MIN(last_block_number), MAX(last_block_number),
                                   COUNT(*), COUNT(*) - COUNT(DISTINCT oid)
                            FROM {SNAP} GROUP BY 1 ORDER BY 3""").fetchall()
    print(f"\n################ {coin}: SNAPSHOTS ################")
    for t, bmin, bmax, n, dup in snaps:
        print(f"   {t} | blocks {bmin:,}-{bmax:,}{'' if bmin == bmax else ' (rows differ in block!)'} | "
              f"{n:,} orders | duplicate order ids {dup}")
    snap_dups = [s[4] for s in snaps]
    if [s[0][:10] for s in snaps] != EXPECTED:
        sys.exit(f"   {coin}: snapshots are not the 4 expected dates {EXPECTED} - stopping.")
    if len(snap_files) == 2:
        both_files = q1(f"""SELECT COUNT(*) FROM (
            SELECT DISTINCT user_address FROM read_parquet('{snap_files[0]}')
            INTERSECT SELECT DISTINCT user_address FROM read_parquet('{snap_files[1]}'))""")[0]
        print(f"   wallets present in BOTH deliveries' snapshot files: {both_files} (want 0)")

    for name, i0, i1, d0, d1, g0, g1 in WINDOWS:
        t_start = time.time()
        s0_time, _, b0, _, _ = snaps[i0]
        s1_time, _, b1, _, _ = snaps[i1]
        print(f"\n==================== {coin} | {name} ====================")
        print(f"   start photo {s0_time} (block {b0:,}) -> end photo {s1_time} (block {b1:,})")

        # the open set C starts as the start photo
        con.execute(f"""CREATE OR REPLACE TABLE c AS
            SELECT CAST(oid AS BIGINT) AS oid, user_address AS u, CAST(NULL AS TIMESTAMPTZ) AS born_ts,
                   CAST(size AS DOUBLE) AS size0, 0.0 AS done
            FROM {SNAP} WHERE CAST(snapshot_time AS VARCHAR) = '{s0_time}'""")
        start_n = q1("SELECT COUNT(*) FROM c")[0]

        tot = dict(events=0, born=0, ends=0, miss=0, multi_open=0, multi_end=0, null_status=0, triggered=0,
                   owner_change=0, size_change=0, wallet_overlap=0, trade_dup=0, event_dup=0, bad_seq=0, null_size=0,
                   fills=0, f_good=0, f_amb=0, f_bad=0)
        miss_by, bad_by, miss_days = {}, {}, {}
        win = f"block_number > {b0} AND block_number <= {b1}"

        for day in pd.date_range(d0, d1).strftime("%Y-%m-%d"):
            of = existing([f"{r}/orders/{coin}/{day}.parquet" for r in ROOTS])
            tf = existing([f"{r}/trades/{coin}/{day}.parquet" for r in ROOTS])
            if not of and not tf:
                continue
            if of:
                con.execute(f"""CREATE OR REPLACE TEMP TABLE d AS
                    SELECT CAST(oid AS BIGINT) AS oid, user_address AS u, status, CAST(tif AS VARCHAR) AS tif,
                           block_number, seq, timestamp, orig_size
                    FROM {rp(of)} WHERE {win}""")
            else:                                   # a day with trades but no order file: still process the trades
                con.execute("""CREATE OR REPLACE TEMP TABLE d AS SELECT CAST(NULL AS BIGINT) AS oid,
                    CAST(NULL AS VARCHAR) AS u, CAST(NULL AS VARCHAR) AS status, CAST(NULL AS VARCHAR) AS tif,
                    CAST(NULL AS BIGINT) AS block_number, CAST(NULL AS BIGINT) AS seq,
                    CAST(NULL AS TIMESTAMPTZ) AS timestamp, CAST(NULL AS DOUBLE) AS orig_size WHERE FALSE""")
            n, max_seq, n_null, n_trig, bad_seq, null_size, ev_dup = q1("""SELECT COUNT(*), MAX(seq),
                COUNT(*) FILTER (WHERE status IS NULL), COUNT(*) FILTER (WHERE status = 'triggered'),
                COUNT(*) FILTER (WHERE seq IS NULL OR seq < 0),
                COUNT(*) FILTER (WHERE status = 'open' AND orig_size IS NULL),
                COUNT(*) - COUNT(DISTINCT (oid, status, block_number, seq)) FROM d""")
            tot["bad_seq"] += bad_seq
            tot["null_size"] += null_size
            tot["event_dup"] += ev_dup
            if n == 0 and not tf:
                continue
            if max_seq is not None and max_seq >= K:
                sys.exit(f"seq {max_seq} too large on {day}; the order key needs changing.")
            if len(of) == 2:                         # the two deliveries must not share wallets
                tot["wallet_overlap"] += q1(f"""SELECT COUNT(*) FROM (
                    SELECT DISTINCT user_address FROM read_parquet('{of[0]}')
                    INTERSECT SELECT DISTINCT user_address FROM read_parquet('{of[1]}'))""")[0]
            tot["null_status"] += n_null
            tot["triggered"] += n_trig

            # one row per order seen today
            con.execute(f"""CREATE OR REPLACE TEMP TABLE a AS
                SELECT oid, ANY_VALUE(u) AS u, ANY_VALUE(tif) AS tif,
                       COUNT(DISTINCT u) AS n_owner, COUNT(DISTINCT orig_size) AS n_size,
                       COUNT(*) FILTER (WHERE status = 'open') AS n_open,
                       COUNT(*) FILTER (WHERE {ENDS}) AS n_end,
                       MIN({KEY}) FILTER (WHERE status = 'open') AS ko,
                       MIN({KEY}) FILTER (WHERE {ENDS}) AS ke,
                       ARG_MIN(timestamp, {KEY}) FILTER (WHERE status = 'open') AS born_ts,
                       ARG_MIN(orig_size, {KEY}) FILTER (WHERE status = 'open') AS size0,
                       ARG_MIN(status, {KEY}) FILTER (WHERE {ENDS}) AS end_status
                FROM d WHERE status IS NOT NULL GROUP BY oid""")
            # endings must hit a known order: in the open set
            r = con.execute("""
                SELECT COALESCE(SUM(a.n_end), 0),
                       COUNT(*) FILTER (WHERE a.ke IS NOT NULL AND c.oid IS NULL
                                        AND NOT COALESCE(a.ko < a.ke, FALSE)),
                       COUNT(*) FILTER (WHERE a.n_open > 1 OR (a.n_open >= 1 AND c.oid IS NOT NULL)),
                       COUNT(*) FILTER (WHERE a.n_end > 1),
                       COUNT(*) FILTER (WHERE a.n_open >= 1),
                       COUNT(*) FILTER (WHERE a.n_owner > 1 OR (c.oid IS NOT NULL AND c.u IS DISTINCT FROM a.u)),
                       COUNT(*) FILTER (WHERE a.n_size > 1)
                FROM a LEFT JOIN c ON a.oid = c.oid""").fetchone()
            tot["events"] += n
            tot["ends"] += r[0]
            tot["miss"] += r[1]
            tot["multi_open"] += r[2]
            tot["multi_end"] += r[3]
            tot["born"] += r[4]
            tot["owner_change"] += r[5]
            tot["size_change"] += r[6]
            if r[1]:
                miss_days[day] = r[1]
                for st, tif, m in con.execute("""
                        SELECT a.end_status, a.tif, COUNT(*) FROM a LEFT JOIN c ON a.oid = c.oid
                        WHERE a.ke IS NOT NULL AND c.oid IS NULL AND NOT COALESCE(a.ko < a.ke, FALSE)
                        GROUP BY 1, 2""").fetchall():
                    miss_by[(st, tif)] = miss_by.get((st, tif), 0) + m

            # fills: born before
            if tf:
                con.execute(f"""CREATE OR REPLACE TEMP TABLE f AS
                    SELECT CAST(order_id AS BIGINT) AS oid, block_number AS b, CAST(size AS DOUBLE) AS size,
                           tx_hash, trade_id,
                           lower(CAST(crossed AS VARCHAR)) = 'true' AS taker,
                           twap_id IS NOT NULL AS twap,
                           lower(CAST(is_liquidation AS VARCHAR)) = 'true' AS liq
                    FROM {rp(tf)} WHERE {win}""")
                tot["trade_dup"] += q1("SELECT COUNT(*) - COUNT(DISTINCT (tx_hash, trade_id, oid)) FROM f")[0]
                con.execute("CREATE OR REPLACE TEMP TABLE f AS SELECT DISTINCT * FROM f")   # drop EXACT repeats only
                tot["trade_conflict"] = tot.get("trade_conflict", 0) + q1(
                    "SELECT COUNT(*) - COUNT(DISTINCT (tx_hash, trade_id, oid)) FROM f")[0]
                con.execute(f"""CREATE OR REPLACE TEMP TABLE fx AS
                    SELECT f.*,
                           (c.oid IS NOT NULL OR COALESCE(a.ko // {K} < f.b, FALSE)) AS born_before,
                           COALESCE(a.ko // {K} = f.b, FALSE) AS born_same,
                           COALESCE(a.ke // {K} < f.b, FALSE) AS ended_before,
                           COALESCE(a.ke // {K} = f.b, FALSE) AS ended_same
                    FROM f LEFT JOIN c ON f.oid = c.oid LEFT JOIN a ON f.oid = a.oid""")
                fr = q1("""SELECT COUNT(*),
                        COUNT(*) FILTER (WHERE (born_before OR born_same) AND NOT ended_before
                                         AND NOT born_same AND NOT ended_same),
                        COUNT(*) FILTER (WHERE (born_before OR born_same) AND NOT ended_before
                                         AND (born_same OR ended_same)),
                        COUNT(*) FILTER (WHERE NOT (born_before OR born_same) OR ended_before)
                        FROM fx""")
                tot["fills"] += fr[0]
                tot["f_good"] += fr[1]
                tot["f_amb"] += fr[2]
                tot["f_bad"] += fr[3]
                if fr[3]:
                    for why, tk, tw, lq, m in con.execute("""
                            SELECT CASE WHEN ended_before THEN 'after its end' ELSE 'never born' END,
                                   taker, twap, liq, COUNT(*) FROM fx
                            WHERE NOT (born_before OR born_same) OR ended_before GROUP BY 1, 2, 3, 4""").fetchall():
                        bad_by[(why, tk, tw, lq)] = bad_by.get((why, tk, tw, lq), 0) + m
                con.execute("CREATE OR REPLACE TEMP TABLE fd AS SELECT oid, SUM(size) AS s FROM f GROUP BY oid")
            else:
                con.execute("CREATE OR REPLACE TEMP TABLE fd AS SELECT CAST(NULL AS BIGINT) AS oid, 0.0 AS s WHERE FALSE")

            # carry the open set to tomorrow
            con.execute("""CREATE OR REPLACE TABLE c AS
                SELECT c.oid, c.u, c.born_ts, c.size0, c.done + COALESCE(fd.s, 0) AS done
                FROM c LEFT JOIN a ON c.oid = a.oid LEFT JOIN fd ON c.oid = fd.oid
                WHERE a.ke IS NULL
                UNION ALL
                SELECT a.oid, a.u, a.born_ts, CAST(a.size0 AS DOUBLE), COALESCE(fd.s, 0)
                FROM a LEFT JOIN c ON a.oid = c.oid LEFT JOIN fd ON a.oid = fd.oid
                WHERE a.ko IS NOT NULL AND a.ke IS NULL AND c.oid IS NULL""")
            log(f"   {coin} {name} {day}: {n:>12,} events | open now {q1('SELECT COUNT(*) FROM c')[0]:>7,} "
                f"| misses today {r[1]:,} | {time.time() - t_start:,.0f}s")

        # ---------- report ----------
        print(f"   start photo: {start_n:,} resting orders")
        print(f"   events replayed: {tot['events']:,} | orders born: {tot['born']:,} | endings: {tot['ends']:,}")
        print("\n   0. DATA INTEGRITY (want 0 everywhere)")
        print(f"      wallets in both deliveries (summed over days): {tot['wallet_overlap']:,} | "
              f"duplicate trade records: {tot['trade_dup']:,} | events with no status: {tot['null_status']:,}")
        print(f"      orders whose owner changed: {tot['owner_change']:,} | orders whose size changed: {tot['size_change']:,}")
        print(f"      'triggered' events (treated as NOT ending the order): {tot['triggered']:,}")
        print(f"      duplicate order events: {tot['event_dup']:,} | missing/negative seq: {tot['bad_seq']:,} | "
              f"opens with no size: {tot['null_size']:,} | conflicting trade duplicates: {tot.get('trade_conflict', 0):,}")
        integrity = (tot['event_dup'] == 0 and tot['bad_seq'] == 0 and tot['null_size'] == 0 and tot['null_status'] == 0
                     and tot.get('trade_conflict', 0) == 0 and tot['owner_change'] == 0 and tot['size_change'] == 0
                     and tot['multi_open'] == 0 and tot['multi_end'] == 0 and tot['f_bad'] == 0
                     and snap_dups[i0] == 0 and snap_dups[i1] == 0)
        print(f"      DATA INTEGRITY VERDICT: {'PASS' if integrity else 'CHECK'}"
              f"{' (exact duplicate trades were removed before sizes were added up)' if tot['trade_dup'] else ''}")
        print("\n   A. ENDINGS THAT HIT A KNOWN ORDER (want 0 misses, except right after the outage)")
        print(f"      misses: {tot['miss']:,}")
        for dday, m in sorted(miss_days.items()):
            print(f"         {dday}: {m:,}{'   <- outage day' if dday == g0[:10] else ''}")
        for (st, tif), m in sorted(miss_by.items(), key=lambda x: -x[1])[:8]:
            print(f"         status {st}, tif {tif}: {m:,}")
        print("\n   B. CLEAN LIFE CYCLES (want 0, 0)")
        print(f"      opened more than once: {tot['multi_open']:,} | ended more than once: {tot['multi_end']:,}")
        print("\n   C. FILLS")
        print(f"      {tot['fills']:,} fills | certainly valid {tot['f_good']:,} | "
              f"ambiguous (same block) {tot['f_amb']:,} | INVALID {tot['f_bad']:,}")
        for (why, tk, tw, lq), m in sorted(bad_by.items(), key=lambda x: -x[1])[:8]:
            print(f"         {why}: taker={tk} twap={tw} liquidation={lq}: {m:,}")

        # ---------- D
        con.execute(f"""CREATE OR REPLACE TEMP TABLE s1 AS
            SELECT CAST(oid AS BIGINT) AS oid, user_address AS u, CAST(size AS DOUBLE) AS size,
                   {born} AS born_ts
            FROM {SNAP} WHERE CAST(snapshot_time AS VARCHAR) = '{s1_time}'""")
        G0, G1 = f"TIMESTAMPTZ '{g0}+00'", f"TIMESTAMPTZ '{g1}+00' + INTERVAL 2 SECOND"
        DAY0 = f"TIMESTAMPTZ '{g0[:10]} 00:00:00+00'"
        res = q1(f"""SELECT
            (SELECT COUNT(*) FROM s1),
            (SELECT COUNT(*) FROM c),
            (SELECT COUNT(*) FROM s1 JOIN c USING (oid)),
            (SELECT COUNT(*) FROM s1 JOIN c USING (oid) WHERE s1.u <> c.u),
            (SELECT COUNT(*) FROM s1 JOIN c USING (oid) WHERE abs(s1.size - (c.size0 - c.done)) < 1e-8),
            (SELECT COUNT(*) FROM s1 ANTI JOIN c USING (oid)),
            (SELECT COUNT(*) FROM s1 ANTI JOIN c USING (oid) WHERE born_ts BETWEEN {G0} AND {G1}),
            (SELECT COUNT(*) FROM c ANTI JOIN s1 USING (oid)),
            (SELECT COUNT(*) FROM c ANTI JOIN s1 USING (oid) WHERE born_ts >= {DAY0} AND born_ts < {G0}),
            (SELECT COUNT(*) FROM c ANTI JOIN s1 USING (oid) WHERE born_ts IS NULL OR born_ts < {DAY0}),
            (SELECT COUNT(*) FROM c ANTI JOIN s1 USING (oid) WHERE born_ts > {G1}),
            (SELECT COUNT(*) FROM c ANTI JOIN s1 USING (oid) WHERE born_ts >= {G0} AND born_ts <= {G1})""")
        (n_s, n_c, both, owner_bad, size_ok, s_only, s_only_gap,
         c_only, c_only_today, c_only_old, c_only_after, c_only_during) = res
        s_unexpl = s_only - s_only_gap
        print("\n   D. REBUILT SHOP WINDOWS vs THE END PHOTO (the main check)")
        print(f"      photo: {n_s:,} open orders | rebuilt: {n_c:,} | CONFIRMED MATCH (in both): {both:,}")
        print(f"      of the matches: same owner {both - owner_bad:,} of {both:,} | "
              f"remaining size matches (size at birth - executed) {size_ok:,} of {both:,} "
              f"-> {'PASS' if size_ok == both and owner_bad == 0 else 'CHECK'}")
        print(f"      only in the photo: {s_only:,}")
        print(f"         consistent with the outage (born inside it, we never saw the birth): {s_only_gap:,}")
        print(f"         UNEXPLAINED (born outside the outage, we should have seen it): {s_unexpl:,}")
        print(f"      only in our rebuild: {c_only:,}")
        print(f"         consistent with the outage (born earlier on the outage day, or inside it): "
              f"{c_only_today + c_only_during:,}")
        print(f"         AMBIGUOUS (older order; its end may be inside the outage, or missed elsewhere): {c_only_old:,}")
        print(f"         UNEXPLAINED (born after the outage, end should be in our data): {c_only_after:,}")
        good = s_unexpl == 0 and c_only_after == 0 and size_ok == both and owner_bad == 0
        print(f"      RECONSTRUCTION VERDICT: {'PASS' if good else 'CHECK'}")
        print(f"      OVERALL (reconstruction AND data integrity): {'PASS' if good and integrity else 'CHECK'}"
              f"{f' ({c_only_old:,} ambiguous old orders - see section E)' if good and c_only_old else ''}")
        if s_unexpl:
            for row in con.execute(f"""SELECT oid, strftime(born_ts, '%Y-%m-%d %H:%M:%S') FROM s1 ANTI JOIN c USING (oid)
                                       WHERE born_ts IS NULL OR NOT (born_ts BETWEEN {G0} AND {G1}) LIMIT 5""").fetchall():
                print(f"         example photo-only order {row[0]} born {row[1]}")

        # ---------- E. the ambiguous old orders: how old are they? ----------
        if c_only_old:
            print("\n   E. AMBIGUOUS OLD ORDERS (in our rebuild, not in the photo) by birth date")
            for dd, m in con.execute(f"""SELECT COALESCE(strftime(born_ts, '%Y-%m-%d'), 'start photo'), COUNT(*)
                                         FROM c ANTI JOIN s1 USING (oid)
                                         WHERE born_ts IS NULL OR born_ts < {DAY0}
                                         GROUP BY 1 ORDER BY 2 DESC LIMIT 10""").fetchall():
                print(f"      {dd}: {m:,}")
            print("      (a few long-lived orders are normal; a big pile on one day points to missed endings that day)")
        print(f"   ({time.time() - t_start:,.0f} seconds)")
print()
