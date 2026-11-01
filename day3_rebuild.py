"""Day 3: rebuild one wallet's open orders for one day, starting from the snapshot.

Run:  python E:\\p3\\day3_rebuild.py > E:\\p3\\day3_output.txt

Idea: snapshot (photo of resting orders on May 31, 23:58:55) + every event after it.
'open' adds an order to the window; an ending status removes it. Every ending and every fill
must point to an order we already know (from the photo or an earlier 'open').
We use the EARLIEST snapshot on purpose: June 1 comes right after it.
"""
import os
import sys

import duckdb

COIN, DAY, NEXT = "SOL", "2026-06-01", "2026-06-02"
ROOTS = ["E:/hl_export", "E:/hl_export/delivery-2"]
NOT_ENDINGS = {"open", "triggered"}      # 'triggered' = a trigger order waking up; the order lives on


def existing(paths):
    return [p for p in paths if os.path.exists(p)]


def rp(paths):
    if not paths:
        sys.exit("No files found - check the E: drive paths.")
    return "read_parquet([" + ", ".join(f"'{p}'" for p in paths) + "], union_by_name=true)"


SNAP = rp(existing([f"{r}/snapshots/{COIN}_resting_orders.parquet" for r in ROOTS]))
ORD = rp(existing([f"{r}/orders/{COIN}/{d}.parquet" for r in ROOTS for d in ("2026-05-31", DAY)]))
TRD = rp(existing([f"{r}/trades/{COIN}/{d}.parquet" for r in ROOTS for d in ("2026-05-31", DAY)]))

con = duckdb.connect()
con.execute("SET TimeZone = 'UTC'")

# ---------- 0. what the files look like ----------
print("\n0. COLUMNS")
cols = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM {SNAP}").fetchall()]
print("   snapshot: " + ", ".join(cols))
print("   trades:   " + ", ".join(c[0] for c in con.execute(f"DESCRIBE SELECT * FROM {TRD}").fetchall()))
size_col = "size" if "size" in cols else None

# ---------- 1. the photo we start from (the earliest snapshot = May 31) ----------
snap_time, snap_block = con.execute(f"""
    SELECT CAST(snapshot_time AS VARCHAR), MAX(last_block_number) FROM {SNAP}
    GROUP BY 1 ORDER BY 2 LIMIT 1""").fetchone()
print(f"\n1. START: snapshot at {snap_time}, block {snap_block:,}")
print(f"   is it the May 31 snapshot? {'PASS' if snap_time.startswith('2026-05-31') else 'CHECK'}")
first_ev = con.execute(f"""SELECT MIN(block_number), strftime(MIN(timestamp), '%Y-%m-%d %H:%M:%S')
                           FROM {ORD} WHERE block_number > {snap_block}""").fetchone()
if first_ev[0] is None:
    sys.exit("No order events after the snapshot.")
print(f"   first order event after it: block {first_ev[0]:,} at {first_ev[1]} "
      f"(gap of {first_ev[0] - snap_block} blocks; we want a small number)")

WIN = f"block_number > {snap_block} AND timestamp < TIMESTAMP '{NEXT} 00:00:00'"

# ---------- 2. pick the wallet: busiest maker that is in the photo ----------
wallet, n_open = con.execute(f"""
    SELECT user_address, COUNT(DISTINCT oid) FROM {ORD}
    WHERE status = 'open' AND {WIN}
      AND user_address IN (SELECT DISTINCT user_address FROM {SNAP} WHERE last_block_number = {snap_block})
    GROUP BY 1 ORDER BY 2 DESC LIMIT 1""").fetchone()
print(f"\n2. WALLET {wallet[:10]}... placed {n_open:,} new orders in the window")

# ---------- 3. load its photo, its events and its fills ----------
S = con.execute(f"""SELECT CAST(oid AS BIGINT) AS oid, {size_col or 'NULL'} AS size FROM {SNAP}
                    WHERE last_block_number = {snap_block} AND user_address = '{wallet}'""").df()
E = con.execute(f"""SELECT CAST(oid AS BIGINT) AS oid, status, block_number, seq, orig_size
                    FROM {ORD} WHERE user_address = '{wallet}' AND {WIN}
                    ORDER BY block_number, seq""").df()
F = con.execute(f"""SELECT CAST(order_id AS BIGINT) AS oid, block_number, size,
                           lower(CAST(crossed AS VARCHAR)) = 'true' AS taker
                    FROM {TRD} WHERE user_address = '{wallet}' AND {WIN}""").df()
if E.empty:
    sys.exit("This wallet has no order events in the window.")
print(f"   photo: {len(S):,} resting orders | events: {len(E):,} | fills: {len(F):,}")

snap_dup = int(S["oid"].duplicated().sum())
snap_missing = int(S["oid"].isna().sum())
event_dup = int(E.duplicated(["oid", "status", "block_number", "seq"]).sum())
trade_dup = int(con.execute(f"""SELECT COUNT(*) - COUNT(DISTINCT (tx_hash, trade_id, order_id)) FROM {TRD}
                                WHERE user_address = '{wallet}' AND {WIN}""").fetchone()[0])
print("\n3b. DATA CLEANLINESS (we want 0 everywhere)")
print(f"   photo: duplicate order ids {snap_dup}, missing order ids {snap_missing}")
print(f"   events: exact duplicates (same order, status, block, seq) {event_dup}")
print(f"   fills: duplicate trade records (same tx, trade id, order) {trade_dup}")
print("   every status seen (only 'open' and 'triggered' are treated as NOT ending an order):")
for st, n in E["status"].value_counts().items():
    print(f"      {st:22} {n:>10,}")

in_photo = set(S["oid"])
opens = E[E["status"] == "open"]
fo = (opens.sort_values(["block_number", "seq"])                    # the real first 'open' row
      .drop_duplicates("oid", keep="first").set_index("oid")[["block_number", "seq"]])
ends = E[~E["status"].isin(NOT_ENDINGS)]
fe = (ends.sort_values(["block_number", "seq"])                     # the real first ending row
      .drop_duplicates("oid", keep="first").set_index("oid")["block_number"])

# ---------- 4. THE TEST: every ending and every fill hits a known order ----------
# endings: same file as the opens, so we can order exactly by (block, seq)
b, s = ends["oid"].map(fo["block_number"]), ends["oid"].map(fo["seq"])
end_ok = ends["oid"].isin(in_photo) | (b < ends["block_number"]) | \
         ((b == ends["block_number"]) & (s < ends["seq"]))
# fills: the trade file has no seq comparable with the order file, so within one block we cannot
# tell the order. Each fill must come after the order's birth AND not after its ending.
fb, eb, blk = F["oid"].map(fo["block_number"]), F["oid"].map(fe), F["block_number"]
born_before = F["oid"].isin(in_photo) | (fb < blk)
born_same = ~born_before & (fb == blk)
not_born = ~born_before & ~born_same
ended_before = eb < blk                                             # order already gone: invalid
ended_same = eb == blk                                              # ended in the same block: ambiguous
fill_bad = not_born | ended_before
fill_amb = ~fill_bad & (born_same | ended_same)
fill_good = ~fill_bad & ~fill_amb
print("\n4. THE TEST (we want 0 misses)")
print(f"   endings: {len(ends):,} | hit a known order: {int(end_ok.sum()):,} | MISSES: {int((~end_ok).sum()):,}")
print(f"   fills:   {len(F):,} | certainly valid: {int(fill_good.sum()):,} | "
      f"ambiguous (same block as birth or ending): {int(fill_amb.sum()):,} | INVALID: {int(fill_bad.sum()):,}")
print(f"      invalid because order never born: {int(not_born.sum()):,} | "
      f"because order had already ended: {int((~not_born & ended_before).sum()):,}")
print(f"      ambiguous - same block as birth: {int((~fill_bad & born_same).sum()):,} | "
      f"same block as ending: {int((~fill_bad & ended_same).sum()):,}")
fill_miss = fill_bad
if (~end_ok).any():
    print("   missed endings by status:")
    for st, n in ends.loc[~end_ok, "status"].value_counts().items():
        print(f"      {st:22} {n:,}")
if fill_miss.any():
    m = F[fill_miss]
    print(f"   missed fills: {int(m['taker'].sum()):,} taker, {int((~m['taker']).sum()):,} maker")

# ---------- 5. no order ends twice, no order opens twice ----------
multi_end = int((ends.groupby("oid").size() > 1).sum())
multi_open = int((opens.groupby("oid").size() > 1).sum())
print("\n5. CLEAN LIFE CYCLES (we want 0, 0 - section 6 is only trusted if both are 0)")
print(f"   orders with more than one ending: {multi_end:,}")
print(f"   orders opened more than once:     {multi_open:,}")

# ---------- 6. the shop window over time ----------
step = E["status"].map(lambda x: 1 if x == "open" else (0 if x in NOT_ENDINGS else -1))
step[step.index.isin(ends.index[~end_ok])] = 0          # an unknown ending cannot remove anything
window = len(S) + step.cumsum()
ended = set(ends.loc[end_ok, "oid"])
still_open = (in_photo | set(opens["oid"])) - ended
same = len(still_open) == window.iloc[-1]
print("\n6. OPEN ORDERS OVER THE DAY")
print(f"   start {len(S):,} | lowest {window.min():,} | highest {window.max():,} | end of day {window.iloc[-1]:,}")
print(f"   cross-check: orders known and not ended = {len(still_open):,} -> {'PASS' if same else 'CHECK'}")
print(f"   window never below zero -> {'PASS' if window.min() >= 0 else 'CHECK'}")
ok6 = (same and window.min() >= 0 and multi_end == 0 and multi_open == 0
       and snap_dup == 0 and snap_missing == 0 and event_dup == 0 and trade_dup == 0
       and (~end_ok).sum() == 0 and fill_bad.sum() == 0)
print(f"   RECONSTRUCTION -> {'PASS' if ok6 else 'FAILED (see sections 3b-6)'}"
      f"{' (with ' + format(int(fill_amb.sum()), ',') + ' ambiguous same-block fills)' if fill_amb.any() else ''}")

# ---------- 7. sizes: executed amount vs size at start ----------
new_size = opens.groupby("oid")["orig_size"].first()                 # born today: original size
old_size = S.set_index("oid")["size"] if size_col else None          # from the photo: size left at photo time
done = F.groupby("oid")["size"].sum()
last_status = ends.groupby("oid")["status"].last()
print("\n7. SIZES")
for label, sz in (("born today", new_size), ("from the photo", old_size)):
    if sz is None:
        print(f"   {label}: snapshot has no size column, skipped")
        continue
    ids = last_status.index.intersection(sz.index)
    st = last_status.reindex(ids)
    ex = done.reindex(ids).fillna(0)
    full = st == "filled"
    eq = (sz.reindex(ids) - ex).abs() < 1e-8
    over = ex > sz.reindex(ids) + 1e-8
    part = (st != "filled") & (ex > 0)
    print(f"   {label}:")
    print(f"      ended 'filled': {int(full.sum()):,} | executed = size: {int((full & eq).sum()):,} "
          f"-> {'PASS' if (eq | ~full).all() else 'CHECK'}")
    print(f"      partly filled, then ended another way: {int(part.sum()):,} | "
          f"of those, executed < size: {int((part & (ex < sz.reindex(ids) - 1e-8)).sum()):,} "
          f"-> {'PASS' if (~part | (ex < sz.reindex(ids) - 1e-8)).all() else 'CHECK'}")
    print(f"      orders that executed MORE than their size: {int(over.sum()):,} (want 0)")
print()
