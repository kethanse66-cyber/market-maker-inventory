"""Day 12: quality check of delivery 3 (the 15 replication coins + 3 clean-placebo coins). ~5-10 minutes.

Run:  python E:\\p3\\day12_check_delivery3.py > E:\\p3\\day12_output.txt

Checks (no outcomes are looked at - quote sides near the boundaries are never touched):
  1. every file in manifest.csv exists, and its Parquet row count equals the manifest's row count
  2. the columns of orders / trades / snapshots are the same as in deliveries 1 and 2
  3. coin-days: 385 delivered of 386 requested (CHIP Apr 21 is the expected missing one)
  4. order events and trades per coin vs the vendor README table
  5. one sample day per coin: no duplicate (block_number, seq, oid, status) and no duplicate fills;
     only that coin, only listed wallets; trades carry start_position
  6. the vendor's rebuild tests: how many snapshot points rebuild exactly, before and from May 1
"""
import csv
import glob
import os
import sys

import duckdb
import pandas as pd
import pyarrow.parquet as pq

D3 = sys.argv[1] if len(sys.argv) > 1 else r"G:\hl_export\delivery-3"
OLD = r"E:\hl_export"
MAN = os.path.join(os.path.dirname(D3), "manifest.csv")
README_ROWS = {  # coin: (order events, trades) from the vendor README
    "XPL": (407109491, 3938469), "SUI": (296094388, 1208287), "DOGE": (242209149, 1273718),
    "PENGU": (184286971, 868195), "AAVE": (172365945, 1176920), "LIT": (94908488, 1236983),
    "UNI": (82138156, 663655), "CASHCAT": (67226104, 2320548), "STRK": (67421084, 981527),
    "ZEC": (67516797, 646101), "LINK": (55852938, 366902), "CHIP": (34708289, 339564),
    "ORDI": (21319528, 161076), "FARTCOIN": (17133582, 96316), "XRP": (14322174, 99498),
    "SKY": (12826874, 60603), "VVV": (10212806, 151898), "RESOLV": (7239418, 81899)}
problems = []


def bad(msg):
    problems.append(msg)
    print("   PROBLEM: " + msg)


print(f"DELIVERY 3 QUALITY CHECK  ({D3})")

# ---- 1. files and row counts vs manifest ----
man = list(csv.DictReader(open(MAN, newline="", encoding="utf-8-sig")))
low = {c.lower(): c for c in man[0]}
rows_c = next((low[k] for k in ("rows", "row_count", "num_rows") if k in low), None)
miss, mism = 0, 0
for r in man:
    f = os.path.join(D3, *r[low["path"]].split("/"))
    if not os.path.exists(f):
        miss += 1
        continue
    if rows_c and f.endswith(".parquet") and r[rows_c]:
        n = pq.ParquetFile(f).metadata.num_rows
        if n != int(float(r[rows_c])):
            mism += 1
            bad(f"row count {n:,} vs manifest {r[rows_c]} in {r[low['path']]}")
print(f"\n1. Files: {len(man)} in manifest | missing {miss} | Parquet row-count mismatches {mism}"
      f"{'' if rows_c else ' (manifest has no row column)'}")
if miss:
    bad(f"{miss} files missing")

con = duckdb.connect()


# ---- 2. columns same as earlier deliveries ----
def cols(pattern):
    fs = glob.glob(pattern)
    return set(pq.ParquetFile(fs[0]).schema_arrow.names) if fs else None


print("\n2. Columns vs deliveries 1-2")
for kind, new, old in (("orders", f"{D3}/orders/*/*.parquet", f"{OLD}/orders/SOL/*.parquet"),
                       ("trades", f"{D3}/trades/*/*.parquet", f"{OLD}/trades/SOL/*.parquet"),
                       ("snapshots", f"{D3}/snapshots/*.parquet", f"{OLD}/snapshots/*.parquet")):
    a, b = cols(new), cols(old)
    if a is None:
        bad(f"no {kind} files in delivery 3")
        continue
    if b is None:
        print(f"   {kind}: {len(a)} columns (no old file found to compare)")
        continue
    print(f"   {kind}: {len(a)} columns | only in new: {sorted(a - b) or 'none'} | only in old: {sorted(b - a) or 'none'}")
    if b - a:
        bad(f"{kind} lost columns {sorted(b - a)}")

# ---- 3. coin-days ----
cd_f = os.path.join(D3, "coin_days.csv")
if os.path.exists(cd_f):
    cd = pd.read_csv(cd_f)
    dcol = next((c for c in cd.columns if "deliver" in c.lower()), None)
    if dcol:
        delivered = cd[dcol].astype(str).str.lower().isin(["true", "1", "yes", "delivered"])
        print(f"\n3. Coin-days: requested {len(cd)} | delivered {int(delivered.sum())} | not delivered: "
              f"{cd[~delivered].iloc[:, :2].astype(str).agg(' '.join, axis=1).tolist()}")
    else:
        print(f"\n3. Coin-days file has {len(cd)} rows; columns {list(cd.columns)}")
od = sorted(glob.glob(f"{D3}/orders/*/*.parquet"))
print(f"   order files on disk: {len(od)} across {len({os.path.basename(os.path.dirname(f)) for f in od})} coins")

# ---- 4. rows per coin vs README ----
print("\n4. Rows per coin vs vendor README")
print("   coin       order events      README      trades     README")
for coin, (ro, rt) in README_ROWS.items():
    no = sum(pq.ParquetFile(f).metadata.num_rows for f in glob.glob(f"{D3}/orders/{coin}/*.parquet"))
    nt = sum(pq.ParquetFile(f).metadata.num_rows for f in glob.glob(f"{D3}/trades/{coin}/*.parquet"))
    flag = "" if (no == ro and nt == rt) else "   <- differs"
    print(f"   {coin:<9} {no:>13,} {ro:>13,} {nt:>10,} {rt:>10,}{flag}")
    if flag:
        bad(f"{coin} row totals differ from README")

# ---- 5. sample day per coin: duplicates, coin, wallets, start_position ----
wal = set(pd.read_csv(os.path.join(D3, "wallets.csv")).iloc[:, 0].str.lower()) \
    if os.path.exists(os.path.join(D3, "wallets.csv")) else None
print("\n5. One sample day per coin (smallest file)")
for coin in README_ROWS:
    fs = sorted(glob.glob(f"{D3}/orders/{coin}/*.parquet"), key=os.path.getsize)
    ts = sorted(glob.glob(f"{D3}/trades/{coin}/*.parquet"), key=os.path.getsize)
    if not fs or not ts:
        bad(f"{coin}: no files")
        continue
    o = con.execute(f"""SELECT count(*) - count(DISTINCT (block_number, seq, oid, status)),
                               count(DISTINCT coin), max(coin),
                               count(DISTINCT lower(user_address))
                        FROM read_parquet('{fs[0]}')""").fetchone()
    t = con.execute(f"""SELECT count(*) - count(DISTINCT (timestamp, trade_id, user_address, crossed)),
                               count(*) FILTER (WHERE start_position IS NULL), count(*)
                        FROM read_parquet('{ts[0]}')""").fetchone()
    stray = 0
    if wal is not None:
        ws = con.execute(f"SELECT DISTINCT lower(user_address) FROM read_parquet('{fs[0]}')").fetchall()
        stray = sum(w[0] not in wal for w in ws)
    ok = o[0] == 0 and o[1] == 1 and o[2] == coin and t[0] == 0 and t[1] == 0 and stray == 0
    print(f"   {coin:<9} {os.path.basename(fs[0])}: dup order events {o[0]}, coins {o[1]} ({o[2]}), "
          f"wallets {o[3]} (not listed: {stray}) | trades dup {t[0]}, no start_position {t[1]} of {t[2]}"
          f"{'' if ok else '   <- CHECK'}")
    if not ok:
        bad(f"{coin} sample day failed a check")

# ---- 6. vendor rebuild tests ----
rb_f = os.path.join(D3, "rebuild_tests.csv")
if os.path.exists(rb_f):
    rb = pd.read_csv(rb_f)
    print(f"\n6. Vendor rebuild tests: {len(rb)} snapshot points | columns {list(rb.columns)}")
    tcol = next((c for c in rb.columns if "time" in c.lower() or "date" in c.lower()), None)
    mcol = next((c for c in rb.columns if "exact" in c.lower() or "match" in c.lower() or "result" in c.lower()), None)
    if tcol and mcol:
        rb["_t"] = pd.to_datetime(rb[tcol], utc=True, errors="coerce")
        rb["_ok"] = rb[mcol].astype(str).str.lower().isin(["true", "1", "yes", "exact", "match", "pass", "matched"])
        for lab, g in (("before May 1", rb[rb["_t"] < "2026-05-01"]), ("from May 1", rb[rb["_t"] >= "2026-05-01"])):
            print(f"   {lab}: {int(g['_ok'].sum())} of {len(g)} rebuild exactly")

print("\nRESULT: " + ("ALL CHECKS PASSED" if not problems else f"{len(problems)} PROBLEM(S) - see above"))
