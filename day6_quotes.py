"""Day 6 (schedule: Thu Nov 5): the QUOTE TABLE - every quote placed by a firm account, joined to the
market at that moment and to the account's inventory at that moment.

Run:  python E:\\p3\\day6_quotes.py > E:\\p3\\day6_output.txt
(progress on screen, results in the file; about 30-90 minutes)

A quote = an 'open' event of a post-only (Alo) limit order, not reduce-only, not a trigger order
(pre-analysis decision 5). For each quote we record:
  - side, price, size
  - the market at that moment: best bid, best ask, mid, spread (top of book every 0.5 s; the book's
    clock runs 0.25 s late, so we move it 0.25 s earlier - decision 6), and how old that book row is
  - distance from mid in basis points, and how many ticks behind the best price on its own side
  - the account's inventory just before the quote (from Day 5; a position only moves on fills)
Excluded windows (decision 11) are kept but flagged, so nothing silently disappears.
Output: E:\\p3\\data\\quotes\\<COIN>\\<day>.parquet   and 20 random quotes printed for a hand check.
"""
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd

EXPORT = ["E:/hl_export", "E:/hl_export/delivery-2"]
BOOK = r"E:\hl_data\book"
INV = r"E:\p3\data\inventory"
OUT = r"E:\p3\data\quotes"
FIRMS = r"E:\hl_data\firm_wallets_all.csv"
COINS = {"SOL": 2, "XRP": 0}                     # coin -> szDecimals (sets the smallest allowed tick)
DAYS = list(pd.date_range("2026-06-01", "2026-06-14").strftime("%Y-%m-%d")) + \
       list(pd.date_range("2026-08-15", "2026-10-25").strftime("%Y-%m-%d"))
BOOK_SHIFT_S = 0.25                              # book time - 0.25 s = true time (decision 6)
MAX_BOOK_AGE_S = 1.0                             # older book row -> market unknown for that quote
GAPS = [("2026-06-05 22:00:10", "2026-06-05 22:24:16"),      # order-data gaps (decision 11)
        ("2026-08-29 20:21:00", "2026-08-29 20:32:00"),
        ("2026-10-25 09:32:20", "2026-10-25 10:28:57")]
BOOK_GAP_DAYS = {"2026-06-07", "2026-06-08", "2026-06-09", "2026-06-14"}
NS = 1_000_000_000


def log(s):
    print(s, file=sys.stderr, flush=True)


def tick_of(price, sz_dec):
    """Hyperliquid tick: 5 significant figures, at most 6 - szDecimals decimals."""
    p = np.asarray(price, dtype=float)
    t = 10.0 ** (np.floor(np.log10(p)) - 4)
    return np.maximum(t, 10.0 ** -(6 - sz_dec))


def pick(cols, options, what):
    for o in options:
        if o in cols:
            return o
    sys.exit(f"STOP: cannot find the {what} column in the book file. Columns are: {cols}")


fw = pd.read_csv(FIRMS)
firm_of = dict(zip(fw["wallet"].str.lower(), fw["firm"].str.lower()))
con = duckdb.connect()
con.execute("SET TimeZone = 'UTC'")
gap_ns = [(pd.Timestamp(a, tz="UTC").value, pd.Timestamp(b, tz="UTC").value) for a, b in GAPS]

for coin, szd in COINS.items():
    t0 = time.time()
    os.makedirs(os.path.join(OUT, coin), exist_ok=True)
    for old in os.listdir(os.path.join(OUT, coin)):           # no stale files from an earlier run
        if old.endswith(".parquet"):
            os.remove(os.path.join(OUT, coin, old))
    tot = dict(quotes=0, no_file=0, no_prior=0, stale=0, crossed=0, locked=0, gap=0, no_inv=0, near=0,
               near_main=0, unmapped=0, off_grid=0, dup_open=0)
    sample = []
    carry = {}                                   # wallet -> position after its last fill on any earlier day
    for d in DAYS:
        if d == "2026-08-15":
            carry = {}                           # new window: the Day 5 chain restarts too
        of = [p for p in (f"{r}/orders/{coin}/{d}.parquet" for r in EXPORT) if os.path.exists(p)]
        bf = os.path.join(BOOK, coin, f"{coin}_{d}.parquet")
        if not of:
            log(f"   {coin} {d}: no order file")
            continue
        # ---------- the quotes ----------
        ocols = {c[0] for c in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({of}, union_by_name=true)").fetchall()}
        trig = "AND NOT COALESCE(CAST(is_trigger AS BOOLEAN), FALSE)" if "is_trigger" in ocols else ""
        q = con.execute(f"""
            SELECT CAST(oid AS BIGINT) AS oid, lower(user_address) AS wallet, side,
                   CAST(limit_price AS DOUBLE) AS price, CAST(orig_size AS DOUBLE) AS size,
                   epoch_ns(timestamp) AS ts, block_number
            FROM read_parquet({of}, union_by_name=true)
            WHERE status = 'open' AND CAST(tif AS VARCHAR) = 'Alo'
              AND CAST(order_type AS VARCHAR) = 'Limit'
              AND NOT COALESCE(CAST(reduce_only AS BOOLEAN), FALSE) {trig}
            ORDER BY ts""").df()
        if q.empty:
            continue
        bad = ~q["side"].astype(str).isin(["B", "A"]) | ~np.isfinite(q[["price", "size"]]).all(axis=1) \
            | (q["price"] <= 0) | (q["size"] <= 0)
        if bad.any():
            sys.exit(f"STOP on {coin} {d}: {int(bad.sum())} quotes with an unknown side or a bad price/size.")
        tot["dup_open"] += int(q.duplicated("oid").sum())
        q = q.drop_duplicates("oid")                # an order is placed once (decision 5)
        q["firm"] = q["wallet"].map(firm_of)
        tot["unmapped"] += int(q["firm"].isna().sum())

        # ---------- the market at that moment ----------
        if os.path.exists(bf) and d not in BOOK_GAP_DAYS:
            b = pd.read_parquet(bf)
            bc = list(b.columns)
            c_t = pick(bc, ["timestamp", "ts", "time", "local_timestamp"], "time")
            c_b = pick(bc, ["bid_px", "best_bid", "bid_price", "bid", "bid_price_1", "bid_px_0", "bid_0_px"], "best bid")
            c_a = pick(bc, ["ask_px", "best_ask", "ask_price", "ask", "ask_price_1", "ask_px_0", "ask_0_px"], "best ask")
            b = pd.DataFrame({"bts": pd.to_datetime(b[c_t], utc=True, format="ISO8601").dt.as_unit("ns").astype("int64")
                              - int(BOOK_SHIFT_S * NS),
                              "best_bid": pd.to_numeric(b[c_b]), "best_ask": pd.to_numeric(b[c_a])})
            b = b.dropna().drop_duplicates("bts", keep="last").sort_values("bts")
            q = pd.merge_asof(q.sort_values("ts"), b, left_on="ts", right_on="bts", direction="backward")
            q["book_age_s"] = (q["ts"] - q["bts"]) / NS
            q = q.drop(columns="bts")
        else:
            q["best_bid"] = q["best_ask"] = q["book_age_s"] = np.nan
        known = q["book_age_s"].le(MAX_BOOK_AGE_S) & (q["best_ask"] > q["best_bid"]) & (q["best_bid"] > 0)
        if not (os.path.exists(bf) and d not in BOOK_GAP_DAYS):
            tot["no_file"] += len(q)
        else:
            tot["no_prior"] += int(q["book_age_s"].isna().sum())
        tot["stale"] += int((q["book_age_s"] > MAX_BOOK_AGE_S).sum())
        tot["crossed"] += int((q["best_ask"] < q["best_bid"]).sum())
        tot["locked"] += int((q["best_ask"] == q["best_bid"]).sum())
        for c in ("best_bid", "best_ask"):
            q.loc[~known, c] = np.nan
        q["mid"] = (q["best_bid"] + q["best_ask"]) / 2
        q["spread_bp"] = (q["best_ask"] - q["best_bid"]) / q["mid"] * 1e4
        is_bid = q["side"].astype(str) == "B"
        q["rel_mid_bp"] = (q["price"] - q["mid"]) / q["mid"] * 1e4          # signed: + above mid
        q["tick"] = tick_of(q["price"], szd)                              # from the order's OWN price
        units = q["price"] / q["tick"]
        tot["off_grid"] += int((np.abs(units - np.round(units)) > 1e-6).sum())   # want 0
        q["ticks_behind_best"] = np.where(is_bid, (q["best_bid"] - q["price"]),
                                          (q["price"] - q["best_ask"])) / q["tick"]
        q["near_best"] = q["ticks_behind_best"].round(6) <= 2              # main sample (estimand)
        q["above_boundary"] = q["price"] >= (100.0 if coin == "SOL" else 1.0)

        # ---------- the account's inventory just before the quote ----------
        ip = os.path.join(INV, coin, f"{d}.parquet")
        q["inventory"] = np.nan
        if os.path.exists(ip):
            iv = pd.read_parquet(ip, columns=["ts", "wallet", "pos_before", "pos_after"]).sort_values("ts", kind="stable")
            iv["ts"] = iv["ts"].astype("int64")
            q = q.sort_values("ts")
            back = pd.merge_asof(q[["ts", "wallet"]], iv[["ts", "wallet", "pos_after"]], on="ts",
                                 by="wallet", direction="backward", allow_exact_matches=False)
            fwd = pd.merge_asof(q[["ts", "wallet"]], iv[["ts", "wallet", "pos_before"]], on="ts",
                                by="wallet", direction="forward")
            # last fill before the quote -> its 'after'; if none yet, the next fill's 'before' (no fill in
            # between, so the position did not move)
            q["inventory"] = back["pos_after"].to_numpy()
            q["inventory"] = q["inventory"].fillna(pd.Series(fwd["pos_before"].to_numpy(), index=q.index))
        # no fill today before the quote and none after it: the position is still the one after the
        # account's last fill on an earlier day (the chain across days was checked in Day 5)
        q["inventory"] = q["inventory"].fillna(q["wallet"].map(carry))
        if os.path.exists(ip):
            carry.update(iv.groupby("wallet", sort=False)["pos_after"].last().to_dict())
        q["inventory_usd"] = q["inventory"] * q["mid"]
        tot["no_inv"] += int(q["inventory"].isna().sum())

        q["in_gap"] = False
        for a, z in gap_ns:
            q.loc[(q["ts"] >= a) & (q["ts"] <= z), "in_gap"] = True
        tot["gap"] += int(q["in_gap"].sum())
        tot["quotes"] += len(q)
        tot["near"] += int(q["near_best"].sum())
        tot["near_main"] += int((q["near_best"] & ~q["in_gap"]).sum())

        q.to_parquet(os.path.join(OUT, coin, f"{d}.parquet"), index=False)
        ok = q.dropna(subset=["mid", "inventory"])
        if len(ok):
            sample.append(ok.sample(min(2, len(ok)), random_state=len(sample)))
        log(f"   {coin} {d}: {len(q):>10,} quotes | market known {known.mean():6.1%} | "
            f"inventory known {q['inventory'].notna().mean():6.1%} | near best {q['near_best'].mean():6.1%} "
            f"| {time.time() - t0:,.0f}s")

    n = max(tot["quotes"], 1)
    print(f"\n==================== {coin} ====================")
    print(f"   quotes: {tot['quotes']:,} | repeated 'open' rows dropped {tot['dup_open']:,} | "
          f"from wallets not in the firm list {tot['unmapped']:,}")
    print(f"   near the best price (within 2 ticks): all {tot['near']:,} ({tot['near'] / n:.1%}) | "
          f"MAIN SAMPLE (outside data gaps) {tot['near_main']:,}")
    print(f"   prices off the legal tick grid: {tot['off_grid']:,} (want 0)")
    print(f"   market unknown: no book that day {tot['no_file']:,} | no earlier book row {tot['no_prior']:,} | "
          f"book older than {MAX_BOOK_AGE_S:.0f} s {tot['stale']:,} | crossed {tot['crossed']:,} | "
          f"locked (bid = ask) {tot['locked']:,}")
    print(f"   inventory unknown: {tot['no_inv']:,} ({tot['no_inv'] / n:.2%})  (accounts with no SOL/XRP fill in "
          f"the window - their inventory is unknown, not zero)")
    print(f"   inside an excluded data gap (kept, flagged in_gap): {tot['gap']:,}")
    if sample:
        s = pd.concat(sample).sample(min(20, sum(len(x) for x in sample)), random_state=1)
        s["time"] = pd.to_datetime(s["ts"], unit="ns", utc=True).dt.strftime("%m-%d %H:%M:%S.%f").str[:-3]
        print("\n   HAND CHECK: 20 random quotes. For each, check: a bid sits at or below best_bid, an ask at or "
              "above best_ask;\n   rel_mid_bp = (price - mid) / mid x 10,000; ticks_behind = distance to own best "
              "/ tick; book_age <= 1 s.")
        cols = ["time", "wallet", "side", "price", "size", "best_bid", "best_ask", "mid", "rel_mid_bp",
                "ticks_behind_best", "book_age_s", "inventory"]
        s["wallet"] = s["wallet"].str[:8]
        print(s[cols].round({"rel_mid_bp": 2, "ticks_behind_best": 1, "book_age_s": 3, "mid": 6})
              .to_string(index=False))
    print(f"   ({time.time() - t0:,.0f} seconds)")
print()
