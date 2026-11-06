"""Day 5 (schedule: Wed Nov 4): inventory of every firm account over time, plus firm totals.

Run:  python E:\\p3\\day5_inventory.py > E:\\p3\\day5_output.txt
(progress on screen, results in the file; about 30-60 minutes)

Idea: every trade row carries the account's position just BEFORE the trade (start_position).
      position after = before + size (buy) or - size (sell).
      On Hyperliquid a position only changes through fills (trades, liquidations, deleveraging are all
      fills), so between two fills it is constant - that is what lets us fill the minutes in between.
      CHECK: one fill's "after" must equal the account's next fill's "before"  (want >= 99.9%).
Inputs:  our whole-market trades in E:\\hl_data\\<COIN>\\trades\\  +  E:\\hl_data\\firm_wallets_all.csv
Outputs (data, not uploaded to GitHub):
  E:\\p3\\data\\inventory\\<COIN>\\<day>.parquet         every fill of every firm account, position before/after
  E:\\p3\\data\\inventory\\<COIN>_firm_minute.parquet    each firm's position at the end of every minute,
      summed over the firm's accounts that have traded in the window so far (an account that never trades
      in the window is not visible in trade data, so this is "traded-account inventory", not a full audit)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

OURS = r"E:\hl_data"
OUT = r"E:\p3\data\inventory"
COINS = ["SOL", "XRP"]
WINDOWS = {"June": pd.date_range("2026-06-01", "2026-06-14"),
           "Aug-Oct": pd.date_range("2026-08-15", "2026-10-25")}
TOL = 1e-6
BUY, SELL = {"B"}, {"A", "S"}


def log(s):
    print(s, file=sys.stderr, flush=True)


def close(a, b):
    return abs(a - b) < TOL * max(1.0, abs(a), abs(b))


def chain_order(sp, sg, ts, diag):
    """Order one account's fills so each starts where the previous ended (fills in the same
    millisecond arrive in unknown order). Counts same-millisecond groups in diag."""
    order, i, n = [], 0, len(sp)
    while i < n:
        j = i
        while j < n and ts[j] == ts[i]:
            j += 1
        grp = list(range(i, j))
        if len(grp) > 1:
            diag["groups"] += 1
        clean = True
        size0 = len(grp)
        pos = sp[order[-1]] + sg[order[-1]] if order else None
        while grp:
            cands = [idx for idx, r in enumerate(grp) if pos is not None and close(sp[r], pos)]
            if cands:
                if len(cands) > 1 and size0 > 1:
                    clean = False                # two fills could both come next
                k = cands[0]
            else:                                # no fill continues: start from the one nothing leads into
                ends = [sp[r] + sg[r] for r in grp]
                heads = [idx for idx, r in enumerate(grp) if not any(close(sp[r], e) for e in ends)]
                if size0 > 1 and (len(heads) != 1 or pos is not None):
                    clean = False                # no unique start, or the chain broke inside the group
                k = heads[0] if heads else 0
            r = grp.pop(k)
            order.append(r)
            pos = sp[r] + sg[r]
        if j - i > 1:
            diag["clean" if clean else "unclear"] += 1
        i = j
    return np.array(order, dtype=np.int64)


# ---------- the wallet -> firm list must be clean ----------
fw = pd.read_csv(os.path.join(OURS, "firm_wallets_all.csv"))
fw["wallet"] = fw["wallet"].str.lower().str.strip()
fw["firm"] = fw["firm"].str.lower().str.strip()
n_null = int(fw[["wallet", "firm"]].isna().any(axis=1).sum())
conflicts = fw.dropna().groupby("wallet")["firm"].nunique()
n_conf = int((conflicts > 1).sum())
n_dup = int(fw.duplicated("wallet").sum())
print(f"\nWALLET LIST: {len(fw):,} rows | empty wallet/firm {n_null} | duplicate wallets {n_dup} | "
      f"wallets in 2+ firms {n_conf}")
if n_null or n_conf:
    sys.exit("   STOP: the wallet list has empty or conflicting rows - fix it first.")
fw = fw.drop_duplicates("wallet")
firm_of = dict(zip(fw["wallet"], fw["firm"]))
print(f"Firms: {fw['firm'].nunique()} | firm accounts: {len(firm_of):,}")

for coin in COINS:
    t0 = time.time()
    os.makedirs(os.path.join(OUT, coin), exist_ok=True)
    for old in os.listdir(os.path.join(OUT, coin)):           # no stale files from an earlier run
        if old.endswith(".parquet"):
            os.remove(os.path.join(OUT, coin, old))
    links = ok_links = unverified = fills_total = dup_rows = lookalike_rows = 0
    missing_days, empty_days = [], 0
    diag = {"groups": 0, "clean": 0, "unclear": 0}
    per_wallet = {}                              # wallet -> [links, ok]
    firm_minutes = []
    seen_wallets = set()
    for wname, days in WINDOWS.items():
        prev_end = {}                            # wallet -> position after its last fill (restarts per window)
        prev_day_end = {}                        # wallet -> position at the end of the previous day
        for d in days.strftime("%Y-%m-%d"):
            p = os.path.join(OURS, coin, "trades", f"{coin}_{d}.parquet")
            if not os.path.exists(p):
                missing_days.append(d)
                unverified += len(prev_end)      # these accounts lose their baseline
                prev_end, prev_day_end = {}, {}  # unknown until each account's next fill re-baselines it
                log(f"   {coin} {d}: NO TRADE FILE - every account's baseline restarted")
                continue
            t = pd.read_parquet(p, columns=["timestamp", "user_address", "side", "price", "size", "start_position", "tx_hash", "trade_id"])
            t["user_address"] = t["user_address"].astype(str).str.lower()
            t = t[t["user_address"].isin(firm_of)].copy()

            if not t.empty:
                # ---- input checks: side, numbers ----
                side = t["side"].astype(str)
                bad_side = ~side.isin(BUY | SELL)
                for col in ("price", "size", "start_position"):
                    t[col] = pd.to_numeric(t[col], errors="coerce")
                bad_num = ~np.isfinite(t[["price", "size", "start_position"]]).all(axis=1) | (t["size"] <= 0) | (t["price"] <= 0)
                if bad_side.any() or bad_num.any():
                    sys.exit(f"   STOP on {coin} {d}: {int(bad_side.sum())} rows with unknown side "
                             f"{side[bad_side].value_counts().to_dict()}, {int(bad_num.sum())} rows with bad numbers.")
                # a duplicate = the same execution twice (same tx and trade id). Two fills that only LOOK alike
                # (one order hit by two counterparties at once) have different trade ids and are both real.
                n_dup = int(t.duplicated(["tx_hash", "trade_id", "user_address"]).sum())
                if n_dup:
                    sys.exit(f"   STOP on {coin} {d}: {n_dup} trades with the same trade id twice - investigate.")
                lookalike_rows += int(t.duplicated(["timestamp", "user_address", "side", "price", "size",
                                                    "start_position"]).sum())
                tt = pd.to_datetime(t["timestamp"], utc=True, format="ISO8601", errors="coerce")
                if tt.isna().any():
                    sys.exit(f"   STOP on {coin} {d}: {int(tt.isna().sum())} rows with an unreadable timestamp.")
                t["ts"] = tt.dt.as_unit("ns").astype("int64")
                t["sp"] = t["start_position"]
                t["sg"] = np.where(side.isin(BUY), 1.0, -1.0) * t["size"]
                t = t.sort_values(["user_address", "ts"], kind="stable").reset_index(drop=True)

                parts = []
                for w, g in t.groupby("user_address", sort=False):
                    o = chain_order(g["sp"].to_numpy(), g["sg"].to_numpy(), g["ts"].to_numpy(), diag)
                    g = g.iloc[o].copy()
                    g["pos_after"] = g["sp"] + g["sg"]
                    before = g["pos_after"].shift(1)
                    if w in prev_end:
                        before.iloc[0] = prev_end[w]
                    ok = pd.Series([close(a, b) if pd.notna(a) else False
                                    for a, b in zip(before, g["sp"])], index=g.index)
                    g["linked"] = ok.astype(float).where(before.notna())   # 1 linked, 0 broken, NaN = nothing to link to
                    has = before.notna()
                    n_l, n_ok = int(has.sum()), int(ok[has].sum())
                    links += n_l
                    ok_links += n_ok
                    pw = per_wallet.setdefault(w, [0, 0])
                    pw[0] += n_l
                    pw[1] += n_ok
                    prev_end[w] = float(g["pos_after"].iloc[-1])
                    parts.append(g)
                day_df = pd.concat(parts, ignore_index=True)
                day_df["firm"] = day_df["user_address"].map(firm_of)
                fills_total += len(day_df)
                seen_wallets.update(day_df["user_address"].unique())
                day_df[["ts", "user_address", "firm", "side", "price", "size", "sp", "pos_after", "linked"]].rename(
                    columns={"user_address": "wallet", "sp": "pos_before"}).to_parquet(
                    os.path.join(OUT, coin, f"{d}.parquet"), index=False)
            else:
                empty_days += 1
                day_df = pd.DataFrame(columns=["ts", "user_address", "sp", "pos_after"])

            # ---- each firm's position at the end of every minute (positions only move on fills) ----
            day_min0 = int(pd.Timestamp(d, tz="UTC").value // 60_000_000_000)
            minutes = np.arange(day_min0, day_min0 + 1440)
            traded = set(day_df["user_address"]) if len(day_df) else set()
            cols = sorted(traded | set(prev_end))
            if not cols:
                continue
            piv = pd.DataFrame(np.nan, index=minutes, columns=cols)
            if len(day_df):
                mins = day_df["ts"].astype("int64") // 60_000_000_000
                last = day_df.assign(minute=mins).groupby(["user_address", "minute"], sort=False)["pos_after"] \
                    .last().unstack(0)
                piv.update(last.reindex(index=minutes))
                first_sp = day_df.groupby("user_address", sort=False)["sp"].first()
            else:
                first_sp = pd.Series(dtype=float)
            # start of the day: position before the account's first fill of the day. If it already traded
            # earlier in the window, that equals yesterday's end (checked by the chain); else it is the
            # 'before' of today's first fill, valid because nothing but a fill moves a position.
            start_pos = {w: (first_sp[w] if w in first_sp.index else prev_day_end.get(w, np.nan)) for w in cols}
            piv.iloc[0] = piv.iloc[0].fillna(pd.Series(start_pos, dtype=float))
            piv = piv.ffill()
            prev_day_end = piv.iloc[-1].to_dict()
            firm_tot = piv.T.groupby(pd.Series({w: firm_of[w] for w in cols})).sum(min_count=1).T
            firm_tot.index = pd.to_datetime(firm_tot.index * 60, unit="s", utc=True)
            firm_tot.index.name = "minute"
            firm_minutes.append(firm_tot.stack().rename("position").reset_index().rename(columns={"level_1": "firm"}))
            log(f"   {coin} {wname} {d}: {len(day_df):>9,} firm fills | {len(traded):>4} accounts traded "
                f"| chain so far {ok_links / max(links, 1):.4%} | {time.time() - t0:,.0f}s")

    fm = pd.concat(firm_minutes, ignore_index=True) if firm_minutes else pd.DataFrame()
    if len(fm):
        fm.to_parquet(os.path.join(OUT, f"{coin}_firm_minute.parquet"), index=False)

    rate = ok_links / links if links else float("nan")
    rates = pd.Series({w: ok / n for w, (n, ok) in per_wallet.items() if n >= 50})
    print(f"\n==================== {coin} ====================")
    print(f"   days: missing trade file {len(missing_days)} {missing_days[:5]} | days with no firm fills {empty_days}")
    print(f"   look-alike fills (same order hit twice at once; same 'before' reported on both): {lookalike_rows:,}")
    print(f"   firm fills: {fills_total:,} | true duplicates (same trade id) {dup_rows:,} (want 0) | "
          f"accounts that traded: {len(seen_wallets):,} of {len(firm_of):,}")
    complete = not missing_days and dup_rows == 0
    print(f"   CHAIN: {ok_links:,} of {links:,} links match = {rate:.4%}  -> "
          f"{'PASS' if rate >= 0.999 else 'CHECK'} (want >= 99.9%)")
    print(f"      accounts whose baseline restarted after a missing day: {unverified:,}")
    print(f"   DATA COMPLETENESS: {'PASS' if complete else 'CHECK'} (no missing days, no duplicates)")
    print(f"   OVERALL: {'PASS' if complete and rate >= 0.999 else 'CHECK'}")
    print(f"   same-millisecond groups: {diag['groups']:,} | one clear order {diag['clean']:,} | "
          f"unclear order {diag['unclear']:,} (positions still add up; only the order inside the ms is unknown)")
    print(f"   accounts below 99% (with 50+ links): {(rates < 0.99).sum()} of {len(rates)}")
    for w, r in rates[rates < 0.99].sort_values().head(8).items():
        print(f"      {w[:10]}... firm {firm_of[w][:10]}...: {r:.2%} over {per_wallet[w][0]:,} links")
    if len(fm):
        print(f"   firm-minute table (traded-account inventory): {len(fm):,} rows, {fm['firm'].nunique()} firms")
        big = fm.groupby("firm")["position"].agg(lambda s: s.abs().max()).sort_values(ascending=False).head(5)
        print("   largest firm positions ever held (absolute, in coins):")
        for f, v in big.items():
            print(f"      firm {f[:10]}...: {v:,.0f} {coin}")
    print(f"   ({time.time() - t0:,.0f} seconds)")
print()
