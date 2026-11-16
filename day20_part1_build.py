"""Part 1 build (prereg-v3): one table per coin-day with each sampled quote's q, vol5 and outcomes."""
import glob
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd

QUOTES = os.environ.get("P3_QUOTES", r"E:\p3\data\quotes")
INV = os.environ.get("P3_INV", r"E:\p3\data\inventory")
OUT = os.environ.get("P3_PART1", r"E:\p3\part1")
COINS = [c.upper() for c in sys.argv[1:]] or ["SOL", "XRP", "BTC", "ETH", "HYPE"]
BAND_BP, CLIP, STALE_S, SAMPLE = 5.0, 5.0, 5, 5
NS = 1_000_000_000
MIN_NS = 60 * NS
con = duckdb.connect()


def log(s):
    print(s, flush=True)


def write_empty(out, coin, d):
    """A day without usable quotes: an empty table with the normal columns (written atomically)."""
    pd.DataFrame({c: pd.Series(dtype=t) for c, t in (("coin", "object"), ("day", "object"), ("wallet", "object"),
                  ("ts", "int64"), ("bid", "int8"), ("q", "float64"), ("q_raw", "float64"), ("vol5", "float64"),
                  ("price_bp", "float64"), ("log_size", "float64"), ("tick_bp", "float64"), ("at_best", "int8"))}
                 ).to_parquet(out + ".part", index=False)
    os.replace(out + ".part", out)
    diag.append(dict(coin=coin, day=d, sampled=0, no_q=0, no_vol5=0, bad_size=0, kept=0))


def prev_day(d):
    return (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


t00 = time.time()
diag = []
for coin in COINS:
    days = sorted({os.path.basename(f)[:10] for f in glob.glob(os.path.join(QUOTES, coin, "*.parquet"))})
    if not days:
        log(f"{coin}: no quote files - skipped")
        continue
    os.makedirs(os.path.join(OUT, coin), exist_ok=True)
    for d in days:
        out = os.path.join(OUT, coin, f"{d}.parquet")
        if os.path.exists(out):
            continue
        t0 = time.time()
        qf = sorted(glob.glob(os.path.join(QUOTES, coin, f"{d}*.parquet")))
        cols = set(con.execute(f"DESCRIBE SELECT * FROM read_parquet({qf}, union_by_name=true)").df()["column_name"])
        good = "book_quality = 'GOOD'" if "book_quality" in cols else "hour_crossed IS NOT NULL AND NOT poor_book"

        # vol5 from all quotes' mids (frozen code), per second
        D0s = pd.Timestamp(d, tz="UTC").value // NS
        vf = sorted(glob.glob(os.path.join(QUOTES, coin, f"{prev_day(d)}*.parquet"))) + qf  # last 5 min of the day before
        ms = con.execute(f"""SELECT ts // {NS} AS sec, arg_max(mid, ts) AS mid
                             FROM read_parquet({vf}, union_by_name=true)
                             WHERE mid IS NOT NULL AND ts >= {(D0s - 300) * NS}
                             GROUP BY 1 ORDER BY 1""").df()
        if ms.empty:                             # no market price all day (e.g. an order-book outage day)
            log(f"   {coin} {d}: no quote has a market price - empty day")
            write_empty(out, coin, d)
            continue
        m = pd.Series(ms["mid"].to_numpy(), index=ms["sec"].to_numpy())
        full = np.arange(m.index.min(), m.index.max() + 1)
        seen = pd.Series(full, index=full).where(pd.Series(full, index=full).isin(m.index)).ffill()
        age = full - seen.to_numpy()
        m = m.reindex(full).ffill()
        fresh = age <= STALE_S
        ret = (np.log(m).diff() * 1e4).where(fresh & np.r_[False, fresh[:-1]])
        vol = ret.rolling(300, min_periods=150).std().shift(1)

        # the sampled main-sample quotes
        q = con.execute(f"""
            SELECT lower(wallet) AS wallet, ts, side, mid, price, size, inventory, tick, ticks_behind_best
            FROM read_parquet({qf}, union_by_name=true)
            WHERE (ticks_behind_best * tick / mid * 1e4 <= {BAND_BP}) AND NOT in_gap AND {good}
              AND NOT crosses_book AND inventory IS NOT NULL AND mid IS NOT NULL
              AND oid % {SAMPLE} = 0""").df()
        if q.empty:
            log(f"   {coin} {d}: no sampled quotes")
            write_empty(out, coin, d)
            continue
        n0 = len(q)

        # q: inventory / std of the minute-by-minute position over the previous 1,440 minutes (frozen rule)
        invf = [f for f in (os.path.join(INV, coin, f"{x}.parquet") for x in (prev_day(d), d)) if os.path.exists(f)]
        D0 = pd.Timestamp(d, tz="UTC").value // MIN_NS
        sd = pd.DataFrame(columns=["minute", "wallet", "sd"])
        if invf:
            tcol = con.execute(f"DESCRIBE SELECT ts FROM read_parquet({invf}, union_by_name=true)").fetchone()[1]
            T = "epoch_ns(ts)" if "TIMESTAMP" in tcol.upper() else "ts"
            pm = con.execute(f"""SELECT lower(wallet) AS wallet, {T} // {MIN_NS} AS minute,
                                        arg_max(pos_after, {T}) AS pos
                                 FROM read_parquet({invf}, union_by_name=true) GROUP BY 1, 2""").df()
            pm = pm[pm["wallet"].isin(set(q["wallet"]))]
            if len(pm):
                P = pm.pivot_table(index="minute", columns="wallet", values="pos", aggfunc="last")
                P = P.reindex(np.arange(min(P.index.min(), D0 - 1440), D0 + 1440)).ffill()
                S = P.rolling(1440, min_periods=1440).std().shift(1).loc[D0:D0 + 1439]
                sd = S.stack().rename("sd").reset_index().rename(columns={"level_0": "minute"})
                sd = sd[sd["sd"] > 0]
        q = q.assign(minute=q["ts"] // MIN_NS).merge(sd, on=["minute", "wallet"], how="left")
        q["q_raw"] = q["inventory"] / q["sd"]
        q["q"] = q["q_raw"].clip(-CLIP, CLIP)
        q["vol5"] = vol.reindex(q["ts"] // NS).to_numpy()
        n_noq = int(q["q"].isna().sum())
        n_novol = int((q["q"].notna() & q["vol5"].isna()).sum())
        n_size = int((q["q"].notna() & q["vol5"].notna() & ~(q["size"] > 0)).sum())
        q = q[q["q"].notna() & q["vol5"].notna() & (q["size"] > 0)]
        res = pd.DataFrame({
            "coin": coin, "day": d, "wallet": q["wallet"].to_numpy(), "ts": q["ts"].to_numpy(),
            "bid": (q["side"] == "B").astype(np.int8).to_numpy(), "q": q["q"].to_numpy(),
            "q_raw": q["q_raw"].to_numpy(), "vol5": q["vol5"].to_numpy(),
            "price_bp": ((q["price"] - q["mid"]) / q["mid"] * 1e4).to_numpy(),
            "log_size": np.log(q["size"].to_numpy(float)),
            "tick_bp": (q["tick"] / q["mid"] * 1e4).to_numpy(),
            "at_best": (q["ticks_behind_best"].round(6) <= 0).astype(np.int8).to_numpy()})
        res.to_parquet(out + ".part", index=False)
        os.replace(out + ".part", out)
        diag.append(dict(coin=coin, day=d, sampled=n0, no_q=n_noq, no_vol5=n_novol, bad_size=n_size, kept=len(res)))
        log(f"   {coin} {d}: kept {len(res):,} of {n0:,} sampled (no q {n_noq:,}, no vol5 {n_novol:,}, size {n_size}) "
            f"| {res['wallet'].nunique()} makers | {time.time() - t0:,.0f}s "
            f"| total {(time.time() - t00) / 60:,.0f} min")
if diag:
    G = pd.DataFrame(diag)
    G.to_csv(os.path.join(OUT, f"coverage_{time.strftime('%Y%m%d_%H%M')}.csv"), index=False)
    print("\nCOVERAGE BY COIN (this run)")
    print(G.groupby("coin")[["sampled", "no_q", "no_vol5", "bad_size", "kept"]].sum().assign(
        kept_pct=lambda x: (100 * x["kept"] / x["sampled"]).round(1)).to_string())
log(f"\nDone in {(time.time() - t00) / 3600:.1f} h. Next: day21_part1_tests.py")
