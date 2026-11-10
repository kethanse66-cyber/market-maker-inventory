"""Day 10: NATIVE makers vs HEDGERS (pre-analysis decisions 18 and 19). About 30-90 minutes.

Run:  python E:\\p3\\day10_natives.py > E:\\p3\\day10_output.txt          (all 5 coins)
Saves: E:\\p3\\data\\natives.csv  - one row per unit at three levels (account, sleeve, firm)

The idea: a maker's Hyperliquid inventory is its whole risk ONLY if it does not offset it elsewhere.
  - NATIVE: works inventory off on Hyperliquid itself, fast.
  - HEDGER: its Hyperliquid fills are followed by Binance price moves - consistent with hedging on
    Binance (evidence, not proof: common information can do this too).
  - SLOW: inventory takes 60+ minutes to halve, with no Binance footprint.
  - INCOMPLETE: a required test could not be run.
  Not finding a footprint does not prove a maker is native (it could hedge on another venue, another
  instrument, or faster than 1 s). This is an empirical classification.

Decision 18: native = half-life under 60 min AND flow does not predict Binance 1-60 s ahead (5%).
Decision 19: two speeds, both required: 1-s bins with 10 lags (1-10 s), 10-s bins with 6 lags (10-60 s).
Implementation:
  - Granger: Binance log mid change (bp) on its own lags + lags of the unit's net flow (scaled by its std).
    Newey-West (HAC) covariance, bandwidth = 2 x lags, Wald test on the flow lags.
  - Benjamini-Hochberg FDR at 5% within each coin, speed and level. HEDGER if either corrected test is
    significant; NATIVE needs both tests valid and neither significant, plus half-life under 60 min.
  - Placebo (accounts): each account's flow paired with Binance moves from OTHER test days - 3 random
    day permutations with no day left in place (fixed seed). A diagnostic of the 5% size, not a precise
    estimate (3 permutations, shared across accounts, results not independent).
  - Time series: no lag reaches into another day (the first lags+1 bins of each day are not targets);
    Newey-West uses the true bin spacing (missing bins stay as gaps, products across days are dropped);
    a bin is usable only if its stalest Binance second is at most 5 s old.
  - Labels: SLOW needs a real half-life estimate of 60+ min; no estimate -> INCOMPLETE.
  - A test is valid only with 1,000+ usable target bins, 50+ bins with flow, full rank and condition
    number under 1e10; otherwise its p-value is empty (never enters the FDR) and the unit can be INCOMPLETE.
  - Self-check at start: the fast (Frisch-Waugh) test is compared with the full regression on simulated
    multi-day data with autocorrelated returns, zero and non-zero effects - coefficients, covariance and
    Wald statistic must match.
  - The 1 s and 10 s tests use the same market activity: they are overlapping views, not two independent
    confirmations. Either one can mark a hedger (decision 19).
  - Bins are built day by day (bin = day x bins-per-day + bin in day), identical for prices and flows.
    Binance price = last observation in each second (ordered by the raw timestamp); a price older than
    5 s is stale and that bin's return is dropped (set to 0 and excluded as a target).
  - Half-life: minute-end net position, q_t = a + phi q_{t-1}, HL = ln 0.5 / ln phi; also by subperiod
    (Aug 15 - Sep 26 and Sep 27 - Oct 25) as a stability check. A position is held flat only between fills
    on days that HAVE data; minutes on a missing inventory day are unknown and dropped from the fit, and
    after a gap the position restarts from the first fill's "before" position.
  - Levels: account (decision 18 classification), sleeve and firm (flows and positions summed first).
11 test days spread over the window (every 7th day; HYPE only from Sep 24). Accounts: 500+ fills.
"""
import glob
import re
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

INV = os.environ.get("P3_INV", r"E:\p3\data\inventory")
BIN = os.environ.get("P3_BINANCE", r"E:\hl_data\binance")
SLEEVES = os.environ.get("P3_SLEEVES", r"E:\p3\data\sleeves.csv")
OUT = os.environ.get("P3_NATIVES", r"E:\p3\data\natives.csv")
COINS = [c.upper() for c in sys.argv[1:]] or ["SOL", "XRP", "BTC", "ETH", "HYPE"]
DAYS = list(pd.date_range("2026-08-15", "2026-10-25").strftime("%Y-%m-%d"))
SPLIT = "2026-09-27"
TEST_DAYS = DAYS[::7]
BIN_START = {"HYPE": "2026-09-24"}
MIN_FILLS, HALF_LIFE_MAX, ALPHA, STALE_S = 500, 60.0, 0.05, 5
SPEEDS = [("1s", 1, 10), ("10s", 10, 6)]                # bin seconds, lags
NS = 1_000_000_000
N_PLACEBO = 3
MIN_TARGETS, MIN_FLOW_BINS, MAX_COND = 1000, 50, 1e10
rng = np.random.default_rng(20261110)


def log(s):
    print(s, file=sys.stderr, flush=True)


def tns(con, files, coin):
    tcol = con.execute(f"DESCRIBE SELECT ts FROM read_parquet({files}, union_by_name=true)").fetchone()[1]
    if "TIMESTAMP" in tcol.upper():
        return "epoch_ns(ts)"
    mx = con.execute(f"SELECT max(ts) FROM read_parquet({files}, union_by_name=true)").fetchone()[0]
    mult = {19: 1, 16: 1_000, 13: 1_000_000, 10: 1_000_000_000}.get(len(str(int(mx))))
    if mult is None:
        sys.exit(f"STOP: cannot tell the time unit of ts in {coin} inventory (max {mx})")
    return f"(ts * {mult})"


def binance_seconds(con, coin, days):
    """Last Binance price in each second (ordered by the raw timestamp) on the test days."""
    sym = re.compile(rf"(^|[^A-Z0-9]){coin}(USDT|USDC|USD)?([^A-Z0-9]|$)")   # exact symbol, not e.g. SOLV
    files = [f for f in glob.glob(os.path.join(BIN, "**", "*"), recursive=True)
             if os.path.isfile(f) and f.lower().endswith((".parquet", ".csv", ".csv.gz"))
             and sym.search(os.path.relpath(f, BIN).upper().replace("\\", "/"))]
    pq = [f for f in files if f.lower().endswith(".parquet")]
    files, reader = (pq, "read_parquet") if pq else (files, "read_csv_auto")
    if not files:
        return None, f"no Binance files with {coin} in the name under {BIN}"
    print(f"      Binance files for {coin}: {len(files)} | first {os.path.relpath(files[0], BIN)} | "
          f"last {os.path.relpath(sorted(files)[-1], BIN)}")
    real = list(con.execute(f"DESCRIBE SELECT * FROM {reader}({files[:1]})").df()["column_name"])
    name = {c.lower(): c for c in real}
    if "symbol" in name:
        syms = con.execute(f'SELECT DISTINCT "{name["symbol"]}" FROM {reader}({files}, union_by_name=true)').fetchall()
        bad = [x[0] for x in syms if not sym.search(str(x[0]).upper())]
        if bad:
            return None, f"STOP-worthy: other symbols inside the {coin} files: {bad[:5]}"
    t = next((name[c] for c in ("open_time", "timestamp", "ts", "time", "t", "datetime", "date") if c in name), None)
    if "bid" in name and "ask" in name:
        px = f'("{name["bid"]}" + "{name["ask"]}") / 2'
    elif "bid_price" in name and "ask_price" in name:
        px = f'("{name["bid_price"]}" + "{name["ask_price"]}") / 2'
    else:
        p = next((name[c] for c in ("close", "mid", "price", "c", "close_price", "last") if c in name), None)
        px = f'"{p}"' if p else None
    if t is None or px is None:
        return None, f"cannot find time/price columns in {files[0]}: {real}"
    ttype = con.execute(f'SELECT typeof("{t}") FROM {reader}({files[:1]}) LIMIT 1').fetchone()[0].upper()
    if "TIMESTAMP" in ttype or "DATE" in ttype:
        raw_ns = f'epoch_ns("{t}")'
    else:
        mx = con.execute(f'SELECT max("{t}") FROM {reader}({files[:1]})').fetchone()[0]
        mult = {19: 1, 16: 1_000, 13: 1_000_000, 10: NS}.get(len(str(int(mx))))
        if mult is None:
            return None, f"cannot tell the time unit of {t} (max {mx})"
        raw_ns = f'(CAST("{t}" AS BIGINT) * {mult})'
    lo = pd.Timestamp(min(days), tz="UTC").value
    hi = (pd.Timestamp(max(days), tz="UTC") + pd.Timedelta(days=1)).value
    b = con.execute(f"""SELECT {raw_ns} // {NS} AS s, arg_max(CAST({px} AS DOUBLE), {raw_ns}) AS px
                        FROM {reader}({files}, union_by_name=true)
                        WHERE {raw_ns} >= {lo} AND {raw_ns} < {hi} GROUP BY 1 ORDER BY 1""").df()
    return b, f"{len(files)} files, time {t} ({ttype}), price {px}"


_BASE = {}


def hac_sum(u_rows, pos, day, n, lags):
    """Newey-West meat with TRUE time spacing: scores sit at their original bin positions (missing bins =
    0) and each day is followed by 2 x lags empty bins, so no product ever spans two days. Bandwidth 2 x lags."""
    L = 2 * lags
    U = np.zeros((n + (int(day[-1]) + 1) * L, u_rows.shape[1]))
    U[pos + day[pos] * L] = u_rows
    S = u_rows.T @ u_rows
    for j in range(1, L + 1):
        G = U[j:].T @ U[:-j]
        S += (1 - j / (L + 1)) * (G + G.T)
    return S


def hac_granger(r, x, lags, valid, day, full_out=False):
    """Newey-West Wald test that lags of x predict r beyond r's own lags. valid = usable target bins (the
    caller already drops the first lags+1 bins of each day, so no lag reaches into another day).
    Computed fast: r's own lags and the constant are partialled out once per return series
    (Frisch-Waugh), then only the flow lags enter the sandwich (checked against the full model at start)."""
    n = len(r)
    key = (id(r), lags)
    if key not in _BASE:
        keep = valid[lags:]
        Q = np.column_stack([np.ones(n - lags)] + [r[lags - k:n - k] for k in range(1, lags + 1)])[keep]
        y = r[lags:][keep]
        QtQ_i = np.linalg.pinv(Q.T @ Q)
        _BASE.clear()
        _BASE[key] = (keep, Q, QtQ_i, y - Q @ (QtQ_i @ (Q.T @ y)), np.arange(lags, n)[keep])
    keep, Q, QtQ_i, y_res, pos = _BASE[key]
    if len(y_res) < MIN_TARGETS or ((x[lags:] != 0) & keep).sum() < MIN_FLOW_BINS:   # usable flow bins only
        return (np.nan, np.nan) if not full_out else (None, None, np.nan)
    Xx = np.column_stack([x[lags - k:n - k] for k in range(1, lags + 1)])[keep]
    Xr = Xx - Q @ (QtQ_i @ (Q.T @ Xx))
    A = Xr.T @ Xr
    if np.linalg.matrix_rank(A) < lags or np.linalg.cond(A) > MAX_COND:
        return (np.nan, np.nan) if not full_out else (None, None, np.nan)
    A_i = np.linalg.inv(A)
    beta = A_i @ (Xr.T @ y_res)
    S = hac_sum(Xr * (y_res - Xr @ beta)[:, None], pos, day, n, lags)
    V = A_i @ S @ A_i
    W = float(beta @ np.linalg.pinv(V) @ beta)
    if full_out:
        return beta, V, W
    return float(stats.chi2.sf(W, lags)), float(beta.sum())


def full_granger(r, x, lags, valid, day):
    """The plain full regression with the same Newey-West covariance - used only for the self-check."""
    n = len(r)
    keep = valid[lags:]
    X = np.column_stack([np.ones(n - lags)] + [r[lags - k:n - k] for k in range(1, lags + 1)] +
                        [x[lags - k:n - k] for k in range(1, lags + 1)])[keep]
    y = r[lags:][keep]
    Ai = np.linalg.inv(X.T @ X)
    b = Ai @ (X.T @ y)
    S = hac_sum(X * (y - X @ b)[:, None], np.arange(lags, n)[keep], day, n, lags)
    V = Ai @ S @ Ai
    k = slice(1 + lags, 1 + 2 * lags)
    return b[k], V[k, k], float(b[k] @ np.linalg.inv(V[k, k]) @ b[k])


def bh(p):
    """Benjamini-Hochberg: True where significant at FDR ALPHA (NaN -> False)."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.zeros(len(p), bool)
    if ok.sum():
        q = p[ok]
        o = np.argsort(q)
        passed = q[o] <= ALPHA * np.arange(1, len(q) + 1) / len(q)
        kmax = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
        sig = np.zeros(len(q), bool)
        sig[o[:kmax]] = True
        out[ok] = sig
    return out


def half_life(s):
    a, b = s[:-1], s[1:]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 100 or np.std(a) == 0:
        return np.nan, np.nan
    phi = np.polyfit(a, b, 1)[0]
    return (np.log(0.5) / np.log(phi) if 0 < phi < 1 else (np.inf if phi >= 1 else 0.0)), phi


# ---------- self-checks on simulated data (3 days, autocorrelated returns, gaps) ----------
D_, PD_ = 3, 20000
n_ = D_ * PD_
day_ = np.arange(n_) // PD_
for eff in (0.0, -0.02):
    e_ = rng.normal(size=n_)
    r_ = np.zeros(n_)
    for t_ in range(1, n_):
        r_[t_] = 0.3 * r_[t_ - 1] * (t_ % PD_ != 0) + e_[t_]
    x_ = rng.normal(size=n_)
    r_[4:] += eff * x_[:-4]
    for lags_ in (10, 6):
        v_ = (rng.random(n_) > 0.03) & (np.arange(n_) % PD_ > lags_)
        _BASE.clear()
        bf, Vf, Wf = hac_granger(r_, x_, lags_, v_, day_, full_out=True)
        bs, Vs, Ws = full_granger(r_, x_, lags_, v_, day_)
        err = max(np.max(np.abs(bf - bs)) / np.max(np.abs(bs)), np.max(np.abs(Vf - Vs)) / np.max(np.abs(Vs)),
                  abs(Wf - Ws) / max(Ws, 1e-12))
        if err > 1e-6:
            sys.exit(f"STOP: fast Granger test differs from the full regression (relative error {err:.1e})")
# leakage check: flow at the END of one day drives returns at the START of the next day only.
# With day-boundary protection this must NOT be detected.
r_ = rng.normal(size=n_)
x_ = rng.normal(size=n_)
for d_ in range(1, D_):
    r_[d_ * PD_:d_ * PD_ + 5] += 50 * x_[d_ * PD_ - 5:d_ * PD_]
_BASE.clear()
p_leak = hac_granger(r_, x_, 10, np.arange(n_) % PD_ > 10, day_)[0]
_BASE.clear()
if p_leak < 0.01:
    sys.exit(f"STOP: a lag reached across a day boundary (p = {p_leak:.4f})")
print(f"SELF-CHECKS: fast test = full regression (coefficients, covariance, Wald; zero and real effect; gaps) | "
      f"cross-day leak not picked up (p = {p_leak:.2f})")

con = duckdb.connect()
sleeve = pd.read_csv(SLEEVES) if os.path.exists(SLEEVES) else pd.DataFrame(columns=["coin", "wallet", "sleeve"])
out = []
t0 = time.time()
print("NATIVE MAKERS vs HEDGERS (decisions 18 and 19)")
for coin in COINS:
    inv = [f for d in DAYS for f in glob.glob(os.path.join(INV, coin, f"{d}.parquet"))]
    if not inv:
        print(f"\n   {coin}: no inventory files")
        continue
    have_days = {os.path.basename(f)[:10] for f in inv}
    T = tns(con, inv, coin)
    pm = con.execute(f"""
        SELECT lower(wallet) AS wallet, any_value(lower(firm)) AS firm, {T} // {60 * NS} AS minute,
               arg_max(pos_after, {T}) AS pos, arg_min(pos_before, {T}) AS first_before, count(*) AS n
        FROM read_parquet({inv}, union_by_name=true) GROUP BY 1, 3""").df()
    fills = pm.groupby("wallet")["n"].sum()
    firm_of = pm.groupby("wallet")["firm"].first()
    act = sorted(fills[fills >= MIN_FILLS].index)
    sc = sleeve[sleeve["coin"] == coin]
    sl_of = dict(zip(sc["wallet"], sc["sleeve"]))
    units = {"account": {w: [w] for w in act}, "sleeve": {}, "firm": {}}
    for w in act:
        units["sleeve"].setdefault(sl_of.get(w, "none_" + w), []).append(w)
        units["firm"].setdefault(firm_of[w], []).append(w)
    grid = np.arange(pd.Timestamp(DAYS[0], tz="UTC").value // (60 * NS),
                     (pd.Timestamp(DAYS[-1], tz="UTC") + pd.Timedelta(days=1)).value // (60 * NS))
    split_min = pd.Timestamp(SPLIT, tz="UTC").value // (60 * NS)
    day_ix = (grid - grid[0]) // 1440
    missing = np.array([d not in have_days for d in DAYS])[day_ix]          # minute lies on a missing day
    seg = np.cumsum(np.r_[True, missing[1:] != missing[:-1]])                # never carry across a missing day
    P = {}
    for w, g in pm[pm["wallet"].isin(act)].groupby("wallet"):
        g = g.sort_values("minute")
        s = pd.Series(g["pos"].to_numpy(), index=g["minute"].to_numpy()).reindex(grid)
        s = s.groupby(seg).ffill()
        fb = pd.Series(g["first_before"].to_numpy(), index=g["minute"].to_numpy())
        v = s.to_numpy(dtype=float, copy=True)
        mi = np.searchsorted(grid, g["minute"].to_numpy())
        for sg in np.unique(seg[mi]):                      # each data segment starts from its first fill's "before"
            first = mi[seg[mi] == sg][0]
            start = np.searchsorted(seg, sg)
            v[start:first] = float(fb.iloc[np.flatnonzero(seg[mi] == sg)[0]])
        v[missing] = np.nan
        P[w] = v
    print(f"\n   {coin}: {len(act)} active accounts | inventory days present {len(have_days)} of {len(DAYS)}"
          f"{' (minutes on missing days dropped from half-lives)' if len(have_days) < len(DAYS) else ''}")

    # ---------- Binance returns and unit flows on the test days, bins built day by day ----------
    days = [d for d in TEST_DAYS if d >= BIN_START.get(coin, "0") and d in have_days]
    bsec, how = binance_seconds(con, coin, days) if days else (None, "no test days")
    print(f"      Binance: {how}")
    flows = None
    if bsec is not None and len(bsec):
        day0 = np.array([pd.Timestamp(d, tz="UTC").value // NS for d in days])
        sec_all = (day0[:, None] + np.arange(86400)[None, :]).ravel()          # full seconds of each test day
        sv = bsec["s"].to_numpy()
        s_idx = np.clip(np.searchsorted(sec_all, sv), 0, len(sec_all) - 1)
        okb = sec_all[s_idx] == sv
        px = np.full(len(sec_all), np.nan)
        px[s_idx[okb]] = bsec["px"].to_numpy()[okb]
        real = np.isfinite(px)
        last = np.where(real, np.arange(len(px)), -1)
        last = np.maximum.accumulate(last.reshape(len(days), 86400), axis=1).ravel()   # never across days
        age = np.where(last >= 0, np.arange(len(px)) - last, 10**9)
        pxf = np.where(last >= 0, px[np.maximum(last, 0)], np.nan)
        gaps = np.diff(np.flatnonzero(np.r_[True, real, True])) - 1
        print(f"      test days {len(days)} | seconds with a Binance price {real.mean():.1%} | forward-filled "
              f"{int((~real & (last >= 0)).sum()):,} | longest gap {int(gaps.max()):,} s | stale (> {STALE_S} s) "
              f"{(age > STALE_S).mean():.2%}")
        fs = con.execute(f"""
            SELECT lower(wallet) AS wallet, {T} // {NS} AS s, sum(pos_after - pos_before) AS flow
            FROM read_parquet({[f for f in inv if os.path.basename(f)[:10] in days]}, union_by_name=true)
            GROUP BY 1, 2""").df()
        fs = fs[fs["wallet"].isin(act)]
        fv = fs["s"].to_numpy()
        f_idx = np.clip(np.searchsorted(sec_all, fv), 0, len(sec_all) - 1)
        fs = fs[sec_all[f_idx] == fv].assign(i=f_idx[sec_all[f_idx] == fv])
        pos_in = np.arange(len(sec_all))
        flows = {}
        for name, binsec, lags in SPEEDS:
            per_day = 86400 // binsec
            key = (pos_in // 86400) * per_day + (pos_in % 86400) // binsec       # dense, same for both series
            nb = len(days) * per_day
            bp = pd.Series(pxf).groupby(key).last().reindex(range(nb)).to_numpy()
            bage = pd.Series(age).groupby(key).last().reindex(range(nb)).to_numpy()
            bmax = pd.Series(age).groupby(key).max().reindex(range(nb)).to_numpy()   # stalest second in the bin
            r = np.zeros(nb)
            with np.errstate(invalid="ignore", divide="ignore"):
                r[1:] = np.log(bp[1:] / bp[:-1]) * 1e4
            early = (np.arange(nb) % per_day) <= lags              # its lags would reach into the previous day
            valid = np.isfinite(r) & ~early & (bmax <= STALE_S) & np.r_[False, bage[:-1] <= STALE_S]
            dayb = np.arange(nb) // per_day
            r = np.where(valid, r, 0.0)
            W = {w: np.bincount(key[g["i"].to_numpy()], g["flow"].to_numpy(), minlength=nb)
                 for w, g in fs.groupby("wallet")}
            flows[name] = (r, valid, per_day, lags, W, nb, dayb)

    for level, U in units.items():
        for u, members in U.items():
            pos = np.sum([P[w] for w in members], axis=0)
            hl, phi = half_life(pos)
            hl_a, _ = half_life(pos[grid < split_min])
            hl_b, _ = half_life(pos[grid >= split_min])
            row = dict(coin=coin, level=level, unit=u, accounts=len(members), firm=firm_of[members[0]],
                       sleeve=sl_of.get(members[0]) if level != "firm" else None,
                       fills=int(sum(fills[w] for w in members)), half_life_min=hl, phi=phi,
                       minutes=len(pos), half_life_aug_sep=hl_a, half_life_sep_oct=hl_b)
            for name in ("1s", "10s"):
                p = e = np.nan
                if flows is not None:
                    r, valid, per_day, lags, W, nb, dayb = flows[name]
                    x = np.sum([W.get(w, np.zeros(nb)) for w in members], axis=0)
                    if np.std(x) > 0:
                        p, e = hac_granger(r, x / np.std(x), lags, valid, dayb)
                row[f"p_{name}"], row[f"effect_{name}"] = p, e
            out.append(row)
    # ---------- placebo on accounts: Binance from OTHER test days (random permutations, no fixed day) ----------
    if flows is not None and len(days) >= 2:
        acc_rows = {r_["unit"]: r_ for r_ in out if r_["coin"] == coin and r_["level"] == "account"}
        for name in ("1s", "10s"):
            r, valid, per_day, lags, W, nb, dayb = flows[name]
            for k in range(N_PLACEBO):
                while True:
                    perm = rng.permutation(len(days))
                    if not (perm == np.arange(len(days))).any():
                        break
                r_pl = r.reshape(len(days), per_day)[perm].ravel()
                v_pl = valid.reshape(len(days), per_day)[perm].ravel() & ((np.arange(nb) % per_day) > lags)
                _BASE.clear()
                for w, row in acc_rows.items():
                    x = W.get(w, np.zeros(nb))
                    pp = hac_granger(r_pl, x / np.std(x), lags, v_pl, dayb)[0] if np.std(x) > 0 else np.nan
                    row[f"placebo_p_{name}_{k}"] = pp
        _BASE.clear()
    log(f"   {coin} tests done | {time.time() - t0:,.0f}s")

res = pd.DataFrame(out)
if not len(res):
    sys.exit("STOP: nothing to classify")
# ---------- FDR within coin x level x speed, then labels ----------
for name in ("1s", "10s"):
    res[f"sig_{name}"] = False
    for _, idx in res.groupby(["coin", "level"]).groups.items():
        res.loc[idx, f"sig_{name}"] = bh(res.loc[idx, f"p_{name}"].to_numpy())
both_valid = res["p_1s"].notna() & res["p_10s"].notna()
hedger = res["sig_1s"] | res["sig_10s"]
fast = res["half_life_min"] < HALF_LIFE_MAX
hl_ok = res["half_life_min"].notna()                     # inf (never reverts) is a valid estimate
res["label"] = np.select([hedger, ~both_valid | ~hl_ok, fast], ["HEDGER", "INCOMPLETE", "NATIVE"], "SLOW")
res["native"] = res["label"] == "NATIVE"
res["native_1s_only"] = fast & res["p_1s"].notna() & ~res["sig_1s"]
res["native_10s_only"] = fast & res["p_10s"].notna() & ~res["sig_10s"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
res.to_csv(OUT, index=False)

for coin, c in res.groupby("coin", sort=False):
    print(f"\n   {coin}")
    for level in ("account", "sleeve", "firm"):
        d = c[c["level"] == level]
        if not len(d):
            continue
        lab = d["label"].value_counts()
        share = d.groupby("label")["fills"].sum() / d["fills"].sum()
        print(f"      {level:<8} {len(d):>4} units | " + " | ".join(
            f"{k} {lab.get(k, 0)} ({share.get(k, 0):.0%} of fills)" for k in ("NATIVE", "HEDGER", "SLOW", "INCOMPLETE")))
        if level == "account":
            stable = (d["half_life_aug_sep"] < 60) == (d["half_life_sep_oct"] < 60)
            print(f"         half-life median {d['half_life_min'].median():.0f} min | same side of 60 min in both "
                  f"subperiods: {stable.mean():.0%}")
            for name in ("1s", "10s"):
                pcols = [c_ for c_ in d.columns if c_.startswith(f"placebo_p_{name}_")]
                pv = d[pcols].to_numpy().ravel() if pcols else np.array([])
                pv = pv[np.isfinite(pv)]
                print(f"         {name}: valid tests {d[f'p_{name}'].notna().mean():.0%} | raw p < 5% "
                      f"{(d[f'p_{name}'] < ALPHA).mean():.0%}, after FDR {d[f'sig_{name}'].mean():.0%} | PLACEBO "
                      f"({N_PLACEBO} day permutations, {len(pv):,} tests) raw p < 5% "
                      f"{(pv < ALPHA).mean() if len(pv) else np.nan:.1%} (want about 5%)")
            print(f"         native with only the 1 s test {int(d['native_1s_only'].sum())}, only the 10 s test "
                  f"{int(d['native_10s_only'].sum())}")
            for r in d.sort_values("fills", ascending=False).head(6).itertuples():
                print(f"         {r.unit[:8]}… {r.fills:>10,} fills | HL {r.half_life_min:>7.1f} min "
                      f"({r.half_life_aug_sep:.0f} / {r.half_life_sep_oct:.0f}) | p 1s {r.p_1s:.3f} 10s {r.p_10s:.3f} "
                      f"| {r.label}")
print(f"\nSaved {OUT}  ({time.time() - t0:,.0f} seconds). Names private: 8 characters shown.")
print("A HEDGER label is evidence consistent with hedging on Binance, not proof; NATIVE means no footprint found.")
