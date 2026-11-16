"""CHIP top of book from the L2 checkpoints (the 0.5 s feed has no CHIP data)."""
import os
import sys
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, r"E:\p3")
from day12c_extras import BOOK_COLS, dump, page_all, save, utc  # noqa: E402
from oxarchive import Client  # noqa: E402

H = Client(api_key=os.environ["OXARCHIVE_API_KEY"], timeout=180).hyperliquid
for k in range(22, 26):
    d = date(2026, 4, k)
    s, e = utc(d), utc(d + timedelta(days=1))
    recs, pages = page_all(H.l2_orderbook.history, "CHIP", s, e, depth=1)
    rows = []
    for x in map(dump, recs):
        b = (x.get("bids") or [None])[0]
        a = (x.get("asks") or [None])[0]
        rows.append((x.get("timestamp") or x.get("time"),
                     b and b["px"], b and b["sz"], b and b["n"], a and a["px"], a and a["sz"], a and a["n"]))
    df = pd.DataFrame(rows, columns=BOOK_COLS[:7])
    if len(df):
        df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
        for c in BOOK_COLS[1:7]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["mid"] = (df.bid_px + df.ask_px) / 2
        df["spread_bps"] = (df.ask_px - df.bid_px) / df.mid * 1e4
        df = df[(df.time >= s) & (df.time < e)].drop_duplicates("time").sort_values("time")
    save(df.reset_index(drop=True), os.path.join(r"E:\hl_data\book_l2cp\CHIP", f"CHIP_{d}.parquet"))
    gap = df.time.diff().dt.total_seconds().median() if len(df) > 1 else float("nan")
    print(f"CHIP {d}: {len(df):,} snapshots, {pages} pages, median gap {gap:.0f} s, "
          f"first {df.time.min() if len(df) else '-'}")
