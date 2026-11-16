"""Quick probe of the API formats before downloading the delivery 3 extras."""
import glob
import os

import pandas as pd
from oxarchive import Client

print("1. EXISTING FILES")
bf = sorted(glob.glob(r"E:\hl_data\book\SOL\*.parquet"))
if bf:
    b = pd.read_parquet(bf[0])
    print(f"   book file {os.path.basename(bf[0])}: {len(b):,} rows | columns {list(b.columns)}")
    print(b.head(3).to_string())
bn = sorted(glob.glob(r"E:\hl_data\binance\**\*.parquet", recursive=True))
if bn:
    x = pd.read_parquet(bn[0])
    rel = os.path.relpath(bn[0], r"E:\hl_data\binance")
    print(f"\n   binance file {rel}: {len(x):,} rows | columns {list(x.columns)}")
    print(x.head(2).to_string())
print("\n   other folders in E:\\hl_data:", sorted(os.listdir(r"E:\hl_data")))
print("   scripts in E:\\sentinel mentioning 'orderbook':",
      [os.path.basename(f) for f in glob.glob(r"E:\sentinel\*.py") if "orderbook" in open(f, encoding="utf-8", errors="ignore").read()])

print("\n2. API TOP OF BOOK FOR NEW COINS")
H = Client(api_key=os.environ["OXARCHIVE_API_KEY"], timeout=120).hyperliquid
for coin, s, e in (("XPL", "2026-03-12T12:00:00Z", "2026-03-12T12:00:10Z"),
                   ("SUI", "2026-10-15T12:00:00Z", "2026-10-15T12:00:10Z")):
    try:
        r = H.orderbook.history(coin, start=s, end=e, limit=20, depth=1)
        print(f"   {coin} {s[:10]}: {len(r.data)} snapshots in 10 s | has_more {r.next_cursor is not None}")
        if r.data:
            d = r.data[0].model_dump() if hasattr(r.data[0], "model_dump") else dict(r.data[0])
            print(f"      fields: {list(d.keys())}")
            print(f"      first: {str(d)[:400]}")
    except Exception as ex:
        print(f"   {coin} {s[:10]}: ERROR {ex}")
