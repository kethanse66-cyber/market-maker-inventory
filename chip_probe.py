"""Is there any other source of CHIP order-book prices for Apr 21-25? (a few API credits)
Run:  python E:\\p3\\chip_probe.py
"""
import os

from oxarchive import Client

C = Client(api_key=os.environ["OXARCHIVE_API_KEY"], timeout=120)
H = C.hyperliquid
s, e = "2026-04-23T12:00:00Z", "2026-04-23T12:10:00Z"

try:
    cov = C.data_quality.symbol_coverage("hyperliquid", "CHIP", from_time="2026-04-20T00:00:00Z",
                                         to_time="2026-04-26T00:00:00Z")
    d = cov.model_dump() if hasattr(cov, "model_dump") else cov
    print("1. coverage:", str(d)[:1500])
except Exception as ex:
    print("1. coverage: ERROR", ex)

for name, fn, kw in (("2. top of book", H.orderbook.history, {"depth": 1}),
                     ("3. L2 book", H.l2_orderbook.history, {"depth": 1})):
    try:
        r = fn("CHIP", start=s, end=e, limit=50, **kw)
        print(f"{name}: {len(r.data)} snapshots in 10 min; first: {str(r.data[0])[:200] if r.data else '-'}")
    except Exception as ex:
        print(f"{name}: ERROR {ex}")

try:
    r = H.trades.list("CHIP", start=s, end=e, limit=50)
    print(f"4. trades: {len(r.data)} in 10 min")
except Exception as ex:
    print("4. trades: ERROR", ex)
