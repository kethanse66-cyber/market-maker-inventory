"""Day 18: build quotes, inventory, book delay and episodes for the replication coins (delivery 3), unattended."""
import os
import subprocess
import sys
import time

import pandas as pd

P3 = r"E:\p3"
RES = os.path.join(P3, "results")
SKIP = {"CHIP", "XRP", "PENGU", "STRK", "FARTCOIN"}
rq = pd.read_csv(os.path.join(P3, "data", "request_coin_days.csv"))
low = {c.lower(): c for c in rq.columns}
lv = rq.groupby(rq[low["coin"]].str.upper())[low["boundary"]].agg(lambda s: sorted(set(s.astype(float))))
coins = [c for c in lv.index if c not in SKIP]
print("Replication coins to build:", ", ".join(f"{c} (boundary {', '.join(f'{x:g}' for x in lv[c])})" for c in coins))
print("Skipped:", ", ".join(sorted(SKIP & set(lv.index) | (SKIP - set(lv.index)))))
if "--plan" in sys.argv:
    sys.exit(0)
os.makedirs(RES, exist_ok=True)
t00 = time.time()
for c in coins:
    mark = os.path.join(RES, f"day18_DONE_{c}.txt")
    if os.path.exists(mark):
        print(f"{c}: already done - skipped")
        continue
    steps = [("a", ["day16a_inventory_d3.py", c]), ("b", ["day16b_lag_d3.py", c]),
             ("c", ["day16c_quotes_d3.py", c]),
             ("d", ["day16d_episodes_d3.py", f"{c}={','.join(f'{x:g}' for x in lv[c])}"])]
    ok = True
    for tag, (script, *args) in steps:
        t0 = time.time()
        with open(os.path.join(RES, f"day18_{tag}_{c}.txt"), "w", encoding="utf-8") as out:
            r = subprocess.run([sys.executable, os.path.join(P3, script), *args], stdout=out, stderr=subprocess.STDOUT)
        txt = open(os.path.join(RES, f"day18_{tag}_{c}.txt"), encoding="utf-8", errors="ignore").read()
        stop = [l for l in txt.splitlines() if l.startswith("STOP") or "Error" in l]
        print(f"{c} step {tag}: {'OK' if r.returncode == 0 else 'FAILED'} in {time.time() - t0:,.0f}s"
              + (f" | {stop[-1][:150]}" if stop else ""), flush=True)
        if r.returncode != 0:
            ok = False
            break
    if ok:
        open(mark, "w").write(time.strftime("%Y-%m-%d %H:%M"))
print(f"\nFinished in {(time.time() - t00) / 3600:.1f} h. Check: type E:\\p3\\results\\day18_d_<COIN>.txt for episodes.")
