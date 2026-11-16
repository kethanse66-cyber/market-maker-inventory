"""List every data file used in the project.
Writes docs/DATA_INVENTORY.md and .csv."""
import collections
import os
import re
import string
import sys
from datetime import datetime

ROOTS = [r"E:\hl_export", r"G:\hl_export", r"E:\hl_data", r"H:\hl_data", r"E:\p3\data", r"E:\sentinel\data"]
OUT = r"E:\p3\docs"
DATE = re.compile(r"(20\d\d)-?(\d\d)-?(\d\d)")
WALLET = re.compile(r"^0x[0-9a-fA-F]{40}$")
SKIP_EXT = {".part", ".pyc"}


def norm(rel):
    """Folder path with wallet folders replaced by <wallet>."""
    parts = rel.split(os.sep)
    wallets = [p for p in parts if WALLET.match(p)]
    return os.sep.join("<wallet>" if WALLET.match(p) else p for p in parts), (wallets[0].lower() if wallets else None)


def scan(root):
    g = collections.defaultdict(lambda: {"files": 0, "bytes": 0, "dates": set(), "wallets": set(), "ext": collections.Counter()})
    for d, _, files in os.walk(root):
        rel = os.path.relpath(d, root)
        key, w = norm(rel)
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SKIP_EXT:
                continue
            e = g[key]
            e["files"] += 1
            try:
                e["bytes"] += os.path.getsize(os.path.join(d, f))
            except OSError:
                pass
            e["ext"][ext or "(none)"] += 1
            m = DATE.search(f)
            if m:
                try:
                    e["dates"].add(datetime(int(m[1]), int(m[2]), int(m[3])).date())
                except ValueError:
                    pass
            if w:
                e["wallets"].add(w)
    return g


def gb(b):
    return f"{b / 1e9:.2f} GB" if b >= 1e8 else f"{b / 1e6:.1f} MB"


def main():
    roots = sys.argv[1:] or ROOTS
    out = OUT
    rows, md = [], [f"# Project 3 data inventory\n\nMade {datetime.now():%Y-%m-%d %H:%M} by data_inventory.py. "
                    "Dates come from file names; 'days' = distinct dates.\n"]
    other = []
    for drv in string.ascii_uppercase[2:]:  # top-level folders on every drive that look like project data
        top = f"{drv}:\\"
        if os.path.isdir(top):
            try:
                hits = [n for n in os.listdir(top) if re.search(r"hl_|p3|binance|sentinel|kite|nifty|ofi|export", n, re.I)]
            except OSError:
                hits = []
            other += [os.path.join(top, n) for n in hits]
    md.append("Project-like folders on all drives: " + (", ".join(sorted(other)) or "none") + "\n")
    total = 0
    for root in roots:
        if not os.path.isdir(root):
            md.append(f"\n## {root}\n\nnot found\n")
            continue
        g = scan(root)
        rb = sum(e["bytes"] for e in g.values())
        total += rb
        md.append(f"\n## {root}  ({gb(rb)}, {sum(e['files'] for e in g.values()):,} files)\n")
        md.append("| folder | files | size | types | first day | last day | days | wallets |")
        md.append("|---|---|---|---|---|---|---|---|")
        for k in sorted(g):
            e = g[k]
            ds = sorted(e["dates"])
            ty = " ".join(f"{x}:{n}" for x, n in e["ext"].most_common(3))
            r = {"root": root, "folder": k, "files": e["files"], "bytes": e["bytes"], "types": ty,
                 "first_day": ds[0] if ds else "", "last_day": ds[-1] if ds else "", "days": len(ds),
                 "wallets": len(e["wallets"]) or ""}
            rows.append(r)
            md.append(f"| {k} | {e['files']:,} | {gb(e['bytes'])} | {ty} | {r['first_day']} | {r['last_day']} | "
                      f"{r['days'] or ''} | {r['wallets']} |")
    md.append(f"\nTOTAL {gb(total)}\n")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "DATA_INVENTORY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    import csv
    with open(os.path.join(out, "DATA_INVENTORY.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["root"])
        w.writeheader()
        w.writerows(rows)
    print("\n".join(md))
    print(f"\nSaved {os.path.join(out, 'DATA_INVENTORY.md')} and .csv ({len(rows)} folders)")


if __name__ == "__main__":
    main()
