"""Write SHA-256 hashes of the pre-registration files to docs/FREEZE_HASHES.txt."""
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"E:\p3")
MANIFEST = REPO / "docs" / "FREEZE_HASHES.txt"
CODE_EXT = {".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".cfg", ".ini", ".json"}
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "venv", ".venv"}
VENDOR_DIRS = [Path(r"E:\hl_export"), Path(r"E:\hl_export\delivery-2"), Path(r"G:\hl_export")]  # their .csv/.md manifests


def git(*a):
    r = subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True)
    if r.returncode not in (0, 1):
        sys.exit(f"STOP: git {' '.join(a)} failed: {r.stderr.strip()}")
    return r.stdout


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def discover():
    code, data = [], []
    for p in sorted(REPO.rglob("*")):
        if not p.is_file() or SKIP_DIRS & set(p.relative_to(REPO).parts) or p == MANIFEST:
            continue
        rel = p.relative_to(REPO)
        if rel.parts[0] == "data":
            data.append(p)
        elif (p.suffix.lower() in CODE_EXT and not p.name.endswith("_output.txt")) or p.name == ".gitignore":
            code.append(p)
    ext = sorted(p for d in VENDOR_DIRS if d.exists() for p in d.iterdir()
                 if p.is_file() and p.suffix.lower() in (".csv", ".md"))
    return code, data, ext


def name(p):
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def leaked(data):
    """Data files that git tracks or does not ignore (privacy: full wallet addresses)."""
    tracked = set(git("ls-files").splitlines())
    return sorted({name(p) for p in data if name(p) in tracked
                   or subprocess.run(["git", "check-ignore", "-q", name(p)], cwd=REPO).returncode != 0})


def build():
    if not (REPO / ".git").exists():
        sys.exit("STOP: E:\\p3 is not a git repository")
    if git("tag", "-l", "prereg-v1").strip():
        sys.exit("STOP: tag prereg-v1 already exists - the freeze is done; run  python freeze.py --verify")
    code, data, ext = discover()
    bad = leaked(data)
    if bad:
        sys.exit("STOP: data files are tracked or not ignored (add data/ to .gitignore, "
                 "git rm --cached them):\n  " + "\n  ".join(bad[:20]))
    print(f"Hashing {len(code)} code/doc files, {len(data)} data files, {len(ext)} vendor manifests "
          f"({sum(p.stat().st_size for p in data) / 1e9:.1f} GB; this can take several minutes)...", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Pre-registration freeze: {stamp}", "# Tag: prereg-v1",
             "# sha256  bytes  path   (data/ and vendor manifests are hashed, never committed;",
             "#                        this manifest is not in its own list)", ""]
    for group, ps in (("code and docs", code), ("data", data), ("vendor manifests", ext)):
        lines.append(f"## {group}")
        lines += [f"{sha(p)}  {p.stat().st_size:>12}  {name(p)}" for p in ps]
        lines.append("")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines[:60]) + ("\n..." if len(lines) > 60 else ""))
    print(f"\nWrote {name(MANIFEST)}: {len(code)} code/doc, {len(data)} data, {len(ext)} vendor manifests")
    print("Code/doc files NOT yet committed (will be committed in the next step):")
    print("  " + ("\n  ".join(git("status", "--porcelain").splitlines()) or "none"))


def verify():
    if not MANIFEST.exists():
        sys.exit("STOP: docs/FREEZE_HASHES.txt not found - run  python freeze.py  first")
    rows = [l.split(None, 2) for l in MANIFEST.read_text(encoding="utf-8").splitlines()
            if l and not l.startswith("#")]
    bad = 0
    for h, size, path in rows:
        p = Path(path) if ":" in path else REPO / path
        if not p.exists():
            print(f"MISSING  {path}")
            bad += 1
        elif p.stat().st_size != int(size):
            print(f"SIZE CHANGED  {path}")
            bad += 1
        elif sha(p) != h:
            print(f"CHANGED  {path}")
            bad += 1
    code, data, ext = discover()
    listed = {r[2] for r in rows}
    new = [name(p) for p in code + data + ext if name(p) not in listed]
    for n in new:
        print(f"NOT IN MANIFEST  {n}")
    priv = leaked(data)
    for n in priv:
        print(f"DATA NOT IGNORED BY GIT  {n}")
    raw_bad = verify_raw(ext) if "--raw" in sys.argv else None
    dirty = git("status", "--porcelain").strip()
    tag = git("tag", "-l", "prereg-v1").strip()
    head_is_tag = git("rev-list", "-n", "1", "prereg-v1").strip() == git("rev-parse", "HEAD").strip() if tag else False
    print(f"\n{len(rows)} files checked | changed/missing {bad} | new files {len(new)} | "
          f"git clean: {'yes' if not dirty else 'NO'} | tag prereg-v1: {'yes' if tag else 'NO'}"
          f"{' (= HEAD)' if head_is_tag else ''} | data leaks {len(priv)} | raw delivery files: "
          f"{'not checked (add --raw)' if raw_bad is None else ('all match' if raw_bad == 0 else f'{raw_bad} BAD')}")
    if dirty:
        print(dirty)
    ok = not bad and not new and not dirty and tag and head_is_tag and not priv and not raw_bad
    print("FREEZE VERIFIED" if ok else "FREEZE NOT VERIFIED - see above")


def verify_raw(manifests):
    """Re-hash every raw delivery file listed in the vendor manifests (path + sha256 columns)."""
    import csv
    bad, n = 0, 0
    for m in manifests:
        if m.suffix.lower() != ".csv":
            continue
        rows = list(csv.DictReader(open(m, newline="", encoding="utf-8-sig")))
        if not rows:
            continue
        low = {c.lower().strip(): c for c in rows[0]}
        pc = next((low[k] for k in ("path", "file", "filename", "key") if k in low), None)
        hc = next((low[k] for k in ("sha256", "sha_256", "hash", "checksum") if k in low), None)
        if not pc or not hc:
            print(f"RAW: {m} has no path/sha256 columns ({list(low)}) - skipped, not a file manifest")
            continue
        bases = [m.parent] + sorted(d for d in m.parent.iterdir() if d.is_dir())
        print(f"RAW: checking {len(rows)} files from {m} ...", flush=True)
        for r in rows:
            rel = r[pc].replace("\\", "/").lstrip("/")
            p = next((b / rel for b in bases if (b / rel).exists()), None)
            n += 1
            if p is None:
                print(f"RAW MISSING  {rel}  ({m.name})")
                bad += 1
            elif sha(p) != r[hc].strip().lower().removeprefix("sha256:"):
                print(f"RAW CHANGED  {p}")
                bad += 1
    print(f"RAW: {n} raw files checked, {bad} bad")
    return bad


if __name__ == "__main__":
    verify() if "--verify" in sys.argv else build()
