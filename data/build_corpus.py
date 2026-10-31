"""Collect .py files, drop exact duplicates, split train/val by file (never inside a file)."""
import hashlib
import sys
from pathlib import Path


def normalize(b: bytes) -> bytes:
    """Strip line whitespace, blank lines and comment-only lines, so copies that differ only there match."""
    lines = (l.strip() for l in b.decode("utf-8", "replace").splitlines())
    return "\n".join(l for l in lines if l and not l.startswith("#")).encode()


def build(roots: list[str], out_dir: str = "data/raw", val_every: int = 10):
    seen, train, val = set(), [], []
    for root in roots:
        for p in sorted(Path(root).rglob("*.py")):
            try:
                b = p.read_bytes()
            except OSError:
                continue
            h = hashlib.sha256(normalize(b)).hexdigest()
            if h in seen or len(b) < 200:
                continue
            seen.add(h)
            # the hash decides the side, so a file always lands in the same split
            (val if int(h, 16) % val_every == 0 else train).append(b)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.bin").write_bytes(b"\n".join(train))
    (out / "val.bin").write_bytes(b"\n".join(val))
    return len(train), len(val)


if __name__ == "__main__":
    print(build(sys.argv[1:]))