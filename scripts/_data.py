"""Load the built corpus as byte tensors. Shared by the scripts."""
from pathlib import Path

import torch

RAW = Path("data/raw")


def load_split(name: str, raw: Path = RAW) -> torch.Tensor:
    p = raw / f"{name}.bin"
    if not p.exists():
        raise SystemExit(f"{p} not found. Build it first: python3 data/build_corpus.py /path/to/repo ...")
    return torch.frombuffer(bytearray(p.read_bytes()), dtype=torch.uint8).long()
