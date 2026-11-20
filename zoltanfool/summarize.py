"""Stage 7: group run JSONs by label, report mean and range across seeds.

python -m zoltanfool.summarize runs
python -m zoltanfool.summarize runs syntax.0.8          # a nested key, dotted
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def pick(record: dict, key: str) -> object:
    """Follow a dotted key like 'syntax.0.8'. Returns None if any part is missing."""
    node = record
    parts = key.split(".")
    i = 0
    while i < len(parts):
        # join greedily so '0.8' still works as one key
        for j in range(len(parts), i, -1):
            k = ".".join(parts[i:j])
            if isinstance(node, dict) and k in node:
                node, i = node[k], j
                break
        else:
            return None
    return node


def summarize(run_dir: str, key: str = "bits_per_byte") -> list[tuple[str, int, float, float, float]]:
    groups = defaultdict(list)
    for p in Path(run_dir).glob("*.json"):
        r = json.loads(p.read_text())
        v = pick(r, key)
        if isinstance(v, (int, float)):                 # files without this key are skipped
            groups[r.get("label", "unlabeled")].append(v)
    rows = []
    for label, xs in sorted(groups.items()):
        rows.append((label, len(xs), sum(xs) / len(xs), min(xs), max(xs)))
    return rows


if __name__ == "__main__":
    key = sys.argv[2] if len(sys.argv) > 2 else "bits_per_byte"
    for label, n, mean, lo, hi in summarize(sys.argv[1], key):
        print(f"{label:24s} n={n}  mean {mean:.4f}  range {lo:.4f} to {hi:.4f}  spread {hi - lo:.4f}")
