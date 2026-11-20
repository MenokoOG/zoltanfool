"""Module 7, zero-pass fix: fine-tune on docstring-then-body pairs from the train side of your corpus.

python -m scripts.sft checkpoints/debug/best.pt /path/to/repo1 /path/to/repo2 --steps 300 --run-dir checkpoints/sft
Only files that build_corpus put in train are used, and any function named like a ladder task is dropped.
"""
import argparse
import hashlib
import shutil
from collections.abc import Iterator
from pathlib import Path

from data.build_corpus import normalize
from tasks.ladder import ALL
from zoltanfool.registry import load_model, save_spec
from zoltanfool.rft import finetune
from zoltanfool.sft_data import function_examples

TASK_NAMES = {t.name for t in ALL}


def train_side_sources(roots: list[str], val_every: int = 10) -> Iterator[str]:
    """Same rule as build_corpus: hash of the normalized file picks the split. Yield train files only."""
    seen = set()
    for root in roots:
        for p in sorted(Path(root).rglob("*.py")):
            try:
                b = p.read_bytes()
            except OSError:
                continue
            h = hashlib.sha256(normalize(b)).hexdigest()
            if h in seen or len(b) < 200 or int(h, 16) % val_every == 0:
                continue
            seen.add(h)
            yield b.decode("utf-8", "replace")


def leaks(prompt: str) -> bool:
    name = prompt.split("(", 1)[0].replace("def ", "").strip()
    return name in TASK_NAMES


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("ckpt")
    a.add_argument("roots", nargs="+")
    a.add_argument("--steps", type=int, default=300)
    a.add_argument("--batch", type=int, default=8)
    a.add_argument("--peak", type=float, default=1e-4)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--run-dir", required=True)
    args = a.parse_args()
    pairs = [pb for src in train_side_sources(args.roots) for pb in function_examples(src)]
    kept = [p + b for p, b in pairs if not leaks(p)]
    print(f"{len(pairs)} docstring functions, {len(pairs) - len(kept)} dropped as task-name leaks")
    if not kept:
        raise SystemExit("no examples found; point it at repos with documented top-level functions")
    model, spec = load_model(args.ckpt)
    model.train()
    save_spec(args.run_dir, {**spec, "parent": args.ckpt, "stage": "sft"})
    finetune(model, kept, args.steps, args.batch, args.run_dir, peak=args.peak, seed=args.seed)
    shutil.copy(Path(args.run_dir) / "last.pt", Path(args.run_dir) / "model.pt")
    print(f"saved {args.run_dir}/model.pt")


if __name__ == "__main__":
    main()
