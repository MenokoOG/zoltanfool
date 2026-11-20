"""Module 6: one command, one JSON per run in runs/.

python -m scripts.evaluate checkpoints/debug/best.pt --label debug --seed 0
python -m scripts.evaluate --untrained debug --label untrained --seed 0
"""
import argparse

import torch

from scripts._data import load_split
from zoltanfool.evalh import run_eval
from zoltanfool.model import ZoltanFool
from zoltanfool.registry import load_model, make_model


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("ckpt", nargs="?")
    a.add_argument("--untrained", help="model name to score with random weights instead of a checkpoint")
    a.add_argument("--label", required=True)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--n", type=int, default=100, help="samples per temperature")
    a.add_argument("--length", type=int, default=300)
    a.add_argument("--out-dir", default="runs")
    args = a.parse_args()
    if bool(args.ckpt) == bool(args.untrained):
        raise SystemExit("give a checkpoint or --untrained NAME, not both")
    torch.manual_seed(args.seed)
    model, spec = (load_model(args.ckpt) if args.ckpt else make_model(args.untrained))
    if not args.ckpt:
        model.eval()
    if isinstance(model, ZoltanFool):
        model.set_fast(True)                         # same numbers as the explicit path (tested)
    res = run_eval(model, load_split("val"), model.c, seed=args.seed, label=args.label,
                   out_dir=args.out_dir, n=args.n, length=args.length,
                   extra={"model": spec["name"], "checkpoint": args.ckpt})
    print(f"bits/byte {res['bits_per_byte']:.4f}")
    for t, s in res["syntax"].items():
        print(f"T={t}: syntax {s:.0%}  mean valid lines {res['mean_valid_lines'][t]:.1f}")


if __name__ == "__main__":
    main()
