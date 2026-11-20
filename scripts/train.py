"""Modules 2, 4, 5, 9: train any model by name. Logs, checkpoints and resumes through train().

python -m scripts.train --model bigram --steps 300 --batch 32 --run-dir checkpoints/bigram
python -m scripts.train --model debug  --steps 2000 --batch 8 --peak 1e-3 --run-dir checkpoints/debug
Run the same command again after Ctrl+C and it resumes from last.pt.
"""
import argparse

import torch

from scripts._data import load_split
from zoltanfool.model import ZoltanFool
from zoltanfool.registry import NAMES, make_model, save_spec
from zoltanfool.train import train


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--model", choices=NAMES, required=True)
    a.add_argument("--steps", type=int, required=True)
    a.add_argument("--batch", type=int, default=8)
    a.add_argument("--accum", type=int, default=1)
    a.add_argument("--peak", type=float, help="peak learning rate (default 1e-1 bigram, 3e-4 otherwise)")
    a.add_argument("--context", type=int, help="override context_length (Module 8 experiment)")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--run-dir", required=True)
    a.add_argument("--eval-every", type=int, default=250)
    a.add_argument("--ckpt-every", type=int, default=500)
    a.add_argument("--threads", type=int, help="torch.set_num_threads; try your physical core count")
    a.add_argument("--fast", action="store_true", help="fused attention kernel (Module 9)")
    a.add_argument("--fresh", action="store_true", help="ignore last.pt and start over")
    args = a.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model, spec = make_model(args.model, args.context)
    if args.fast and isinstance(model, ZoltanFool):
        model.set_fast(True)
    bigram = args.model == "bigram"
    peak = args.peak if args.peak is not None else (1e-1 if bigram else 3e-4)
    save_spec(args.run_dir, spec)
    n = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {n:,} parameters, peak lr {peak}, threads {torch.get_num_threads()}")
    train(model, load_split("train"), load_split("val"), steps=args.steps, batch=args.batch,
          context=model.c.context_length, run_dir=args.run_dir, seed=args.seed, peak=peak,
          accum=args.accum, eval_every=args.eval_every, ckpt_every=args.ckpt_every,
          resume=not args.fresh, weight_decay=0.0 if bigram else 0.1)


if __name__ == "__main__":
    main()
