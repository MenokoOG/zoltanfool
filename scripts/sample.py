"""Module 2 and 4: print samples from a checkpoint at several temperatures.

python -m scripts.sample checkpoints/bigram/best.pt --temps 0.5 0.8 1.0
"""
import argparse

import torch

from zoltanfool.registry import load_model
from zoltanfool.tokenizer import decode, encode


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("ckpt")
    a.add_argument("--temps", type=float, nargs="+", default=[0.5, 0.8, 1.0])
    a.add_argument("--length", type=int, default=300)
    a.add_argument("--prompt", default="def ")
    a.add_argument("--seed", type=int, default=0)
    args = a.parse_args()
    model, spec = load_model(args.ckpt)
    for t in args.temps:
        torch.manual_seed(args.seed)
        out = model.generate(torch.tensor([encode(args.prompt)]), args.length, t)
        print(f"\n===== {spec['name']}  T = {t} =====\n{decode(out[0].tolist())}")


if __name__ == "__main__":
    main()
