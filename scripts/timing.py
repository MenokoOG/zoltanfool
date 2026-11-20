"""Module 5: tokens per second and peak memory on this machine, per model size.

python -m scripts.timing --models debug target --steps 50 --batch 8
"""
import argparse
import resource
import time

import torch

from zoltanfool.model import ZoltanFool
from zoltanfool.registry import make_model
from zoltanfool.train import get_batch, make_optimizer


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--models", nargs="+", default=["debug", "target"])
    a.add_argument("--steps", type=int, default=50)
    a.add_argument("--batch", type=int, default=8)
    a.add_argument("--threads", type=int)
    a.add_argument("--fast", action="store_true")
    args = a.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    data = torch.randint(0, 256, (200_000,))
    print("threads:", torch.get_num_threads())
    for name in args.models:
        model, _ = make_model(name)
        if args.fast and isinstance(model, ZoltanFool):
            model.set_fast(True)
        opt = make_optimizer(model, 3e-4)
        gen = torch.Generator().manual_seed(0)
        T = model.c.context_length
        for i in range(args.steps + 2):
            if i == 2:
                t0 = time.time()                          # skip two warm-up steps
            x, y = get_batch(data, args.batch, T, gen)
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        tps = args.batch * T * args.steps / (time.time() - t0)
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        n = sum(p.numel() for p in model.parameters())
        print(f"{name:14s} {n / 1e6:6.2f}M params  {tps:8,.0f} tok/s  loss {loss.item():.3f}  "
              f"peak RSS so far {peak_mb:,.0f} MB")
        for d in (50e6, 200e6, 1e9):
            print(f"{'':16s}{d / 1e6:6,.0f}M tokens -> {d / tps / 3600:6.1f} h")


if __name__ == "__main__":
    main()
