"""Module 5 calculator: right-size the model to your measured speed and your hours.

python -m scripts.size --tps 3000 --params 24.99 --hours 10 --corpus-mb 14.6
"""
import argparse

from zoltanfool.config import CONFIGS, expected_params, right_size


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--tps", type=float, required=True, help="measured tokens/s")
    a.add_argument("--params", type=float, required=True, help="millions of params of the timed model")
    a.add_argument("--hours", type=float, required=True)
    a.add_argument("--corpus-mb", type=float, required=True)
    args = a.parse_args()
    r = right_size(args.tps, args.params * 1e6, args.hours, args.corpus_mb * 1e6)
    print(f"effective speed   {r['flops_per_s']:.2e} FLOP/s")
    print(f"budget            {r['compute']:.2e} FLOP")
    print(f"right-sized model {r['params'] / 1e6:.1f}M params")
    print(f"tokens to train   {r['tokens'] / 1e6:.0f}M  ({r['passes']:.1f} passes over the corpus)")
    near = min(CONFIGS, key=lambda k: abs(expected_params(CONFIGS[k]) - r["params"]))
    print(f"nearest config    {near} ({expected_params(CONFIGS[near]) / 1e6:.2f}M)")
    if r["passes"] > 4:
        print("warning: more than ~4 passes; expect memorizing. Add data or pick a smaller model.")


if __name__ == "__main__":
    main()
