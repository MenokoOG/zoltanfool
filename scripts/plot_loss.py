"""Module 2 and 5: plot train and val loss from one or more log.csv files, with the floors drawn in.

python -m scripts.plot_loss checkpoints/bigram checkpoints/debug --unigram 3.389 --out results/loss.png
"""
import argparse
import csv
import math
from pathlib import Path


def read_log(run_dir: str) -> tuple[list[int], list[float], list[float]]:
    with open(Path(run_dir) / "log.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return ([int(r["step"]) for r in rows], [float(r["train_loss"]) for r in rows],
            [float(r["val_loss"]) for r in rows])


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = argparse.ArgumentParser()
    a.add_argument("run_dirs", nargs="+")
    a.add_argument("--unigram", type=float, help="your unigram floor in nats (python -m scripts.baselines)")
    a.add_argument("--out", default="results/loss.png")
    args = a.parse_args()
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in args.run_dirs:
        s, tr, va = read_log(d)
        name = Path(d).name
        line, = ax.plot(s, tr, label=f"{name} train", alpha=0.6)
        ax.plot(s, va, label=f"{name} val", color=line.get_color(), linestyle="--")
    ax.axhline(math.log(256), color="grey", linewidth=1, label="uniform ln 256")
    if args.unigram:
        ax.axhline(args.unigram, color="black", linewidth=1, linestyle=":", label="unigram floor")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (nats per byte)")
    ax.legend()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
