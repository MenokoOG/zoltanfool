"""Module 7: the reward loop. Measure held-out, sample train tasks, keep passers, fine-tune, measure again.

python -m scripts.rft_loop checkpoints/sft/model.pt --rounds 2 --n 20 --seed 0 --run-dir checkpoints/rft-s0
Writes runs/rft-seed{seed}-{time}.json with pass@1 and pass@5 per band, before and after.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from tasks.ladder import HELDOUT, TRAIN
from zoltanfool.evalh import git_commit
from zoltanfool.model import ZoltanFool
from zoltanfool.registry import load_model, save_spec
from zoltanfool.rft import by_band, finetune, measure


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("ckpt")
    a.add_argument("--rounds", type=int, default=1)
    a.add_argument("--n", type=int, default=20, help="samples per task; keep it the same before and after")
    a.add_argument("--length", type=int, default=160)
    a.add_argument("--temperature", type=float, default=0.8)
    a.add_argument("--steps", type=int, default=100, help="fine-tune steps per round")
    a.add_argument("--batch", type=int, default=8)
    a.add_argument("--peak", type=float, default=1e-4)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--run-dir", required=True)
    a.add_argument("--out-dir", default="runs")
    args = a.parse_args()

    torch.manual_seed(args.seed)
    model, spec = load_model(args.ckpt)
    if isinstance(model, ZoltanFool):
        model.set_fast(True)
    kw = dict(n=args.n, length=args.length, temperature=args.temperature)
    before = by_band(model, HELDOUT, **kw)
    print("held-out before:", before)
    rounds = []
    for r in range(args.rounds):
        got = measure(model, TRAIN, **kw)
        print(f"round {r}: train pass@ {got['pass_at']}, kept {len(got['kept'])}")
        rounds.append({"train_pass_at": got["pass_at"], "kept": len(got["kept"])})
        if not got["kept"]:
            print("nothing passed; the loop has nothing to learn from. Run scripts.sft first.")
            break
        rd = f"{args.run_dir}/round{r}"
        save_spec(rd, spec)
        model.train()
        finetune(model, got["kept"], args.steps, args.batch, rd, peak=args.peak, seed=args.seed + r)
        model.eval()
    after = by_band(model, HELDOUT, **kw)
    print("held-out after: ", after)
    res = {"label": f"rft-r{args.rounds}", "seed": args.seed, "commit": git_commit(), "parent": args.ckpt,
           "n": args.n, "temperature": args.temperature, "heldout_before": before,
           "heldout_after": after, "rounds": rounds,
           "heldout_tasks": [t.name for t in HELDOUT], "train_tasks": [t.name for t in TRAIN]}
    Path(args.out_dir).mkdir(exist_ok=True)
    p = Path(args.out_dir) / f"rft-seed{args.seed}-{int(time.time())}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
