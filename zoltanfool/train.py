"""Stages 2 and 4: batches, the training step, schedule, logging, checkpoints, resume."""
import csv
import math
import time
from pathlib import Path

import torch

from .checkpoint import load, save


def get_batch(data: torch.Tensor, batch: int, context: int, gen: torch.Generator):
    ix = torch.randint(len(data) - context - 1, (batch,), generator=gen)
    x = torch.stack([data[i:i + context] for i in ix])
    y = torch.stack([data[i + 1:i + context + 1] for i in ix])   # shifted one byte right
    return x, y


def lr_at(step: int, total: int, peak: float = 3e-4, warmup: int = 100, floor: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))


def fixed_windows(data: torch.Tensor, context: int, count: int):
    """The same non-overlapping validation windows every time, spread across the whole set."""
    n = (len(data) - 1) // context
    starts = [int(i * n / count) * context for i in range(min(count, n))]
    x = torch.stack([data[s:s + context] for s in starts])
    y = torch.stack([data[s + 1:s + context + 1] for s in starts])
    return x, y


@torch.no_grad()
def val_loss(model, vx, vy, chunk: int = 16) -> float:
    """Mean loss over fixed windows. No random generator, so it never disturbs training."""
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(vx), chunk):
        total += model(vx[i:i + chunk], vy[i:i + chunk])[1].item() * len(vx[i:i + chunk])
        n += len(vx[i:i + chunk])
    model.train()
    return total / n


def make_optimizer(model, peak: float, weight_decay: float = 0.1):
    """Decay matrices only. Norm gains and biases are 1-D and stay free of decay."""
    decay = [p for p in model.parameters() if p.ndim >= 2]
    free = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": free, "weight_decay": 0.0}],
                             lr=peak, betas=(0.9, 0.95))


def best_val_so_far(log: Path) -> float:
    """Lowest val loss already logged, so a resumed run does not overwrite a better best.pt."""
    if not log.exists():
        return float("inf")
    with open(log, newline="") as f:
        vals = [float(r["val_loss"]) for r in csv.DictReader(f)]
    return min(vals, default=float("inf"))


def train(model, train_data, val_data, steps, batch, context, run_dir="checkpoints/run",
          seed=0, peak=3e-4, accum=1, eval_every=250, ckpt_every=500, clip=1.0, resume=True,
          weight_decay=0.1):
    """One call trains, logs to run_dir/log.csv, saves last.pt and best.pt, and resumes if last.pt exists."""
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "log.csv"
    vx, vy = fixed_windows(val_data, context, count=64)
    opt = make_optimizer(model, peak, weight_decay)
    start, best = 0, best_val_so_far(log)
    if resume and (out / "last.pt").exists():
        start = load(out / "last.pt", model, opt)
        print(f"resumed at step {start}")
    gen = torch.Generator().manual_seed(seed + start)       # a resumed run draws fresh batches
    new_log = not log.exists()
    t0, tok = time.time(), 0
    model.train()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["step", "train_loss", "val_loss", "lr", "tokens_per_s"])
        for step in range(start, steps):
            lr = lr_at(step, steps, peak)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            for _ in range(accum):                           # accumulate to a bigger effective batch
                x, y = get_batch(train_data, batch, context, gen)
                _, loss = model(x, y)
                (loss / accum).backward()
                tok += batch * context
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"loss is {loss.item()} at step {step}; lower the peak learning rate")
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            last = step == steps - 1
            if step % eval_every == 0 or last:
                v = val_loss(model, vx, vy)
                tps = tok / (time.time() - t0)
                w.writerow([step, f"{loss.item():.4f}", f"{v:.4f}", f"{lr:.2e}", f"{tps:.0f}"])
                f.flush()
                print(f"step {step:6d}  train {loss.item():.3f}  val {v:.3f}  {tps:,.0f} tok/s")
                if v < best:
                    best = v
                    save(out / "best.pt", model, opt, step + 1)
            if (step + 1) % ckpt_every == 0 or last:
                save(out / "last.pt", model, opt, step + 1)
    return model