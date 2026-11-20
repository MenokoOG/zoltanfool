"""Stage 5: one function, three numbers, one JSON file per run."""
import ast
import json
import math
import subprocess
import time
from pathlib import Path

import torch

from .config import Config
from .model import LanguageModel
from .tokenizer import decode, encode
from .train import fixed_windows


@torch.no_grad()
def bits_per_byte(model: LanguageModel, val_ids: torch.Tensor, context: int, chunk: int = 16) -> float:
    """Sweep every non-overlapping window of the validation set, in order. Same answer every call."""
    model.eval()
    x, y = fixed_windows(val_ids, context, count=(len(val_ids) - 1) // context)
    total = 0.0
    for i in range(0, len(x), chunk):
        total += model(x[i:i + chunk], y[i:i + chunk])[1].item() * len(x[i:i + chunk])
    return total / len(x) / math.log(2)


def valid_prefix_lines(text: str, min_lines: int = 3) -> int:
    """Lines in the longest whole-line prefix that parses as Python. 0 if under min_lines.

    A 200-byte sample is cut off mid-statement, so parsing the raw text almost always
    fails. Dropping the broken tail measures what the model wrote before it went wrong.
    """
    lines = text.splitlines()[:-1]                     # the last line is usually cut off
    while len(lines) >= min_lines:
        try:
            ast.parse("\n".join(lines))
            return len(lines)
        except (SyntaxError, ValueError, RecursionError):
            lines.pop()
    return 0


@torch.no_grad()
def sample_texts(model: LanguageModel, n: int, length: int, temperature: float, seed: int,
                 prompt: str = "def ") -> list[str]:
    torch.manual_seed(seed)
    model.eval()
    ids = torch.tensor([encode(prompt)]).repeat(n, 1)
    out = model.generate(ids, length, temperature)
    return [decode(r.tolist()) for r in out]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run_eval(model: LanguageModel, val_ids: torch.Tensor, cfg: Config, seed: int, label: str = "unlabeled",
             out_dir: str = "runs", temps: tuple[float, ...] = (0.5, 0.8, 1.0), n: int = 100, length: int = 300,
             extra: dict | None = None) -> dict:
    res = {"label": label, "seed": seed, "config": cfg.__dict__, "commit": git_commit(),
           "bits_per_byte": bits_per_byte(model, val_ids, cfg.context_length),
           "syntax": {}, "mean_valid_lines": {}, "samples": {}, "n_samples": n, **(extra or {})}
    for t in temps:
        texts = sample_texts(model, n, length, t, seed)
        lines = [valid_prefix_lines(x) for x in texts]
        res["syntax"][str(t)] = sum(1 for k in lines if k > 0) / n
        res["mean_valid_lines"][str(t)] = sum(lines) / n
        res["samples"][str(t)] = texts[:5]             # the first five, never hand-picked
    Path(out_dir).mkdir(exist_ok=True)
    p = Path(out_dir) / f"run-{label}-seed{seed}-{int(time.time())}.json"
    p.write_text(json.dumps(res, indent=2))
    return res
