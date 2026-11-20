"""Stage 6: rejection-sampling fine-tuning. Sample, keep what passes, train on it, measure again."""
from dataclasses import dataclass
from typing import Protocol, Sequence

import torch

from .model import LanguageModel
from .reward import pass_at_k, reward
from .tokenizer import decode, encode
from .train import train


@dataclass(frozen=True)
class Task:
    name: str
    prompt: str
    tests: str


class TaskLike(Protocol):
    """Anything with a name, a prompt and tests: Task here, LadderTask in tasks/ladder.py."""
    @property
    def name(self) -> str: ...
    @property
    def prompt(self) -> str: ...
    @property
    def tests(self) -> str: ...


def cut_completion(text: str) -> str:
    """Keep the function body only. Stop at the first non-empty line that is not indented."""
    kept = []
    for line in text.splitlines(keepends=True):
        if line.strip() and not line[0].isspace():
            break
        kept.append(line)
    return "".join(kept).rstrip() + "\n"


@torch.no_grad()
def sample_completions(model: LanguageModel, task: TaskLike, n: int, length: int, temperature: float) -> list[str]:
    model.eval()
    ids = torch.tensor([encode(task.prompt)]).repeat(n, 1)
    out = model.generate(ids, length, temperature)
    start = ids.size(1)
    return [cut_completion(decode(r[start:].tolist())) for r in out]


def measure(model: LanguageModel, tasks: Sequence[TaskLike], n: int = 20, length: int = 160, temperature: float = 0.8,
            ks: tuple[int, ...] = (1, 5)) -> dict:
    """Mean pass@k over tasks, plus the passing texts so the caller can keep them."""
    ks = tuple(k for k in ks if k <= n)               # pass@k is undefined when k > n
    per_k: dict[int, list[float]] = {k: [] for k in ks}
    kept = []
    for t in tasks:
        comps = sample_completions(model, t, n, length, temperature)
        passed = [c for c in comps if reward(t.prompt, c, t.tests)]
        for k in ks:
            per_k[k].append(pass_at_k(n, len(passed), k))
        kept += [t.prompt + c for c in passed]
    return {"pass_at": {k: sum(v) / len(v) for k, v in per_k.items()}, "kept": kept}


def by_band(model: LanguageModel, tasks: Sequence[TaskLike], n: int = 20, length: int = 160, temperature: float = 0.8,
            ks: tuple[int, ...] = (1, 5)) -> dict:
    """pass@k per difficulty band. Tasks need a .band attribute. Report this, not one blended number."""
    bands: dict[str, list] = {}
    for t in tasks:
        bands.setdefault(getattr(t, "band", "all"), []).append(t)
    return {b: measure(model, ts, n, length, temperature, ks)["pass_at"] for b, ts in bands.items()}


def text_to_ids(texts: list[str], context: int) -> torch.Tensor:
    """Join texts with blank lines and repeat until there is room for at least four windows."""
    ids = encode("\n\n".join(texts) + "\n\n")
    if not ids:
        raise ValueError("no texts to train on")
    while len(ids) < 4 * (context + 1):
        ids = ids + ids
    return torch.tensor(ids, dtype=torch.long)


def finetune(model: LanguageModel, texts: list[str], steps: int, batch: int, run_dir: str, peak: float = 1e-4,
             seed: int = 0, eval_every: int = 50) -> LanguageModel:
    """The fine-tune step: train() from Module 2 on the kept texts, small rate, fresh run, own run_dir."""
    context = model.c.context_length
    data = text_to_ids(texts, context)
    train(model, data, data, steps=steps, batch=batch, context=context, run_dir=run_dir,
          seed=seed, peak=peak, eval_every=eval_every, ckpt_every=steps, resume=False,
          weight_decay=0.0)
    return model
