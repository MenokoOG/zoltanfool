"""Build a model by name, and rebuild it from a run directory. Scripts use this; the core modules do not."""
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .checkpoint import load
from .config import CONFIGS, Config
from .model import Bigram, LanguageModel, ZoltanFool

NAMES = ["bigram", *CONFIGS]


def make_model(name: str, context_length: int | None = None) -> tuple[LanguageModel, dict]:
    """'bigram' or a CONFIGS key. Returns (model, spec) where spec is enough to rebuild it."""
    if name == "bigram":
        model = Bigram(context_length=context_length or 256)
        torch.nn.init.zeros_(model.table.weight)      # equal logits: step-0 loss is exactly ln 256
        return model, {"kind": "bigram", "name": name, "config": asdict(model.c)}
    if name not in CONFIGS:
        raise ValueError(f"unknown model {name!r}; pick one of {NAMES}")
    cfg = CONFIGS[name]
    if context_length:
        cfg = replace(cfg, context_length=context_length)
    return ZoltanFool(cfg), {"kind": "transformer", "name": name, "config": asdict(cfg)}


def save_spec(run_dir: str | Path, spec: dict) -> None:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "spec.json").write_text(json.dumps(spec, indent=2))


def load_model(ckpt: str | Path) -> tuple[LanguageModel, dict]:
    """Read spec.json next to the checkpoint, build the same model, load the weights."""
    ckpt = Path(ckpt)
    spec = json.loads((ckpt.parent / "spec.json").read_text())
    cfg = Config(**spec["config"])
    model: LanguageModel = (Bigram(cfg.vocab_size, cfg.context_length) if spec["kind"] == "bigram"
                            else ZoltanFool(cfg))
    load(ckpt, model)
    return model.eval(), spec
