"""Stage 4: save and resume. Writes to a temp file first so a crash cannot corrupt the last good one."""
import os

import torch
import torch.nn as nn


def save(path: str | os.PathLike, model: nn.Module, opt: torch.optim.Optimizer, step: int) -> None:
    tmp = f"{path}.tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, tmp)
    os.replace(tmp, path)


def load(path: str | os.PathLike, model: nn.Module, opt: torch.optim.Optimizer | None = None) -> int:
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"])
    if opt is not None:
        opt.load_state_dict(ck["opt"])
    return ck["step"]
