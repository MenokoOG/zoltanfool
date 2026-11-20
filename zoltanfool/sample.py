import torch
import torch.nn as nn


@torch.no_grad()
def generate(model: nn.Module, ids: torch.Tensor, n: int, temperature: float = 1.0,
             context: int = 256) -> torch.Tensor:
    for _ in range(n):
        logits, _ = model(ids[:, -context:])
        p = (logits[:, -1] / max(temperature, 1e-6)).softmax(-1)
        ids = torch.cat([ids, torch.multinomial(p, 1)], dim=1)
    return ids
