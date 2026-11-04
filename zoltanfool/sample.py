import torch


@torch.no_grad()
def generate(model, ids, n, temperature=1.0, context=256):
    for _ in range(n):
        logits, _ = model(ids[:, -context:])
        p = (logits[:, -1] / max(temperature, 1e-6)).softmax(-1)
        ids = torch.cat([ids, torch.multinomial(p, 1)], dim=1)
    return ids