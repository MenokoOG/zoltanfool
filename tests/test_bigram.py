import math

import torch

from zoltanfool.registry import make_model
from zoltanfool.train import train


def test_zero_init_bigram_starts_at_ln256():
    m, _ = make_model("bigram")
    x = torch.randint(0, 256, (4, 64))
    assert abs(m(x, torch.roll(x, -1, 1))[1].item() - math.log(256)) < 1e-6


def test_bigram_learns_a_repeating_pattern(tmp_path):
    torch.manual_seed(0)
    data = torch.tensor(list(b"abcabcabc" * 500))
    m, _ = make_model("bigram")
    train(m, data, data, steps=400, batch=8, context=32, run_dir=str(tmp_path), peak=1e-1,
          weight_decay=0.0, eval_every=399, ckpt_every=1000)
    x = data[:33].unsqueeze(0)
    assert m(x[:, :-1], x[:, 1:])[1].item() < 0.5


def test_bigram_generate_shape():
    m, _ = make_model("bigram")
    assert m.generate(torch.tensor([[100]]), 10).shape == (1, 11)
