import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from zoltanfool.config import DEBUG, MODERN_DEBUG, MODERN_TARGET, TARGET, Config, expected_params
from zoltanfool.model import CausalSelfAttention, ZoltanFool

FLAGS = list(itertools.product([False, True], repeat=3))


def small(rope=False, rmsnorm=False, swiglu=False, **kw):
    base = dict(n_layer=2, d_model=48, n_head=3, context_length=32, rope=rope, rmsnorm=rmsnorm, swiglu=swiglu)
    return Config(**{**base, **kw})


def test_param_counts():
    for c, n in ((DEBUG, 2_757_504), (TARGET, 24_991_488)):
        assert expected_params(c) == n
        assert sum(p.numel() for p in ZoltanFool(c).parameters()) == n
    for c in (MODERN_DEBUG, MODERN_TARGET):
        assert sum(p.numel() for p in ZoltanFool(c).parameters()) == expected_params(c)


@pytest.mark.parametrize("rope,rmsnorm,swiglu", FLAGS)
def test_param_count_every_flag_combo(rope, rmsnorm, swiglu):
    c = small(rope, rmsnorm, swiglu)
    assert sum(p.numel() for p in ZoltanFool(c).parameters()) == expected_params(c)


@pytest.mark.parametrize("rope,rmsnorm,swiglu", FLAGS)
def test_causality_every_flag_combo(rope, rmsnorm, swiglu):
    torch.manual_seed(0)
    m = ZoltanFool(small(rope, rmsnorm, swiglu)).eval()
    x = torch.randint(0, 256, (1, 20))
    before, _ = m(x)
    x2 = x.clone()
    x2[0, 15] = (x2[0, 15] + 1) % 256
    after, _ = m(x2)
    assert torch.allclose(before[0, :15], after[0, :15], atol=1e-6)
    assert not torch.allclose(before[0, 15:], after[0, 15:])


def test_attention_matches_pytorch():
    torch.manual_seed(0)
    c = small()
    a = CausalSelfAttention(c)
    x = torch.randn(2, 10, 48)
    B, T, d = x.shape
    q, k, v = a.split(a.q(x)), a.split(a.k(x)), a.split(a.v(x))
    ref = a.out(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, d))
    assert torch.allclose(a(x), ref, atol=1e-5)


@pytest.mark.parametrize("rope", [False, True])
def test_fast_path_equals_explicit_path(rope):
    torch.manual_seed(0)
    m = ZoltanFool(small(rope=rope)).eval()
    x = torch.randint(0, 256, (3, 24))
    slow, _ = m(x)
    m.set_fast(True)
    fast, _ = m(x)
    assert torch.allclose(slow, fast, atol=1e-5)


@pytest.mark.parametrize("rope", [False, True])
def test_kv_cache_matches_full_forward(rope):
    torch.manual_seed(0)
    m = ZoltanFool(small(rope=rope)).eval()
    x = torch.randint(0, 256, (2, 20))
    full, _ = m(x)
    caches = [{} for _ in m.blocks]
    first, _ = m(x[:, :12], caches=caches)                 # prompt in one go
    steps = [first]
    for t in range(12, 20):                                 # then one byte at a time
        out, _ = m(x[:, t:t + 1], caches=caches, pos=t)
        steps.append(out)
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-5)


def test_generate_runs_past_the_window():
    torch.manual_seed(0)
    m = ZoltanFool(small(rope=True)).eval()
    out = m.generate(torch.randint(0, 256, (2, 5)), 70, 1.0)   # context is 32, so the cache restarts
    assert out.shape == (2, 75)


def test_step_zero_loss_near_ln256():
    torch.manual_seed(0)
    for c in (DEBUG, MODERN_DEBUG):
        m = ZoltanFool(c)
        x = torch.randint(0, 256, (4, 64))
        loss = m(x, torch.roll(x, -1, 1))[1].item()
        assert abs(loss - math.log(256)) < 0.1, loss


def test_overfits_one_batch():
    torch.manual_seed(0)
    m = ZoltanFool(small(n_head=4, d_model=64, rope=True, rmsnorm=True, swiglu=True))
    x = torch.randint(0, 256, (4, 32))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
