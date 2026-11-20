import torch
import torch.nn.functional as F

from zoltanfool.config import Config
from zoltanfool.model import CausalSelfAttention, ZoltanFool


def test_attention_matches_pytorch():
    torch.manual_seed(0)
    c = Config(n_layer=1, d_model=48, n_head=3, context_length=16)
    a = CausalSelfAttention(c)
    x = torch.randn(2, 10, 48)
    B, T, d = x.shape
    sp = a.split
    q, k, v = sp(a.q(x)), sp(a.k(x)), sp(a.v(x))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = a.out(ref.transpose(1, 2).reshape(B, T, d))
    assert torch.allclose(a(x), ref, atol=1e-5)


def test_causality():
    torch.manual_seed(0)
    m = ZoltanFool(Config(n_layer=2, d_model=48, n_head=3, context_length=32)).eval()
    x = torch.randint(0, 256, (1, 20))
    before, _ = m(x)
    x2 = x.clone()
    x2[0, 15] = (x2[0, 15] + 1) % 256                  # change a future byte
    after, _ = m(x2)
    assert torch.allclose(before[0, :15], after[0, :15], atol=1e-6)
    assert not torch.allclose(before[0, 15:], after[0, 15:])
