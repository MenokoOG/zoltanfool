import torch
from zoltanfool.config import DEBUG, TARGET, Config, expected_params
from zoltanfool.model import ZoltanFool


def test_param_counts():
    for c, n in ((DEBUG, 2_757_504), (TARGET, 24_991_488)):
        assert expected_params(c) == n
        m = ZoltanFool(c)
        assert sum(p.numel() for p in m.parameters()) == n


def test_overfits_one_batch():
    torch.manual_seed(0)
    c = Config(n_layer=2, d_model=64, n_head=4, context_length=32)
    m = ZoltanFool(c)
    x = torch.randint(0, 256, (4, 32))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = m(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.1