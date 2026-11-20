import torch

from zoltanfool.checkpoint import save
from zoltanfool.config import right_size
from zoltanfool.registry import load_model, make_model, save_spec
from zoltanfool.train import make_optimizer


def test_right_size_matches_course_example():
    r = right_size(3000, 24.99e6, 10, 14.6e6)
    assert abs(r["params"] / 1e6 - 11.6) < 0.1
    assert abs(r["tokens"] / 1e6 - 232) < 2


def test_round_trip_through_spec_and_checkpoint(tmp_path):
    torch.manual_seed(0)
    m, spec = make_model("modern-debug", context_length=64)
    save_spec(tmp_path, spec)
    save(tmp_path / "best.pt", m, make_optimizer(m, 1e-3), 7)
    m2, spec2 = load_model(tmp_path / "best.pt")
    assert spec2["name"] == "modern-debug" and m2.c.context_length == 64
    x = torch.randint(0, 256, (1, 16))
    assert torch.allclose(m.eval()(x)[0], m2(x)[0])
