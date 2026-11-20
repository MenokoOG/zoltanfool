import torch

from zoltanfool.config import Config
from zoltanfool.evalh import bits_per_byte, run_eval, valid_prefix_lines
from zoltanfool.model import ZoltanFool


def test_valid_prefix_lines():
    good = "def f(x):\n    y = x + 1\n    return y\n\ndef g(:\n   ???"
    assert valid_prefix_lines(good) >= 3
    assert valid_prefix_lines("def f(:\n  ???\n  ###\n  @@@") == 0
    assert valid_prefix_lines("x = 1\nx = 2\nx = 3\ny = (4,") == 3     # cut-off tail dropped


def test_bpb_deterministic_and_near_8_untrained():
    torch.manual_seed(0)
    m = ZoltanFool(Config(1, 32, 2, 32))
    v = torch.randint(0, 256, (4000,))
    a, b = bits_per_byte(m, v, 32), bits_per_byte(m, v, 32)
    assert a == b and 7.8 < a < 8.3


def test_run_eval_writes_json(tmp_path):
    m = ZoltanFool(Config(1, 32, 2, 32, rope=True))
    v = torch.randint(0, 256, (4000,))
    r = run_eval(m, v, m.c, seed=0, label="t", out_dir=str(tmp_path), n=4, length=20)
    assert len(r["samples"]["0.8"]) == 4 and list(tmp_path.glob("*.json"))
