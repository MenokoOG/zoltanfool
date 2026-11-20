import torch

from zoltanfool.config import Config
from zoltanfool.model import ZoltanFool
from zoltanfool.rft import finetune, text_to_ids


def test_text_to_ids_pads_short_text():
    ids = text_to_ids(["def f():\n    return 1\n"], context=32)
    assert len(ids) >= 4 * 33


def test_finetune_lowers_loss_on_kept_texts(tmp_path):
    torch.manual_seed(0)
    m = ZoltanFool(Config(2, 48, 3, 32, rope=True))
    texts = ["def add_one(x):\n    return x + 1\n"] * 4
    data = text_to_ids(texts, 32)
    x, y = data[:32].unsqueeze(0), data[1:33].unsqueeze(0)
    before = m(x, y)[1].item()
    finetune(m, texts, steps=150, batch=8, run_dir=str(tmp_path / "ft"), peak=3e-3, eval_every=149)
    assert m(x, y)[1].item() < before / 2
    assert (tmp_path / "ft" / "last.pt").exists()
