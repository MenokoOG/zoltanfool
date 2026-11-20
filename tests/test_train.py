import torch

from zoltanfool.config import Config
from zoltanfool.model import ZoltanFool
from zoltanfool.train import fixed_windows, get_batch, lr_at, make_optimizer, train, val_loss


def test_lr_shape():
    assert lr_at(0, 1000) < lr_at(99, 1000) <= 3e-4
    assert abs(lr_at(999, 1000) - 3e-5) < 3e-6


def test_fixed_windows_are_stable_and_shifted():
    d = torch.arange(10_000) % 256
    x1, y1 = fixed_windows(d, 64, 8)
    x2, _ = fixed_windows(d, 64, 8)
    assert torch.equal(x1, x2)
    assert torch.equal(x1[:, 1:], y1[:, :-1])


def test_no_decay_on_norms():
    m = ZoltanFool(Config(2, 48, 3, 32))
    opt = make_optimizer(m, 1e-3)
    assert opt.param_groups[0]["weight_decay"] > 0 and opt.param_groups[1]["weight_decay"] == 0
    assert all(p.ndim >= 2 for p in opt.param_groups[0]["params"])


def test_val_loss_does_not_touch_training_stream():
    d = torch.randint(0, 256, (5000,))
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(1)
    m = ZoltanFool(Config(1, 32, 2, 32))
    vx, vy = fixed_windows(d, 32, 8)
    val_loss(m, vx, vy)
    assert torch.equal(get_batch(d, 2, 32, g1)[0], get_batch(d, 2, 32, g2)[0])


def test_resume_continues_and_logs(tmp_path):
    torch.manual_seed(0)
    d = torch.randint(0, 256, (20_000,))
    c = Config(1, 32, 2, 32)
    run = str(tmp_path / "r")
    m = ZoltanFool(c)
    train(m, d, d, steps=20, batch=4, context=32, run_dir=run, eval_every=10, ckpt_every=10)
    m2 = ZoltanFool(c)
    train(m2, d, d, steps=30, batch=4, context=32, run_dir=run, eval_every=10, ckpt_every=10)
    rows = (tmp_path / "r" / "log.csv").read_text().splitlines()
    steps = [int(r.split(",")[0]) for r in rows[1:]]
    assert steps == [0, 10, 19, 20, 29], steps       # second run resumed at step 20, did not restart


def test_best_checkpoint_survives_resume(tmp_path):
    from zoltanfool.train import best_val_so_far
    log = tmp_path / "log.csv"
    log.write_text("step,train_loss,val_loss,lr,tokens_per_s\n0,5.5,5.4,1e-3,100\n10,3.0,3.1,1e-3,100\n")
    assert best_val_so_far(log) == 3.1
    assert best_val_so_far(tmp_path / "none.csv") == float("inf")
