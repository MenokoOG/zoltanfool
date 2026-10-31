from zoltanfool.tokenizer import decode, encode
from zoltanfool.baselines import uniform_nats


def test_tokenizer_round_trip():
    s = "def f(x):\n    return x * 2  # naïve ✓\n"
    assert decode(encode(s)) == s


def test_uniform_baseline():
    assert abs(uniform_nats() - 5.545) < 0.001