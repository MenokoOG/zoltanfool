from zoltanfool.config import DEBUG, MID, MODERN_DEBUG, TARGET, expected_params


def test_debug_param_count():
    assert expected_params(DEBUG) == 2_757_504


def test_head_dim():
    assert DEBUG.head_dim == 64
    assert TARGET.head_dim == 64


def test_mid_param_count():
    assert expected_params(MID) == 6_431_232


def test_swiglu_hidden_is_multiple_of_64():
    assert MODERN_DEBUG.ffn_hidden % 64 == 0
