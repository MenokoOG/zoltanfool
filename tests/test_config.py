from zoltanfool.config import DEBUG, TARGET, expected_params


def test_debug_param_count():
    assert expected_params(DEBUG) == 2_757_504


def test_head_dim():
    assert DEBUG.head_dim == 64
    assert TARGET.head_dim == 64
