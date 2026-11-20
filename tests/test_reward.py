import os

import pytest

from tasks.rev_string import PROMPT, TESTS
from zoltanfool.reward import pass_at_k, reward

pytestmark = pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="reward refuses root")


def test_good_bad_loop_and_memory_bomb():
    assert reward(PROMPT, "    return s[::-1]", TESTS) == 1
    assert reward(PROMPT, "    return s", TESTS) == 0
    assert reward(PROMPT, "    while True: pass", TESTS, timeout=1) == 0
    assert reward(PROMPT, "    x = [0] * (10**10)\n    return s[::-1]", TESTS) == 0


def test_pass_at_k():
    assert abs(pass_at_k(10, 5, 1) - 0.5) < 1e-9
    assert abs(pass_at_k(10, 5, 5) - (1 - 1 / 252)) < 1e-9
    assert abs(pass_at_k(20, 4, 1) - 0.2) < 1e-9


def test_pass_at_k_edges():
    assert pass_at_k(2, 0, 1) == 0.0
    assert pass_at_k(5, 0, 5) == 0.0
    with pytest.raises(ValueError):
        pass_at_k(2, 1, 5)
