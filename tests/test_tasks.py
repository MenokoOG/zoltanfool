import os

import pytest

from tasks.ladder import ALL, HELDOUT, TRAIN
from zoltanfool.reward import reward

pytestmark = pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="reward refuses root")


def test_split_is_disjoint_and_covers_every_band():
    names_train, names_held = {t.name for t in TRAIN}, {t.name for t in HELDOUT}
    assert not names_train & names_held
    assert len(names_train | names_held) == len(ALL) == 30
    assert {t.band for t in HELDOUT} == {"trivial", "easy", "medium"}


@pytest.mark.parametrize("task", ALL, ids=lambda t: t.name)
def test_every_task_is_solvable_and_not_trivially_passed(task):
    assert reward(task.prompt, task.solution, task.tests) == 1
    assert reward(task.prompt, "    return None\n", task.tests) == 0
