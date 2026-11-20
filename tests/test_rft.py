from zoltanfool.rft import cut_completion
from zoltanfool.sft_data import function_examples

SRC = '''import os

def add(a, b):
    """Add two numbers."""
    total = a + b
    return total

def no_doc(x):
    return x

class K:
    def m(self):
        """skip methods"""
        return 1
'''


def test_function_examples():
    ex = function_examples(SRC)
    assert len(ex) == 1
    prompt, body = ex[0]
    assert prompt.endswith('"""Add two numbers."""\n') and "return total" in body
    exec(prompt + body + "\nassert add(1, 2) == 3")


def test_cut_completion_stops_at_next_top_level_line():
    text = "    return s[::-1]\n\ndef other():\n    pass\n"
    assert cut_completion(text) == "    return s[::-1]\n"
