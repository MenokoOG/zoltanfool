"""The task ladder for Stage 6: 30 small function tasks in three difficulty bands.

Each task is a signature plus a one-line docstring (the prompt) and asserts (the judge).
`solution` exists only so tests/test_tasks.py can prove every task is solvable. It is
never shown to the model and never written into the corpus.

Split rule: inside each band, every third task is held out. Train on TRAIN, report on HELDOUT.
Keep these names and bodies out of the training corpus, or the held-out numbers mean nothing.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class LadderTask:
    name: str
    band: str          # trivial, easy, medium
    signature: str     # e.g. "def add_one(x):"
    doc: str
    tests: str
    solution: str      # body, indented four spaces; for the solvability test only

    @property
    def prompt(self) -> str:
        return f'{self.signature}\n    """{self.doc}"""\n'


def _t(band, name, args, doc, tests, solution):
    return LadderTask(name, band, f"def {name}({args}):", doc, tests.strip() + "\n", solution)


TRIVIAL = [
    _t("trivial", "add_one", "x", "Return x plus one.",
       "assert add_one(1) == 2\nassert add_one(-1) == 0", "    return x + 1\n"),
    _t("trivial", "return_true", "", "Return True.",
       "assert return_true() is True", "    return True\n"),
    _t("trivial", "first_char", "s", "Return the first character of s.",
       "assert first_char('abc') == 'a'\nassert first_char('z') == 'z'", "    return s[0]\n"),
    _t("trivial", "double", "x", "Return x times two.",
       "assert double(3) == 6\nassert double(0) == 0", "    return x * 2\n"),
    _t("trivial", "identity", "x", "Return x unchanged.",
       "assert identity(5) == 5\nassert identity('a') == 'a'", "    return x\n"),
    _t("trivial", "last_item", "xs", "Return the last item of the list xs.",
       "assert last_item([1, 2, 3]) == 3\nassert last_item(['a']) == 'a'", "    return xs[-1]\n"),
    _t("trivial", "is_zero", "x", "Return True if x is zero.",
       "assert is_zero(0) is True\nassert is_zero(4) is False", "    return x == 0\n"),
    _t("trivial", "length", "xs", "Return the number of items in xs.",
       "assert length([1, 2]) == 2\nassert length([]) == 0", "    return len(xs)\n"),
    _t("trivial", "negate", "x", "Return minus x.",
       "assert negate(3) == -3\nassert negate(-2) == 2", "    return -x\n"),
    _t("trivial", "square", "x", "Return x squared.",
       "assert square(3) == 9\nassert square(-2) == 4", "    return x * x\n"),
]

EASY = [
    _t("easy", "rev_string", "s", "Return s reversed.",
       "assert rev_string('abc') == 'cba'\nassert rev_string('') == ''", "    return s[::-1]\n"),
    _t("easy", "add", "a, b", "Return the sum of a and b.",
       "assert add(2, 3) == 5\nassert add(-1, 1) == 0", "    return a + b\n"),
    _t("easy", "max_of_two", "a, b", "Return the larger of a and b.",
       "assert max_of_two(2, 5) == 5\nassert max_of_two(7, 1) == 7", "    return a if a > b else b\n"),
    _t("easy", "is_even", "n", "Return True if n is even.",
       "assert is_even(4) is True\nassert is_even(7) is False", "    return n % 2 == 0\n"),
    _t("easy", "upper", "s", "Return s in upper case.",
       "assert upper('ab') == 'AB'\nassert upper('') == ''", "    return s.upper()\n"),
    _t("easy", "sum_list", "xs", "Return the sum of the numbers in xs.",
       "assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0", "    return sum(xs)\n"),
    _t("easy", "abs_value", "x", "Return the absolute value of x.",
       "assert abs_value(-3) == 3\nassert abs_value(2) == 2", "    return x if x >= 0 else -x\n"),
    _t("easy", "is_empty", "s", "Return True if the string s is empty.",
       "assert is_empty('') is True\nassert is_empty('a') is False", "    return len(s) == 0\n"),
    _t("easy", "join_words", "words", "Join the list of words with single spaces.",
       "assert join_words(['a', 'b']) == 'a b'\nassert join_words([]) == ''", "    return ' '.join(words)\n"),
    _t("easy", "min_list", "xs", "Return the smallest number in xs.",
       "assert min_list([3, 1, 2]) == 1\nassert min_list([5]) == 5", "    return min(xs)\n"),
]

MEDIUM = [
    _t("medium", "count_vowels", "s", "Return how many vowels (aeiou) are in s.",
       "assert count_vowels('hello') == 2\nassert count_vowels('xyz') == 0",
       "    return sum(1 for c in s if c in 'aeiou')\n"),
    _t("medium", "sum_evens", "xs", "Return the sum of the even numbers in xs.",
       "assert sum_evens([1, 2, 3, 4]) == 6\nassert sum_evens([1, 3]) == 0",
       "    return sum(x for x in xs if x % 2 == 0)\n"),
    _t("medium", "is_palindrome", "s", "Return True if s reads the same forwards and backwards.",
       "assert is_palindrome('abba') is True\nassert is_palindrome('abc') is False",
       "    return s == s[::-1]\n"),
    _t("medium", "factorial", "n", "Return n factorial.",
       "assert factorial(0) == 1\nassert factorial(5) == 120",
       "    out = 1\n    for i in range(2, n + 1):\n        out *= i\n    return out\n"),
    _t("medium", "fizzbuzz", "n", "Return 'Fizz', 'Buzz', 'FizzBuzz' or str(n).",
       "assert fizzbuzz(3) == 'Fizz'\nassert fizzbuzz(5) == 'Buzz'\n"
       "assert fizzbuzz(15) == 'FizzBuzz'\nassert fizzbuzz(7) == '7'",
       "    if n % 15 == 0:\n        return 'FizzBuzz'\n    if n % 3 == 0:\n        return 'Fizz'\n"
       "    if n % 5 == 0:\n        return 'Buzz'\n    return str(n)\n"),
    _t("medium", "count_words", "s", "Return the number of whitespace-separated words in s.",
       "assert count_words('a b  c') == 3\nassert count_words('') == 0", "    return len(s.split())\n"),
    _t("medium", "unique", "xs", "Return the items of xs without duplicates, keeping first-seen order.",
       "assert unique([1, 2, 1, 3]) == [1, 2, 3]\nassert unique([]) == []",
       "    out = []\n    for x in xs:\n        if x not in out:\n            out.append(x)\n    return out\n"),
    _t("medium", "fib", "n", "Return the n-th Fibonacci number, with fib(0) == 0 and fib(1) == 1.",
       "assert fib(0) == 0\nassert fib(1) == 1\nassert fib(10) == 55",
       "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"),
    _t("medium", "char_count", "s, c", "Return how many times the character c appears in s.",
       "assert char_count('banana', 'a') == 3\nassert char_count('', 'a') == 0", "    return s.count(c)\n"),
    _t("medium", "clamp", "x, lo, hi", "Return x limited to the range lo to hi.",
       "assert clamp(5, 0, 3) == 3\nassert clamp(-1, 0, 3) == 0\nassert clamp(2, 0, 3) == 2",
       "    return max(lo, min(x, hi))\n"),
]

ALL = TRIVIAL + EASY + MEDIUM


def _split(band: list[LadderTask]):
    held = [t for i, t in enumerate(band) if i % 3 == 2]
    train = [t for i, t in enumerate(band) if i % 3 != 2]
    return train, held


TRAIN: list[LadderTask] = []
HELDOUT: list[LadderTask] = []
for _band in (TRIVIAL, EASY, MEDIUM):
    _tr, _ho = _split(_band)
    TRAIN += _tr
    HELDOUT += _ho
