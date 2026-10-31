"""Two floors to beat, measured in nats per byte."""
import math
from collections import Counter


def uniform_nats(vocab: int = 256) -> float:
    return math.log(vocab)


def unigram_nats(train: bytes, val: bytes) -> float:
    """Cross-entropy of val under the train byte frequencies, add-one smoothed."""
    counts = Counter(train)
    total = len(train) + 256
    p = {b: (counts.get(b, 0) + 1) / total for b in range(256)}
    return -sum(math.log(p[b]) for b in val) / len(val)


def nats_to_bits(x: float) -> float:
    return x / math.log(2)