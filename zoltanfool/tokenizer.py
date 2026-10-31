"""Bytes in, ints out. Vocab is 256, so there is nothing to train."""

def encode(text: str) -> list[int]:
    """Encode a string to a list of integers."""
    return list(text.encode("utf-8"))


def decode(ids: list[int]) -> str:
    return bytes(ids).decode("utf-8", errors="replace")
