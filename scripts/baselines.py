"""Module 1 project: the two floors on your corpus, and a tokenizer check.

python -m scripts.baselines
"""
from scripts._data import RAW
from zoltanfool.baselines import nats_to_bits, uniform_nats, unigram_nats
from zoltanfool.tokenizer import decode, encode


def main() -> None:
    train, val = (RAW / "train.bin").read_bytes(), (RAW / "val.bin").read_bytes()
    print(f"train {len(train):,} bytes   val {len(val):,} bytes")
    u, g = uniform_nats(), unigram_nats(train, val)
    print(f"uniform  {u:.3f} nats  {nats_to_bits(u):.2f} bits")
    print(f"unigram  {g:.3f} nats  {nats_to_bits(g):.2f} bits")
    text = val.decode("utf-8", errors="strict") if val.isascii() else None
    if text is not None:
        assert decode(encode(text)) == text
        print("tokenizer round trip on val: ok")
    else:
        ok = bytes(encode(val.decode("utf-8", "replace"))) == val
        print(f"val has non-ASCII bytes; exact byte round trip: {'ok' if ok else 'lossy (invalid UTF-8 in corpus)'}")


if __name__ == "__main__":
    main()
