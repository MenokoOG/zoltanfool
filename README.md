# zoltanfool

A small byte-level language model for Python code, written by hand in PyTorch. I'm building it module by module as a course lesson, starting from the data and the baselines it has to beat.

## Status

- Module 1 is done: corpus builder, byte tokenizer, two baselines, tests.
- Module 2 is in progress: a bigram model and the training loop.
- Not started: the transformer, the init check against ln 256, the 14-layer run.

## Corpus

`data/build_corpus.py` collects `.py` files from the repos you point it at, drops files under 200 bytes, and drops duplicates by hashing a normalized copy (stripped lines, no blank or comment-only lines). The hash also picks the split, so a file always lands on the same side: about 1 in 10 goes to val.

My current build, from two local repos:

| Split | Files | Size |
|---|---|---|
| train | 2,996 | 14.6 MB |
| val | 288 | 1.3 MB |

About 8% of val's 8-line windows also appear in train. The dedupe catches exact and whitespace-only copies, so edited near-copies still leak. Treat val loss as slightly optimistic.

The bins aren't in the repo. Build them yourself:

```bash
python3 data/build_corpus.py /path/to/repo1 /path/to/repo2
```

This writes `data/raw/train.bin` and `data/raw/val.bin`. Run it from the repo root.

## Tokenizer

`zoltanfool/tokenizer.py` encodes text to UTF-8 bytes. The vocab is 256, so there is nothing to train. `decode` replaces broken byte sequences with `�`, which a model sampling one byte at a time will sometimes produce.

## Baselines

Two floors, in nats per byte, measured on my current val:

| Baseline | Nats | Bits |
|---|---|---|
| Uniform (ln 256) | 5.545 | 8.00 |
| Unigram, add-one smoothed | 3.389 | 4.89 |

A model's first loss should land near 5.545. If it starts well above that, the initial weights are too large. A model that can't get under 3.389 has learned nothing about context.

## Model sizes

Defined in `zoltanfool/config.py`. Tied head, no bias on the linear layers, context length 256.

| Config | Layers | d_model | Heads | Parameters |
|---|---|---|---|---|
| `DEBUG` | 6 | 192 | 3 | 2,757,504 |
| `TARGET` | 14 | 384 | 6 | 24,991,488 |

`expected_params(config)` computes the count. The model's `sum(p.numel() ...)` should match it exactly.

## Setup and tests

The repo has a `.venv` with torch, numpy, pytest and ruff. There's no `pyproject.toml` or requirements file yet.

```bash
source .venv/bin/activate
python -m pytest tests/
```
