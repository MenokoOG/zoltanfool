# zoltanfool

A small byte-level language model for Python code, written by hand in PyTorch. I'm building it module by module as a course lesson, starting from the data and the baselines it has to beat.

## Status

- Code for every stage is in: bigram, attention, full transformer, training loop with resume, eval harness, reward sandbox, rejection-sampling fine-tune, summarizer, and the Module 9 upgrades (RoPE, RMSNorm, SwiGLU, fused attention, key/value cache).
- 87 tests pass.
- No real training runs yet. Nothing below counts as a result until a run JSON exists in `results/`.

Step-by-step test and run commands for each module: [docs/lessons.md](docs/lessons.md).

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

Defined in `zoltanfool/config.py`. Tied head, no bias on the linear layers, context length 256. The `modern-*` configs switch on RoPE, RMSNorm and SwiGLU.

| Config | Layers | d_model | Heads | Parameters |
|---|---|---|---|---|
| `debug` | 6 | 192 | 3 | 2,757,504 |
| `mid` | 8 | 256 | 4 | 6,431,232 |
| `target` | 14 | 384 | 6 | 24,991,488 |

`expected_params(config)` computes the count. The model's `sum(p.numel() ...)` matches it exactly for every flag combination.

## Layout

```
data/build_corpus.py     corpus and train/val split
zoltanfool/              the library: config, tokenizer, baselines, model, train, checkpoint,
                         evalh, reward, sft_data, rft, summarize, registry
scripts/                 one command per job: baselines, train, sample, timing, size,
                         plot_loss, evaluate, sft, rft_loop
tasks/                   rev_string plus a 30-task ladder in three bands, every third held out
tests/                   pytest suite
results/                 run JSONs and loss curves the README cites
```

## Setup and tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
python -m pytest -q
ruff check .
```

The reward tests run generated code in a subprocess and refuse to run as root. Run them as a normal user.

## Quick start

```bash
python -m scripts.baselines
python -m scripts.train --model bigram --steps 300 --batch 32 --eval-every 50 --run-dir checkpoints/bigram
python -m scripts.train --model debug --steps 1000 --peak 1e-3 --run-dir checkpoints/debug
python -m scripts.evaluate checkpoints/debug/best.pt --label debug --seed 0
```
