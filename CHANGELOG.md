# Changelog

All notable changes to this project are recorded here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Bigram model with its own sampler, zero-init for an exact ln 256 start.
- Causal self-attention with explicit and fused paths, RoPE, and a key/value cache.
- Full transformer: MLP or SwiGLU, LayerNorm or RMSNorm, pre-norm blocks, tied head, GPT-2 residual init.
- `mid` and `modern-*` configs, `right_size` calculator, model registry with `spec.json` per run.
- `checkpoint.py` with atomic save; training loop logs `log.csv`, saves `last.pt` and `best.pt`, resumes.
- Eval harness (`evalh.py`): bits per byte, whole-line syntax rate, raw samples, one JSON per run.
- Reward sandbox with memory, CPU and file-size caps; unbiased pass@k.
- Docstring-then-body extractor, rejection-sampling fine-tune, per-band pass@k.
- 30-task ladder in three bands with a held-out split.
- `summarize.py` with dotted keys; skips files without the key.
- Scripts: baselines, train, sample, timing, size, plot_loss, evaluate, sft, rft_loop.
- `pyproject.toml` with pytest, ruff and mypy config; type hints across the library (mypy clean).
- `docs/lessons.md`, ADR 0002, tests for every module (87 total).

### Fixed
- `pass_at_k` returned 1.0 when no sample passed and n < k.

## [0.1.0]

### Added
- Corpus builder with normalized-hash dedupe and hash-based train/val split.
- Byte tokenizer and round-trip test.
- Uniform and unigram baselines.
- `DEBUG` and `TARGET` model configs with `expected_params`.
- README and project base files.
