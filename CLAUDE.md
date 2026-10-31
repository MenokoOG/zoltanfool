# CLAUDE.md: zoltanfool

Read PROJECT.md first. Standards: the agent library's rules/engineering.md.

## What this repo is
A small byte-level language model for Python code, written by hand in PyTorch. It is a course lesson artifact: Lawrence writes each piece himself and the agent reviews it.

Lives on: WSL /home/jefft/dev/zoltanfool. Run git and builds from that side only.

## Stack
- Python 3.14 in `.venv/`, PyTorch, numpy, pytest, ruff. No pyproject.toml or requirements file yet.

## Commands
- Install: `source .venv/bin/activate` (no install step yet)
- Build corpus: `python data/build_corpus.py <repo1> <repo2>` from the repo root, writes `data/raw/`
- Test: `.venv/bin/python -m pytest tests/`
- Lint: `.venv/bin/ruff check .`

## Layout
- `zoltanfool/`: config, tokenizer, baselines (model and training loop come next)
- `data/`: `build_corpus.py`; `data/raw/` holds the built bins and is gitignored
- `tests/`: pytest
- `results/`, `tasks/`: empty so far

## Exemplar files (reviews calibrate against these)
- `zoltanfool/config.py`: frozen dataclass config plus a closed-form `expected_params`
- `tests/test_tokenizer.py`: round trip on multi-byte text

## Conventions
- Types: Python type hints on every signature
- Errors: surfaced, never swallowed
- Naming: functions and variables snake_case; classes PascalCase; constants UPPER_SNAKE_CASE

## Domain constraints
- Tutor mode. Lawrence writes the model and training loop by hand. Review his code, point to the line and the reason, and don't write the solution.
- Step-0 loss should be near ln 256 = 5.545. Well above that means the initial weights are too large.
- Val loss reads about 8% optimistic from near-copy leakage between train and val. Don't compare across rebuilt corpora.
- The model's parameter count must equal `expected_params(config)`.

## Do not touch
- `data/raw/*.bin`: rebuild with `build_corpus.py`, don't edit
