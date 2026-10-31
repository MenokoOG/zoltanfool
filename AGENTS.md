# AGENTS.md: zoltanfool

Read PROJECT.md first. Work on claude/* branches; never main. Plan, build, review, ship.

## What this repo is
A small byte-level language model for Python code, written by hand in PyTorch. A course lesson artifact: Lawrence writes each piece and the agent reviews it.
Lives on: WSL /home/jefft/dev/zoltanfool.

## Stack
- Python 3.14 in `.venv/`, PyTorch, numpy, pytest, ruff.

## Commands
- Build corpus: `python data/build_corpus.py <repo1> <repo2>` · Test: `.venv/bin/python -m pytest tests/` · Lint: `.venv/bin/ruff check .`

## Exemplar files
- `zoltanfool/config.py`: frozen dataclass config plus `expected_params`
- `tests/test_tokenizer.py`: round trip on multi-byte text

## Domain constraints
- Tutor mode. Review Lawrence's code; don't write the model or training loop for him.
- Step-0 loss should be near ln 256 = 5.545.
- Model parameter count must equal `expected_params(config)`.

## Do not touch
- `data/raw/*.bin`: rebuild, don't edit
