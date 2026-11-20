# 2. Scripts, model registry and a spec file per run

Date: 2026-11-20

## Status

Proposed

## Context

The course gives library code (model, train, eval, reward) but no commands to run it. Every later
stage reloads a checkpoint, so a checkpoint has to say which model built it. The course also adds
three architecture flags (Module 9) that must stay switchable for the ablation in Module 8.

## Decision

- Library code lives in `zoltanfool/`. Each command is one small file in `scripts/`, run as
  `python -m scripts.<name>` from the repo root, so `tasks/` and `data/` import without an install.
- `zoltanfool/registry.py` builds a model by name (`bigram`, `debug`, `mid`, `target`, `modern-*`)
  and writes `spec.json` next to the checkpoints. `load_model` rebuilds from it.
- The upgrades are config flags. All off is the plain model from Modules 3 and 4.
- Every model, bigram included, exposes `.c` (its Config) and `.generate`, typed as `LanguageModel`.

## Consequences

- One loader for eval, sampling, SFT and the reward loop.
- `checkpoint.py` keeps the course's format (model, opt, step). Config lives in `spec.json`, so a
  `.pt` copied away from its folder can't be reloaded without that file.
- Scripts depend on the working directory being the repo root.
