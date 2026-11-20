# zoltanfool

Lives on: WSL /home/jefft/dev/zoltanfool
Status: active
Updated: 2026-11-20

## What it is
A small byte-level language model for Python code, written by hand in PyTorch as a course lesson. Serves the senior AI/ML engineer role.

## Now
Code for every module (2 to 9) is built on branch `menokog-agent/full-model-build`: 87 tests pass, ruff and mypy clean. No real training run yet. Next is Lawrence running the module tests and the bigram run from docs/lessons.md.

## Next
1. Bigram run on the real corpus: step-0 loss 5.545, val under the 3.389 unigram floor, loss plot.
2. Timing at debug and target size on the laptop, then pick a size with `scripts.size`.
3. Debug-config run, then the eval JSONs for untrained, bigram and transformer.

## Open rulings
- Close the 8% train/val leakage with MinHash or a boilerplate filter, or keep it as a known caveat · who decides: Lawrence · evidence: 7.8% of val 8-line windows also in train (2026-10-30 leak check)
- Merge `menokog-agent/full-model-build` to main · who decides: Lawrence
- Agent docs (CLAUDE.md, AGENTS.md, PROJECT.md, DECISIONS.md, log/) are tracked; untrack before going public · who decides: Lawrence

## Links
- Repo: https://github.com/MenokoOG/zoltanfool (private)

<!-- One page max. Overwrite as things change. History goes in log/. -->
