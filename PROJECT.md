# zoltanfool

Lives on: WSL /home/jefft/dev/zoltanfool
Status: active
Updated: 2026-10-30

## What it is
A small byte-level language model for Python code, written by hand in PyTorch as a course lesson. Serves the senior AI/ML engineer role.

## Now
Module 2: Lawrence is writing the bigram model and the training loop. Module 1 (corpus, tokenizer, baselines, tests) is done.

## Next
1. Bigram model: step-0 loss near 5.545, val loss under the 3.389 unigram floor.
2. Transformer, checked against `expected_params(DEBUG)` = 2,757,504.
3. Init check, then the 14-layer TARGET run.

## Open rulings
- Close the 8% train/val leakage with MinHash or a boilerplate filter, or keep it as a known caveat · who decides: Lawrence · evidence: 7.8% of val 8-line windows also in train (2026-10-30 leak check)
- Add pyproject.toml or requirements.txt · who decides: Lawrence

## Links
- Repo: https://github.com/MenokoOG/zoltanfool (private)

<!-- One page max. Overwrite as things change. History goes in log/. -->
