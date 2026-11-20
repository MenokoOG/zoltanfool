# Lesson runbook: test it, then run it

Every command runs from the repo root in WSL, with the venv on:

```bash
cd /home/jefft/dev/zoltanfool
source .venv/bin/activate
```

Scripts run as modules (`python -m scripts.x`) so `zoltanfool`, `tasks` and `data` import without an install.
Never run any of this as root. The reward sandbox refuses root, and those tests skip.

Whole suite, any time: `python -m pytest -q` (87 tests). Lint and types: `ruff check . && mypy`

## Module 1: bytes, loss and baselines

Test:

```bash
python -m pytest -q tests/test_tokenizer.py tests/test_config.py
```

Run:

```bash
python3 data/build_corpus.py /path/to/repo1 /path/to/repo2 \
  "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["stdlib"])')"
python -m scripts.baselines
```

Write down the unigram number in nats. Modules 2 and 6 compare against it.

## Module 2: the bigram and the training loop

Test:

```bash
python -m pytest -q tests/test_bigram.py
```

Run (zero init, peak 1e-1, no weight decay are the defaults for `--model bigram`):

```bash
python -m scripts.train --model bigram --steps 300 --batch 32 --eval-every 50 --run-dir checkpoints/bigram
python -m scripts.sample checkpoints/bigram/best.pt --temps 0.5 0.8 1.0
python -m scripts.plot_loss checkpoints/bigram --unigram <your unigram nats> --out results/bigram-loss.png
```

Step 0 should read 5.545. Val should end well under your unigram floor.
To see why 3e-4 is too slow, rerun with `--peak 3e-4 --run-dir checkpoints/bigram-slow --fresh` and plot both.

## Module 3: attention by hand

Test:

```bash
python -m pytest -q tests/test_attention.py
```

Break it on purpose: in `zoltanfool/model.py`, change `scores.masked_fill(~mask, ...)` to
`scores.masked_fill(torch.zeros_like(mask), ...)`. Rerun. The causality test must fail. Put it back.

## Module 4: the block and the parameter count

Test:

```bash
python -m pytest -q tests/test_model.py -k "param_counts or overfits or ln256"
```

Run the full model on the debug config and compare with the bigram:

```bash
python -m scripts.train --model debug --steps 1000 --batch 8 --peak 1e-3 --eval-every 100 --run-dir checkpoints/debug
python -m scripts.sample checkpoints/debug/best.pt
python -m scripts.plot_loss checkpoints/bigram checkpoints/debug --unigram <nats> --out results/bigram-vs-debug.png
```

## Module 5: scale and measure

Test:

```bash
python -m pytest -q tests/test_train.py tests/test_registry_and_size.py
```

Run (plugged in, best performance mode, inside tmux):

```bash
python -m scripts.timing --models debug target --steps 50 --batch 8
python -m scripts.timing --models debug --steps 50 --batch 8 --threads 4    # try physical cores
python -m scripts.size --tps <debug tok/s> --params 2.76 --hours <hours> --corpus-mb <train MB>
```

Learning-rate sweep, 300 steps each, then compare the last val numbers:

```bash
for lr in 3e-4 1e-3 3e-3; do
  python -m scripts.train --model debug --steps 300 --batch 8 --peak $lr --eval-every 50 --run-dir checkpoints/lr-$lr
done
```

Kill test: start a long run, press Ctrl+C halfway, run the same command again. It prints
`resumed at step N` and `log.csv` keeps going.

## Module 9: better model, faster loop (do this before 6)

Test:

```bash
python -m pytest -q tests/test_model.py
```

That covers all eight flag combinations, fused equals explicit, and cached equals full forward.
Run plain against upgraded at your size, 3 seeds each:

```bash
for s in 0 1 2; do
  python -m scripts.train --model debug        --steps 1000 --peak 1e-3 --seed $s --run-dir checkpoints/plain-s$s --fast
  python -m scripts.train --model modern-debug --steps 1000 --peak 1e-3 --seed $s --run-dir checkpoints/modern-s$s --fast
done
```

## Module 6: the eval harness

Test:

```bash
python -m pytest -q tests/test_evalh.py
```

Run, one JSON per run in `runs/`:

```bash
python -m scripts.evaluate --untrained debug --label untrained --seed 0
python -m scripts.evaluate checkpoints/bigram/best.pt --label bigram --seed 0
python -m scripts.evaluate checkpoints/debug/best.pt --label debug --seed 0
```

## Module 7: the reward loop

Test:

```bash
python -m pytest -q tests/test_reward.py tests/test_rft.py tests/test_rft_finetune.py tests/test_tasks.py
```

Run. First the format fine-tune (fixes the zero-pass problem), then the loop on 3 seeds:

```bash
python -m scripts.sft checkpoints/debug/best.pt /path/to/repo1 /path/to/repo2 --steps 300 --run-dir checkpoints/sft
for s in 0 1 2; do
  python -m scripts.rft_loop checkpoints/sft/model.pt --rounds 2 --n 20 --seed $s --run-dir checkpoints/rft-s$s
done
```

If `scripts.sft` prints 0 docstring functions, point it at repos with documented top-level functions under 400 characters.
Each run writes `runs/rft-seed*.json` with held-out pass@1 and pass@5 per band, before and after.
The task ladder is `tasks/ladder.py`: 30 tasks, 10 per band, every third one held out.

## Module 8: experiments and honest results

Test:

```bash
python -m pytest -q tests/test_summarize.py
```

Run, after each experiment has 3 seeds evaluated with the same `--label`:

```bash
python -m zoltanfool.summarize runs                 # bits per byte
python -m zoltanfool.summarize runs syntax.0.8      # syntax rate at T = 0.8
```

A gap smaller than the seed spread gets written as "no real difference."
