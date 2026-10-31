# Decisions

Append-only. One line per decision. Big ones get an ADR in docs/adr/ and a link here. To change a decision, add a new line that supersedes the old one.

| Date | Decision | Why | Decided by | ADR |
|---|---|---|---|---|
| 2026-10-30 | Record architecture decisions as ADRs | Keep the reasoning next to the code | Lawrence | 0001 |
| 2026-10-30 | Byte-level vocab of 256, no trained tokenizer | Nothing to train, round trip is exact | Lawrence | - |
| 2026-10-30 | Hash a normalized copy of each file for dedupe and the train/val split | A file always lands on one side, whitespace-only copies collapse | Lawrence | - |
| 2026-10-30 | Keep the remaining ~8% train/val leakage as a known caveat for now | Fixing it needs MinHash, more machinery than a first baseline needs | Lawrence | - |
| 2026-10-30 | Private repo, All rights reserved | Course work in progress | Lawrence | - |
