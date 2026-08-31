# G0 Quick SFT Snapshot — 2026-08-31

This directory records the first training-ready aggregation of G0 Evo D16
traces. The actual dataset is stored on the L20 node; large generated data is
not committed to Git.

## Recipe

- Keep strict parent-to-child transitions:
  - `Debug`: invalid parent -> valid child.
  - `Improve`: valid child strictly beats its valid parent.
  - `Crossover`: valid child strictly beats the stronger of two valid parents.
- For each task with a verified valid result, rewrite its best endpoint as a
  `Draft` target using the root task prompt.
- Do not add historical anchor/replay examples.
- Keep only complete D16 traces (16 operator executions).
- Deduplicate by SHA-256 of the full `messages` value.
- Enforce a 32,768-token maximum using the Frontis MA1 tokenizer during source
  recipe construction.

The endpoint examples reuse the score/status verified during the original
sandbox execution. This quick snapshot does not independently rerun and
rescore every endpoint.

## Frozen inputs

Snapshot timestamp:

```text
20260831T1047
```

| Source | Complete D16 task manifests | Tasks contributing SFT | SFT rows |
|---|---:|---:|---:|
| fixed-d16 | 62 | 51 | 259 |
| l20-continuous | 62 | 59 | 384 |
| h20-continuous | 2 | 2 | 13 |
| **Combined** | **126** | **112** | **656** |

The two incomplete tasks from the original fixed 64-task batch were excluded:

```text
chembl22@1                                      13/16
building-sites-power-consumption-dataset@2     14/16
```

## Combined statistics

```text
Rows:                       656
Unique task names:          112
Strict transitions:         544
Verified endpoints/Draft:   112
Exact duplicate drops:      0

Operator targets:
  Draft:                    112
  Debug:                    472
  Improve:                   69
  Crossover:                  3

Training tokens:
  total:              5,401,675
  mean:                 8,234.26
  median:               8,292
  p90:                 12,165.5
  p95:                 13,431.25
  max:                 16,364
```

Validation passed:

- 656 unique record IDs;
- 656 unique full-message hashes;
- every row has exactly `system`, `user`, `assistant`;
- all message contents are non-empty;
- every row is within the 32,768-token gate;
- JSONL and Parquet contain identical rows;
- manifest and training row counts agree.

## L20 paths

Root:

```text
/data2/openrsi/experiments/g0_quick_sft_20260831/snapshot-20260831T1047
```

Training-ready outputs:

```text
combined/train.parquet
combined/train.jsonl
combined/manifest.jsonl
combined/tasks.jsonl
combined/summary.json
combined/source_snapshots.json
```

Recommended SFT input:

```bash
DATA_PATH=/data2/openrsi/experiments/g0_quick_sft_20260831/snapshot-20260831T1047/combined/train.parquet
INPUT_KEY=messages
```

Artifact hashes:

```text
b5cac07a9e6cbbc9d29b0c43b18bd3d186c72825ec4208c3f2682dab0d6e2536  train.jsonl
ea317a2ca401479f9735725ca2c5b4ced914098d259048c901db11962bfef651  train.parquet
0abaa9f3c1fe42eb5e33f533956375cb13e6e4ff707b128e1ea1f58cf666a4d6  manifest.jsonl
33b5a6cdd46339f3cd08b7e8a176ff3fa68d044e6b1d34621310a437f4548c73  tasks.jsonl
e0b49b53113a3d76f47a5dcb01a943d844e12e474791b9a0f67d47fe5bb57af7  summary.json (current generated snapshot)
```

Continuous H20/L20 trace collection remained running after the snapshot was
frozen, so later data can be aggregated as a separate version without changing
this training input.
