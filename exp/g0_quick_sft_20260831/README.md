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

## Round terminology

A `Round` is one candidate-promotion attempt:

```text
build one SFT recipe
-> initialize from the current accepted parent
-> train one candidate
-> run the frozen promotion evaluation
-> accept or reject
```

It is not a model generation. Both Round 1 and Round 2 below started from the
same public G0 checkpoint and produced alternative G1 candidates. Neither
candidate passed the promotion gate, so the accepted lineage is still G0.

## Round 1 result

Training:

```text
rows:                 656
tasks:                112
training tokens:      5,401,675
optimizer updates:    10
learning rate:        1.5e-5
initial checkpoint:   public G0
```

Frozen 8-task Evo4 promotion result:

| Metric | G0 | Round 1 candidate |
|---|---:|---:|
| direct@1 | 0.1250 | 0.0478 |
| best@4 | 0.4995 | 0.0481 |
| AUC | 0.3434 | 0.0479 |
| valid rate | 28.125% | 6.25% |

Decision: **Reject**.

The dominant immediate failure was training/evaluation environment mismatch.
The 656-row SFT set contained many examples using packages unavailable in the
frozen evaluator:

```text
rows mentioning LightGBM: 430
rows mentioning XGBoost:  344
rows mentioning CatBoost: 157
```

During promotion, 22 of 32 operator executions failed specifically because
CatBoost, LightGBM, or XGBoost was unavailable. The recipe also strongly
overweighted long Debug transitions (472 of 656 targets).

Canonical comparison:

```text
/root/shared/.clusters/.tmp/g0-quick-sft-results-20260831/
  comparison/promotion.json
```

## Round 2 result

Round 2 was a lower-update, package-filtered attempt, again initialized from
the public G0 rather than the rejected Round 1 candidate.

Training:

```text
rows:                 344
tasks:                117
training tokens:      1,108,169
optimizer updates:    5
learning rate:        4e-6 -> 4e-7
initial checkpoint:   public G0
```

Frozen 8-task Evo4 promotion result:

| Metric | G0 | Round 2 candidate | Delta |
|---|---:|---:|---:|
| direct@1 | 0.1250 | 0.0307 | -0.0944 |
| best@4 | 0.4995 | 0.1312 | -0.3683 |
| AUC | 0.3434 | 0.0676 | -0.2758 |
| valid rate | 28.125% | 21.875% | -6.25 pp |

Decision: **Reject**.

Round 2 removed CatBoost, LightGBM, and XGBoost from the SFT targets and
substantially recovered executability relative to Round 1 (7/32 valid operator
executions versus 2/32), but it still lost the high-quality solutions that made
G0 strong. Its AUC delta bootstrap 95% interval was entirely negative:

```text
[-0.5426, -0.0458]
```

The principal recipe problem is semantic rewriting. The source operators were:

```text
Debug:       249
Improve:      68
Draft:        21
Crossover:     6
```

All 344 examples were rewritten as:

```text
root Draft prompt -> code-only successful endpoint
```

Thus 323/344 targets originated from a non-Draft operator but were taught as
Draft. Parent code, execution feedback, and local-edit intent were discarded.
The recipe also retained up to four merely valid endpoints per task, rather
than only a clearly superior endpoint. This trains G0 to imitate its own
successful search products without providing a stronger teacher signal and
without preserving the interface used by Evo at evaluation time.

Artifacts:

```text
/data2/openrsi/experiments/g0_quick_sft_20260831/
  round2-verified-draft-20260831T041709Z/
    data/combined/
    checkpoints/g1-r2-verified-draft-hf/
    eval/g1-r2-vs-g0-promotion/comparison/promotion.json
```

## Failure diagnosis and stop decision

The observed regression is not explained by the recurring
`No user query found in messages` log alone. That auxiliary HTTP 400 condition
also occurs in the G0 baseline under the same harness. The differentiating
failures are:

1. Round 1 learned unavailable-package behavior from its SFT set.
2. Round 2 destroyed operator-conditioned semantics by converting Debug,
   Improve, and Crossover endpoints into root Draft demonstrations.
3. Both recipes perform self-distillation from G0 search outputs without a
   stronger teacher, margin target, preference objective, replay/KL
   constraint, or other signal that guarantees improvement over G0.
4. A small, correlated set of endpoints can move a strong full-parameter
   parent away from useful search diversity even when training loss falls.

Therefore no Round 3 using either existing recipe should be launched.
As of 2026-08-31:

- the accepted parent remains G0;
- H20 trace production has been stopped and its completed artifacts retained;
- the L20 node has no Round 3 job and all eight GPUs are idle;
- Round 1 and Round 2 data, checkpoints, logs, and evaluations are retained
  for diagnosis.

Any later SFT experiment should be treated as a new controlled ablation, not an
automatic continuation. At minimum it must preserve the true operator prompt
and parent/feedback context, use package-safe verified targets, and isolate the
weight-update magnitude before another full promotion run.
