# G0 Continuous D16 Trace Collection

This experiment extends the fixed 64-task D16 trace-bank run into a continuously
refilled, globally unique Search-Train collection. It uses the same frozen public
MA1 checkpoint and the same D16 search policy. It does **not** add seeds: every
task is assigned at most once with `sample_index=0`.

## Frozen protocol

```text
checkpoint: FrontisAI/Frontis-MA1-35B
revision: 79a29e43e7f96b96b06eaf24dcc885ad0318aa01
sample_index: 0
operator execution budget: 16
operators: Draft / Debug / Improve / Crossover
individuals_per_generation: 5
num_generations: 100
crossover_prob: 0.5
num_generations_till_crossover: 2
max_debug_depth: 10
max output tokens: 12,288
```

A shared atomic-claim ledger gives each slot a new task only after its previous
one reaches a terminal state. The H20 and L20 inventories are disjoint. Promotion,
Final Test, and source-family aliases of held-out tasks are excluded.

Inventory frozen on August 31, 2026:

```text
eligible unique Search-Train tasks: 928
H20 pool: 825
L20 pool: 103
overlap between pools: 0
```

The first 64 tasks from the fixed D16 experiment are also excluded from the
continuous inventories.

## Runtime layout

Shared H20 orchestration and ledger:

```text
/root/shared/.clusters/.tmp/openrsi-continuous-20260831/
/root/shared/.clusters/.tmp/openrsi-continuous-20260831/h20-state/
```

H20 node-local work and immutable per-task archives:

```text
/root/.cache/openrsi/experiments/g0_unique_continuous_20260831/
/root/sync/openrsi/g0_unique_continuous_20260831/raw/
```

Original 8xL20D node:

```text
/data2/openrsi/experiments/g0_unique_continuous_20260831/
```

The original node reuses its healthy TP8 SGLang server on port `30010` and
sandbox on port `6583`; its worker runs in the validated Slime image because the
host rollout environment is incomplete.

## Collector invariants

`continuous_worker.py` enforces:

- atomic `mkdir` task claims;
- one fixed `sample_index=0` attempt per task;
- immediate slot refill after completion or failure;
- separate `claims/`, `done/`, `failed/`, and `heartbeats/` ledgers;
- one lock-holding top-level worker per node-local work root;
- complete task config, runner config, search trace, score, timing, and token data;
- one `.tar.zst` archive and SHA-256 sidecar per terminal attempt.

Infrastructure failures that occur before search may be released for the **same**
task retry only when an audit confirms that no `search_events.jsonl` exists. This
is not a new seed. Model, execution, scoring, or task-format failures remain
terminal records.

## Validation

`validate_completed_task.py` checks a completed task for:

- exactly 16 contiguous operator executions;
- `stat.json`, `search_events.jsonl`, and `search_state.json`;
- complete and prior-only parent references;
- event/program-node consistency;
- score/status/runtime/token fields;
- `sample_index=0` and frozen D16 metadata;
- archive existence and SHA-256 agreement.

The first completed batch on the L20 node passed this validator. At the
25-task checkpoint on August 31, 2026, all 25 completed traces passed:

```text
operator executions: 400
Draft / Debug / Improve / Crossover: 111 / 220 / 63 / 6
prompt tokens: 1,288,144
completion tokens: 2,375,006
scored executions: 162
end-to-end task time: median 13.6 min, mean 15.7 min
```

The dispatcher refilled every released slot and kept eight attempts active,
demonstrating dynamic scheduling rather than a one-shot static shard.

One large OCR task was retained as a terminal task-format failure because the
published package did not contain the required tabular `train.csv` and
`test.csv`. Earlier transient failures caused solely by a missing `unzstd`
executable in the worker container were preserved in an infrastructure-recovery
ledger and safely retried with the same task and seed after installing the small
`zstd_python` shim.

## Files

- `continuous_worker.py`: atomic dispatcher, materializer, runner, archiver.
- `h20_transition_to_continuous.sh`: switch one H20 shard after fixed-D16 archive.
- `launch_l20_continuous.sh`: launch the L20 Docker worker.
- `zstd_python`: minimal Python-zstandard CLI compatibility inside that image; mounted as both `zstd` and `unzstd`.
- `recover_l20_unzstd_failures.py`: release only proven pre-search infra failures.
- `validate_completed_task.py`: per-task D16 lineage/artifact validator.
- `h20_inventory.jsonl`, `l20_inventory.jsonl`: disjoint frozen task pools.
- `inventory_summary.json`: selection and exclusion counts.

Collection remains active until explicitly stopped. Raw traces are not yet an
SFT recipe and are not evidence that G1 improves over G0; that conclusion still
requires offline recipe construction, continued SFT, and held-out evaluation.
