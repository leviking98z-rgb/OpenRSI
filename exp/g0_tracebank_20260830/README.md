# G0 Adaptive-Depth Trace Bank Plan

This experiment freezes the public `FrontisAI/Frontis-MA1-35B` checkpoint and
collects a reusable self-generated trace bank. The earlier Evo4 run is retained
only as a pipeline health check. It is too shallow to be the production data
run.

The paper's released evolutionary SFT rows were produced with a 64-execution
search limit. In the released 9,014 evolutionary rows, the selected operator is
at global execution position 17.3 on average and position 12 at the median.
About 61.8% occur by position 16, 82.5% by position 32, and 99.7% by position
64. These positions are search-tree execution indices, not parent-chain depth.

The production run therefore uses adaptive depth:

```text
Stage D16: 64 fresh tasks x first 16 executions       = 1,024
Stage D32: best 32 tasks x another 16 executions      =   512
Stage D64: best 16 tasks x another 32 executions      =   512
                                                        -----
Maximum raw operator executions in the first bank     = 2,048
```

If the first bank misses the train-ready yield gate, repeat the same protocol
once on a second, disjoint 64-task shard. Do not blindly deepen unproductive
tasks.

## Resource layout

- Eight nodes, each hosting one TP8 SGLang replica of the same frozen G0.
- Eight Search-Train tasks per node in D16.
- Four continuing tasks per node in D32.
- Two continuing tasks per node in D64.
- A task and its complete Program DB remain on the same node across stages.
- Task sharding, not seed replication, is the primary use of parallel compute.
- Promotion and Final Test tasks remain completely excluded, including
  competition/dataset-family aliases.

The production search profile is derived from the released evolutionary
profile rather than the Evo4 smoke profile:

```text
Draft, Improve, Debug, and Crossover enabled
individuals_per_generation = 5
crossover_prob = 0.5
num_generations_till_crossover = 2
max_debug_depth = 10
execution budgets = 16 -> 32 -> 64
```

The execution budget is the maximum number of generated-and-executed operator
calls for one task. It must not be described as literal parent-chain depth.

## Continuation gates

A D16 task may continue to D32 when at least one of the following holds:

- it produced an invalid-to-valid Debug transition;
- it produced a valid Improve child or another valid candidate with measurable
  score headroom;
- it has two distinct valid parents and is therefore Crossover-ready;
- its latest valid score is still improving rather than saturated.

A D32 task may continue to D64 when it has at least one strict
Improve/Crossover transition, two useful valid branches, or continued progress
in its most recent eight executions.

Stop deepening a task when it repeatedly produces the same deterministic
failure, only infrastructure failures, duplicate programs, or no useful parent
state. Infrastructure failures may be retried and are tagged separately;
model-generated failures, stderr, scoring failures, and timeouts remain in the
raw bank.

## Immutable raw bank

The raw bank preserves every attempted operator:

- `draft`
- `debug`
- `improve`
- `crossover`

Each row records task and shard identity, source checkpoint, operator,
parent/child IDs, complete prompts and responses, code, execution feedback,
score direction, validity, token counts, runtime, and content hashes. After
validation, the raw bank is immutable. All SFT recipes are deterministic
offline views over these executions.

## SFT views

| Recipe | Selection |
| --- | --- |
| `strict_only` | Invalid-to-valid Debug, valid Improve better than its parent, and valid Crossover better than its stronger parent |
| `verified_endpoint` | Best verified valid endpoints, materialized as Draft supervision |
| `causal_segment` | Score-qualified Draft/Improve/Crossover roots plus Debug descendants that are inherited by a valid endpoint |
| `main_g1` | Quality-capped union of current-bank Draft endpoints and selected non-Draft rows; no historical replay anchors |

Every selected row keeps an explicit selection tier such as `strict`,
`causal`, or `endpoint`. Non-strict rows are never relabeled as strict.
An external causal-inheritance judge is an optional offline view and is not a
blocking dependency for raw trace production.

The first-bank train-ready target is:

```text
400-800 total SFT rows
at least 32 contributing tasks
at least 128 non-Draft rows
at least 64 Debug rows
at least 32 Improve rows
at least 8 Crossover rows
```

These are yield gates, not quotas that permit weak examples. If a class misses
its target, run the second fresh task shard rather than admitting unverifiable
rows. Per-task caps prevent a few easy tasks from dominating. Draft is capped
at four unique-score endpoints per task, matching the paper's broad collection
principle; all useful non-Draft rows are retained subject to deduplication and
quality gates.

Every recipe:

- preserves task, shard, source checkpoint, operator, scores, and lineage in a
  manifest;
- re-scores selected endpoints before training;
- performs exact full-message and code deduplication, followed by near-duplicate
  reporting;
- rejects private-answer references, external-network behavior, empty fields,
  malformed role order, and mismatched parent edges;
- applies the G0 tokenizer's 32,768-token full-message limit;
- writes both JSONL and Parquet;
- retains all dropped-row reasons and content hashes.

## Decision boundary

This trace bank is intended to construct a credible G1 training set. It does
not itself prove improvement. Performance evidence still comes from a frozen,
held-out comparison:

```text
G0 vs G1
direct@1
fixed-state operator eval
fixed-budget Evo on Promotion
one-shot Final Test after promotion decision
```

The existing Evo4 pilot remains an auditable health-check artifact and is not
mixed into the production bank by default.
