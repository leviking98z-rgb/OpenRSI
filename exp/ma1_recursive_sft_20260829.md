# MA1 Recursive SFT Core Experiment

> Run dates: 2026-08-29 to 2026-08-30  
> Status: completed  
> Result: the update path worked end to end, but G1 and G2 were rejected  
> Formal champion after the experiment: G0

> Historical-scope note: this completed run included a token-matched SFT
> control. On 2026-08-30 that control was removed from the default critical
> path. Its artifacts and measurements below remain unchanged for auditability,
> but future generations train and evaluate only the candidate versus its
> parent.

## 1. Objective

This experiment tested a minimal weight-update loop while keeping
OpenMLE-Gym and OpenMLE-Evo fixed:

```text
frozen MA1
  -> fixed-budget Evo search
  -> verified Improve/Debug transitions
  -> continued SFT
  -> candidate checkpoint
  -> held-out promotion evaluation
```

The experiment deliberately used SFT rather than adding another RL/GSPO stage.
The intended core claim was:

```text
candidate MA1 + the same Evo budget
>
parent MA1 + the same Evo budget
```

That claim was not supported by this run.

## 2. Fixed protocol

| Item | Setting |
| --- | --- |
| Parent G0 | RL checkpoint inherited by G1 |
| Search harness | OpenMLE-Evo, unchanged |
| Environment and scorer | OpenMLE-Gym/sandbox, unchanged |
| Promotion tasks | 8 |
| Final Test tasks | 8 |
| Search budget | Evo4: four program executions per task |
| Generation limit | 12,288 output tokens |
| Eval concurrency | task/LLM/sandbox concurrency = 8 |
| Seed | `20260829` |
| SFT topology | TP2 / PP1 / CP1 / EP8 |
| SFT updates | 3 per trained candidate |
| Checkpoint selection | Final Test was not used for selection |

Correct G0 checkpoint:

```text
/data2/openrsi/experiments/full_loop_20260827/checkpoints/rl-frontis-qwen36-public-full-v033-g0-hf
```

An earlier Final Test accidentally served the pristine Qwen3.6 base checkpoint
as G0. That comparison is invalid and is excluded from all results below.

## 3. Core result 1: evaluation

### 3.1 Final Test: fixed G1 versus the correct G0

Both models completed all 8 tasks and all 32 program executions.

| Metric | G0 | G1 | G1 - G0 | Win/Tie/Loss | 95% paired bootstrap CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct@1 | 0.000000 | 0.000000 | 0.000000 | 0/8/0 | [0.000000, 0.000000] |
| Best@4 | 0.327075 | 0.125000 | **-0.202075** | **1/4/3** | [-0.625000, 0.250000] |
| Search AUC | 0.176382 | 0.031250 | **-0.145132** | 1/4/3 | [-0.344933, 0.031250] |
| Valid rate | 0.218750 | 0.062500 | **-0.156250** | 1/4/3 | [-0.406250, 0.093750] |

Raw validity counts:

```text
G0: 7 / 32 valid executions = 21.875%
G1: 2 / 32 valid executions =  6.250%
```

Decision:

```text
G1 final result: no improvement
formal champion: G0
```

The confidence interval is wide because this is an 8-task, one-seed core
experiment. The promotion decision nevertheless remains `reject` under the
fixed gate, especially because both Best@4 and valid rate regressed.

### 3.2 Promotion results

Promotion values below were recomputed with commit `20f0eb0`, using pooled raw
scores across parent, candidate and control for each task. Theoretical metric
bounds are used when available.

| Candidate comparison | Best@4 delta | Win/Tie/Loss | Valid-rate delta | Decision |
| --- | ---: | ---: | ---: | --- |
| G1 vs matched-concurrency G0 parent | -0.204352 | 3/1/4 | -0.125000 | reject |
| G1 vs token-matched SFT control | +0.092921 | 2/5/1 | 0.000000 | insufficient; CI crosses zero |
| G2 vs experimental G1 parent | -0.093165 | 1/4/3 | -0.093750 | reject |
| G2 vs G1 token-matched control | -0.000243 | 1/5/2 | -0.093750 | reject |

The G1-versus-control result suggests that the verified Evo rows may contain
more useful signal than matched replay alone. However, continued SFT as a whole
did not preserve the parent model's system performance.

### 3.3 Generational data yield

| Update attempt | Search tasks | Executions | Strict transitions | Training mixture | Outcome |
| --- | ---: | ---: | ---: | --- | --- |
| G0 -> G1 | 16 | 64 | 8: 6 Debug + 2 Improve | 8 Evo + 120 anchors | trained; rejected |
| G1 -> G2 | 8 | 32 | 1 Improve | 1 Evo + 127 anchors | trained; rejected |
| G2 -> G3 | 24 | 96 | 0 | none | stopped before SFT |

Strict-transition yield fell from `12.50%` to `3.13%` and then `0%`. This is
the clearest indication that verified experience production, rather than SFT
runtime, is the current bottleneck.

## 4. Core result 2: W&B training curves

Project:

<https://wandb.ai/leviking98z-zhejiang-university/openrsi-ma1-recursive-sft>

Runs:

| Run | W&B |
| --- | --- |
| G1 candidate | <https://wandb.ai/leviking98z-zhejiang-university/openrsi-ma1-recursive-sft/runs/ma1-g1-candidate-20260829-backfill> |
| G1 matched control | <https://wandb.ai/leviking98z-zhejiang-university/openrsi-ma1-recursive-sft/runs/ma1-g1-control-20260829-backfill> |
| G2 candidate | <https://wandb.ai/leviking98z-zhejiang-university/openrsi-ma1-recursive-sft/runs/ma1-g2-candidate-20260830-backfill> |

Important provenance note:

- The original launches had `use_wandb=False`.
- The W&B runs above were backfilled on 2026-08-30 from the immutable trainer
  logs; they are not live training telemetry.
- Each run contains only three optimizer updates. The curves are useful for
  checking execution, loss scale, gradient norm, throughput and step time, but
  are too short to establish convergence.

### 4.1 Logged points

#### G1 candidate

| Step | Train loss | Grad norm | Step time | Actor tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.383922 | 24.5718 | 289.1s | 5,063 |
| 1 | 0.383922 | 24.5708 | 104.9s | 13,600 |
| 2 | 0.389219 | 104.1497 | 118.0s | 12,088 |

#### G1 matched control

| Step | Train loss | Grad norm | Step time | Actor tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.403120 | 24.7660 | 202.4s | 7,416 |
| 1 | 0.403118 | 24.7456 | 121.9s | 11,708 |
| 2 | 0.404655 | 37.7850 | 102.0s | 13,986 |

#### G2 candidate

| Step | Train loss | Grad norm | Step time | Actor tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.397720 | 26.1376 | 296.7s | 5,609 |
| 1 | 0.397722 | 25.4831 | 126.0s | 12,838 |
| 2 | 0.383896 | 15.4415 | 120.8s | 13,366 |

## 5. Core result 3: efficiency

### 5.1 Evaluation efficiency smoke

The comparison preserved the task set and output budget.

| Variant | Work completed | Wall time | Generated tokens | Approx. tokens/s | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| TP8, concurrency 4 | 8 tasks x Evo1 | 307s | 74,494 | 242.7 | reject |
| TP8, concurrency 8 | 8 tasks x Evo1 | **234s** | 88,111 | **376.5** | use |
| DP2 x TP4, concurrency 8 | 5/8 tasks before abort | about 644s | 34,892 | about 54.2 | reject |

Moving from concurrency 4 to 8 reduced wall time by 23.8% and increased
aggregate token throughput by approximately 1.55x. DP2 x TP4 was substantially
slower for this long-context MoE workload.

Selected eval configuration:

```text
one TP8 server
task/LLM/sandbox concurrency = 8
max output tokens = 12,288
reuse one serving process across splits for the same checkpoint
```

### 5.2 SFT efficiency smoke

| Variant | First step | Warm/steady step | Observation |
| --- | ---: | ---: | --- |
| CPU optimizer offload + full recompute | 722.1s | not measured | reject |
| GPU optimizer + full recompute, cold compile | 719.5s | 131.5s | viable after warmup |
| GPU optimizer, no recompute | 754.5s | 268.1s | slower and much higher VRAM |
| GPU optimizer + full recompute + warm cache + 16K packing | **284.4s** | **111.8s** | use |

Selected SFT configuration:

```text
GPU optimizer
full uniform recompute
16,384 packed tokens per GPU
shared warm compile cache
do not save optimizer/RNG payload for this prototype
```

The optimized configuration reduced a representative first/warm-cache step
from roughly 722 seconds to 284-297 seconds. Formal steady steps were generally
about 102-126 seconds.

## 6. Interpretation

The experiment demonstrates:

1. The proposed SFT-only generational weight-update path is executable.
2. The current small-data recipe does not improve held-out system performance.
3. The formal champion must remain G0.
4. Experience yield degrades sharply across the experimental generations.
5. Training speed is acceptable after optimization; verified experience quality
   and valid-program production are the higher-priority problems.

A feedback-cleaning bug was found after the historical runs: sandbox stderr was
not included in the clear feedback passed to search, so Debug could receive only
`code_execution_error` rather than the traceback. The repository fix preserves
stderr for future runs. It does not alter or retroactively reinterpret the
results recorded here.

## 7. Artifacts and provenance

Node experiment root:

```text
/data2/openrsi/experiments/ma1_recursive_sft_20260829
```

Key result paths:

```text
eval/final-test-g1-vs-g0-rl-parent/
eval/g1-promotion-matched-c8-pooled/
eval/g2-promotion-pooled/
eval/g3-stop/
eval/recomputed-eval-20f0eb0.tar.gz
efficiency/SMOKE_REPORT.md
logs/sft-g1-candidate.log
logs/sft-g1-control.log
logs/sft-g2-candidate.log
```

Recomputed evaluation bundle:

```text
sha256:
7eae3eaf89cf891b7c1a45c09d65f247f5b263d243ebbb306d4fb6e89adf1af8
```

Repository:

```text
fork:   leviking98z-rgb/OpenRSI
branch: repro/h20-two-node-20260826
```

Relevant source commits:

```text
20f0eb0  pooled raw-score normalization for promotion
ea4ad81  stderr preservation and recursive experiment result documentation
```
