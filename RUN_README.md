# Validated Frontis Single-Node Runbook

This document records the end-to-end Frontis/OpenRSI acceptance run completed
on **2026-08-28**. It is a reproducible engineering runbook and an artifact
index for the successful run.

The run completed the full executable training path:

```text
public assets
  -> full-parameter SFT updates
  -> torch-dist checkpoint
  -> Hugging Face conversion
  -> SGLang deployment
  -> OpenMLE rollout
  -> sandbox execution and scoring
  -> GSPO optimizer updates
  -> post-training weight synchronization
  -> terminal evaluation
  -> full-state checkpoint
  -> offline W&B logging
```

This proves the complete system loop on one 8-GPU node. It is **not** a claim
of reproducing the paper's final benchmark numbers or full training duration:
the unreleased RL corpus was replaced with an explicitly marked public
reconstruction, SFT was bounded to two optimizer steps, and RL ran one rollout
block with two optimizer updates.

## Source revision

The validated source started from this fork branch:

```text
repository: https://github.com/leviking98z-rgb/OpenRSI
branch:     repro/h20-two-node-20260826
base:       6111e008ef32979d212cefc191d2534bbe60f96f
```

The branch includes the following runtime-hardening changes:

| Commit | Purpose |
| --- | --- |
| `9a8d101` | Validate single-node launch and correctly handle resume state |
| `a70d1f7` | Make distributed initialization timeout configurable |
| `6278cfb` | Allow pageable CPU tensor backups to avoid exhausting pinned memory |
| `6111e00` | Opt-in terminal-checkpoint ordering fix and cursor recovery utility |

Reward shaping, Program Database semantics, Draft/Improve/Debug/Crossover
operators, and the GSPO loss were not changed by these commits.

## Validated hardware and software

| Component | Validated value |
| --- | --- |
| Hardware | 1 node, 8 × NVIDIA L20D |
| Container image | `openrsi/slime:v0.3.2-cu129-sglang0515-sm103fa2-20260828` |
| Image ID | `sha256:0951ce0ad9e73ad2fa6d365baf0893e517532561954ea1462da4a75bd4a0595d` |
| SLIME | `a3f500977f33b82de6ac0414c0c43abfe656c3e0` |
| SGLang | `0.5.15.post1` |
| PyTorch | `2.11.0+cu129` |
| Transformer Engine | `2.16.1` |
| FlashAttention | `2.8.3` |
| Ray | `2.58.0` |
| TE SM103 patch SHA-256 | `7314131e1c778a32dd450f56c43010fae4aeefbac87aaca9b30acbb41dc45173` |

SGLang was upgraded as part of the node-local image rather than keeping the
older incompatible serving stack. The successful run used a dedicated image
and did not depend on a pre-existing model service.

## Public assets and provenance

### Base model

```text
repo:     Qwen/Qwen3.6-35B-A3B
revision: 995ad96eacd98c81ed38be0c5b274b04031597b0
```

Verified model structure:

```text
layers:              40
hidden size:         2048
experts:             256
experts per token:   8
attention head dim:  256
vocabulary:          248320
```

### SFT corpus

```text
repo:     FrontisAI/OpenMLE-SFT-Traces
revision: dcb1d89f67c50660b2322efdb58f0769b0036395
rows:     26259
```

### RL corpus

The paper's original RL corpus and leaderboard were not publicly released.
This run therefore used a clearly labeled public reconstruction:

```text
task:             Titanic
training prompts: 16
evaluation rows:  1
leaderboard:      synthetic calibration grid, not paper data
```

Do not use this run's reward or evaluation values as paper-result
comparisons. They validate execution and optimization only.

## Single-node topology

The official Qwen3.6 profile is presented as a multi-node configuration. The
validated single-node mapping was:

| Plane | Topology |
| --- | --- |
| Training | TP=2, PP=1, CP=1, EP=8 |
| Rollout | 8 GPUs total, 4 GPUs per engine, therefore 2 × TP4 engines |
| Placement | Training and rollout colocated on the same 8 GPUs |
| SGLang DP/EP | DP=1, EP=1 |
| CUDA graphs | Disabled |
| Tensor backup | Pageable CPU memory (`SLIME_TENSOR_BACKUP_PIN_MEMORY=0`) |

This topology preserves the model, rollout geometry, optimizer geometry,
operator probabilities, and generation limits while adapting parallelism to a
single node.

## Validated training configuration

### SFT acceptance gate

The full-parameter SFT launcher used these material settings:

```text
model:                       Qwen3.6-35B-A3B
learning rate:               3e-5
global batch size:           128
micro batch size:            1
maximum context length:      32768
maximum tokens per GPU:      8192
TP / PP / CP / EP:           2 / 1 / 1 / 8
sequence parallel:           enabled
optimizer CPU offload:       enabled
overlap optimizer D2H/H2D:   enabled
MoE token dispatcher:        flex
DeepEP:                      enabled
optimizer state saved:       yes
RNG state saved:             yes
```

The acceptance gate intentionally stopped after two optimizer steps:

| Step | Loss | Gradient norm |
| ---: | ---: | ---: |
| 0 | `0.420363` | `10.3078` |
| 1 | `0.423678` | `22.0849` |

The resulting `iter_0000001` checkpoint was converted to a complete Hugging
Face directory with 1,045 indexed weights, 16 weight files, and
71,903,776,976 bytes.

### RL acceptance run

| Setting | Value |
| --- | ---: |
| Algorithm | GSPO |
| Rollout blocks | 1 |
| Prompts per block | 16 |
| Samples per prompt | 16 |
| Total generated samples | 256 |
| Maximum response length | 24,576 |
| Optimizer updates | 2 |
| Global batch size | 128 |
| Learning rate | `1e-6` |
| Evaluation interval | 1 |
| Evaluation maximum response length | 32,768 |
| Draft probability | 0.50 |
| Improve probability | 0.17 |
| Debug probability | 0.17 |
| Crossover probability | 0.16 |

The realized random operator counts were:

```text
Draft:      192
Debug:       32
Improve:     16
Crossover:   16
```

## Node-local acceptance layout

The successful node reused large assets already present under `/data2`; no
multi-hundred-gigabyte model or checkpoint transfer was required.

```text
/data2/openrsi/experiments/full_loop_20260827/
├── assets/
│   ├── datasets/OpenMLE-SFT-Traces-dcb1d89/
│   └── models/Qwen3.6-35B-A3B-995ad96/
├── checkpoints/
│   ├── Qwen3.6-35B-A3B-995ad96-torch-dist/
│   ├── sft-frontis-qwen36-35b-paper-gate3/
│   ├── sft-frontis-qwen36-35b-paper-gate3-hf/
│   └── rl-frontis-qwen36-35b-public-full-v033/
├── configs/
│   └── rl_frontis_public_full_v033.env
├── data/rl_public_reconstruction/
├── logs/
│   └── rl_frontis_qwen36_public_full_v033.log
├── scripts/
│   ├── launch_frontis_sft_gate3.sh
│   ├── convert_sft_gate3_to_hf.sh
│   └── run_rl_frontis_v033_container.sh
├── artifacts/rl-frontis-qwen36-public-full-v033/
└── wandb/rl-frontis-qwen36-35b-public-full-v033/
```

These exact paths describe the validated node. For another machine, preserve
the relationships but replace the experiment root and asset paths.

## Launch sequence

The node-local wrappers pin image/source hashes, refuse ambiguous overwrites,
check GPU idleness, and validate required ports before launch.

For a **new output namespace**, the sequence is:

```bash
export EXP=/data2/openrsi/experiments/full_loop_20260827

# 1. Full-parameter SFT acceptance gate.
bash "${EXP}/scripts/launch_frontis_sft_gate3.sh"

# 2. Convert the completed torch-dist SFT checkpoint to Hugging Face format.
bash "${EXP}/scripts/convert_sft_gate3_to_hf.sh"

# 3. Validate the complete RL configuration without starting Ray or training.
bash "${EXP}/scripts/run_rl_frontis_v033_container.sh" full precheck

# 4. Run the 256-sample rollout and two GSPO optimizer updates.
bash "${EXP}/scripts/run_rl_frontis_v033_container.sh" full run
```

The validated wrappers intentionally refuse to overwrite existing checkpoints
or reuse an existing container name. Change the output namespace before
repeating a run; do not delete successful artifacts merely to make a launcher
pass.

For a portable installation inside a pinned SLIME checkout, use the repository
launcher:

```bash
cp examples/openmle_rl/configs/sync_single_node.env.example \
  /absolute/path/frontis-single-node.env

# Edit all model, data, service, output, tracking, and topology values first.
PRECHECK_ONLY=1 \
  bash examples/openmle_rl/scripts/run_openmle_rl_sync_single_node.sh \
  /absolute/path/frontis-single-node.env

bash examples/openmle_rl/scripts/run_openmle_rl_sync_single_node.sh \
  /absolute/path/frontis-single-node.env
```

For Qwen3.6 on the validated 8-GPU topology, the portable configuration must
include:

```bash
MODEL_CONFIG="${OPS_DIR}/model_configs/qwen3.6-35B-A3B.sh"

ROLLOUT_BATCH_SIZE=16
N_SAMPLES_PER_PROMPT=16
NUM_STEPS_PER_ROLLOUT=2
GLOBAL_BATCH_SIZE=128
ROLLOUT_MAX_RESPONSE_LEN=24576

TENSOR_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
EXPERT_MODEL_PARALLEL_SIZE=8
ROLLOUT_NUM_GPUS=8
ROLLOUT_NUM_GPUS_PER_ENGINE=4
SGLANG_MEM_FRACTION_STATIC=0.70
SGLANG_ENABLE_DP_ATTENTION=0
SGLANG_DP_SIZE=1
SGLANG_EP_SIZE=1
SGLANG_ENABLE_DP_LM_HEAD=0
SGLANG_MOE_RUNNER_BACKEND=flashinfer_cutlass
SGLANG_DISABLE_CUDA_GRAPH=1
ATTENTION_BACKEND=flash

DRAFT_PROBABILITY=0.50
IMPROVE_PROBABILITY=0.17
DEBUG_PROBABILITY=0.17
CROSSOVER_PROBABILITY=0.16

SLIME_TENSOR_BACKUP_PIN_MEMORY=0
OPENRSI_DEFER_FINAL_CHECKPOINT=1
DISTRIBUTED_TIMEOUT_MINUTES=60
WANDB_MODE=offline
```

Service credentials belong in an untracked configuration file or secret
mount. Never commit API keys, tokens, `.netrc`, or live service credentials.

## Service and port contract

The validated run used:

```text
127.0.0.1:6580   sandbox controller
127.0.0.1:18080  judge / hack-check endpoint
18081            must remain unused
18082            must remain unused
```

The SFT/RL launchers should fail closed if protected ports are occupied. Do not
start an unrelated model server on 18081 or 18082 as part of this runbook.

## Acceptance results

The successful RL container was:

```text
name:     openrsi-rl-frontis-qwen36-public-full-v033
state:    exited
exit:     0
OOM:      false
Ray:      Job succeeded
```

Observed end-to-end evidence:

- Initial Megatron-to-SGLang synchronization completed `132/132` tensors.
- Rollout completed `256/256` samples.
- All 16 groups reached `16/16` samples.
- Native rollout data and a Program Database snapshot were saved.
- Reference log-probability computation completed `31/31` batches.
- Two real optimizer updates completed:

| Step | Loss | Gradient norm |
| ---: | ---: | ---: |
| 0 | `-0.2345628` | `0.518145` |
| 1 | `-0.2002680` | `0.611230` |

- Post-training weight synchronization completed `132/132` tensors.
- Terminal evaluation reached the judge and sandbox over HTTP.
- The full-state checkpoint completed atomically.
- W&B data was written in offline mode.

Direct deserialization of the rollout artifact showed:

```text
samples:                 256
groups:                  16
mean response length:    8769.22 tokens
valid responses:         221
empty responses:         12
hack-rejected responses: 23
actually scored programs: 23
raw reward range:        -1.0 to 1.5
```

Program Database counts:

```text
program_database_iter_0.db: 248 rows
program_database_ops.db:    249 rows, including terminal evaluation
```

The database started from four seed programs and contains actual
Draft/Improve/Debug/Crossover outputs.

## Terminal evaluation interpretation

The terminal evaluation infrastructure succeeded:

```text
HTTP status:          200
sandbox job:          completed
code category:        valid
hack flag:            0
```

The generated candidate itself deleted the `Fare` column and later attempted
to read it, producing:

```text
KeyError: 'Fare'
```

That candidate therefore received `score=None` and reward `0`. This is a model
sample failure, not a serving, sandbox, judge, or reward-pipeline failure. The
training rollout independently contains 23 programs with real execution
scores, which verifies the execution-to-reward-to-optimizer path.

## Checkpoint and artifact verification

The RL checkpoint is:

```text
/data2/openrsi/experiments/full_loop_20260827/checkpoints/
  rl-frontis-qwen36-35b-public-full-v033/
  OPS_openrsi_frontis_qwen36_public_full_v033-temp1-draft0.50-debug0.17-20260828065030/
```

Final read-only verification found:

```text
distcp shards:                   16
actor checkpoint directory:     iter_0000000
latest iteration marker:        1
.metadata:                      present
common.pt:                      present
total bytes including metadata: 497843706573
```

Checkpoint metadata contains model state, optimizer state including Adam
`exp_avg` and `exp_avg_sq`, optimizer scheduler state, and per-rank RNG state.
It is a full training-state checkpoint rather than a weights-only export.

Useful non-mutating checks:

```bash
export EXP=/data2/openrsi/experiments/full_loop_20260827
export CKPT="${EXP}/checkpoints/rl-frontis-qwen36-35b-public-full-v033/OPS_openrsi_frontis_qwen36_public_full_v033-temp1-draft0.50-debug0.17-20260828065030"

docker inspect \
  --format 'status={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' \
  openrsi-rl-frontis-qwen36-public-full-v033

find "${CKPT}" -type f -name '*.distcp' | wc -l
find "${CKPT}" -type f \( -name '.metadata' -o -name 'common.pt' \)
du -sb "${CKPT}"

nvidia-smi \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

ss -ltn | grep -E ':(6580|18080|18081|18082)[[:space:]]'
```

The offline W&B run is stored under:

```text
/data2/openrsi/experiments/full_loop_20260827/wandb/
  rl-frontis-qwen36-35b-public-full-v033/
```

Its run ID is `of5cajj1`, and the final training history includes
`train/step=1`, `train/loss=-0.200268`, and
`train/grad_norm=0.611230`.

## Reproduction boundary

The following parts match the public Frontis configuration:

- Qwen3.6-35B-A3B model family and pinned public revision
- released OpenMLE SFT traces
- full-parameter SFT path
- GSPO
- 16 prompts × 16 samples
- 24,576-token rollout cap
- two optimizer steps per rollout block
- Draft/Improve/Debug/Crossover probability mix
- Program Database, sandbox execution, scoring, reward, evaluation, and
  checkpoint paths

The following parts differ from a full paper reproduction:

- one 8-GPU single-node topology instead of the official multi-node topology
- a public Titanic RL reconstruction instead of the unreleased RL corpus
- a synthetic calibration leaderboard instead of paper leaderboard assets
- two SFT optimizer steps instead of the full SFT schedule
- one RL rollout block instead of long-duration RL training
- offline W&B rather than an online experiment run

Accordingly, this run is a **complete engineering-loop validation under public
conditions**, not a final paper-quality or benchmark-equivalence result.
