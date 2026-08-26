# OpenMLE-ERL: Supervised Fine-Tuning — Usage Guide

This directory contains the rollout, data-selection, and full-parameter SFT code used to train the OpenMLE models. It is a source release, not a collection of paper-only pseudocode: the launchers are adapted from the production rollout and SLIME runs, with cluster paths, credentials, run artifacts, and internal dataset aliases removed.

The released pipeline builds a 26,574-example training corpus:

| Source | Examples |
| --- | ---: |
| Parallel full responses | 17,344 |
| Evolutionary trajectory steps | 9,230 |
| Total | 26,574 |

The paper configuration uses BF16 full-parameter training on 8 NVIDIA H200 GPUs, a global batch size of 128, a `3e-5` peak learning rate with cosine decay, a 0.1 warmup fraction, three epochs, and a 32,768-token context limit.

## Repository layout

```text
.
├── scripts/
│   ├── evaluate_pass_k_glm-47_4-valid.sh
│   ├── evaluate_airaevo.py
│   ├── run_evolutionary_rollout.sh
│   └── sft_data_selection/
├── tts_search/                 # Parallel and evolutionary rollout runtime
├── slime_scripts/
│   ├── qwen3_30b/train.sh
│   ├── qwen3_6_35b/train.sh
│   └── common/run_slime_sft.sh
├── slime/                      # Vendored SLIME source used by the launchers
└── third_party/aira-evo/       # Evolutionary-search runtime and MLE bridge
```

Only one representative configuration is included for each rollout path. Generated samples, checkpoints, logs, W&B state, and the exact final sample list are intentionally excluded.

## Prerequisites

- Linux with Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)
- A CUDA environment compatible with the selected rollout or training profile
- The existing SLIME training image for full SFT runs
- A compatible sandbox evaluation service for rollout scoring
- Eight H200-class GPUs for the paper training configuration
- A converted Megatron `torch_dist` checkpoint for SFT

Use uv for rollout, data selection, lockfile validation, and source-level development:

```bash
# Rollout and data selection
UV_PROJECT_ENVIRONMENT=.venv-rollout \
  uv sync --locked --extra rollout
source .venv-rollout/bin/activate
```

The two training extras are kept separate so their Python dependency graphs can be resolved and inspected independently:

```bash
UV_PROJECT_ENVIRONMENT=.venv-qwen3-30b \
  uv sync --locked --extra train-qwen3-30b

UV_PROJECT_ENVIRONMENT=.venv-qwen3-6-35b \
  uv sync --locked --extra train-qwen3-6-35b
```

Do not install both training extras into the same environment. These uv extras do not replace the compiled Megatron, Transformer Engine, FlashAttention, and NCCL stack used for training. Run the full SFT launchers in a SLIME image compatible with the vendored `slime/` source. This repository intentionally does not add or maintain a separate Dockerfile.

## Paths and credentials

Commands use environment variables instead of machine-specific defaults. Values in angle brackets are placeholders that must be replaced:

| Placeholder or variable | Meaning |
| --- | --- |
| `<REPO_ROOT>` | Local checkout of this SFT source release |
| `<DATA_ROOT>` | Directory containing user-provided rollout or SFT data |
| `<MODEL_ROOT>` | Directory containing Hugging Face and converted model weights |
| `<MODEL_PATH>` | Hugging Face model directory |
| `<REF_LOAD_PATH>` | Converted Megatron `torch_dist` checkpoint |
| `<OUTPUT_DIR>` | New directory for run artifacts or checkpoints |
| `<OUTPUT_ROOT>` | Writable host directory mounted for training outputs |
| `<SANDBOX_GPU_URL>` | Base URL of a compatible GPU sandbox service |
| `<SANDBOX_CPU_URL>` | Optional CPU sandbox base URL |
| `ZAI_API_KEY` | GLM API credential |
| `ZAI_API_BASE` | Optional OpenAI-compatible GLM endpoint |
| `SANDBOX_GPU_API_KEY` | GPU sandbox credential |
| `SANDBOX_CPU_API_KEY` | Optional CPU sandbox credential |
| `OPENMLE_STORAGE_ROOT` | Runtime cache and temporary-file root |

Credentials are read only from the environment. The repository contains no default API keys.

## Input formats

Parallel rollout reads Parquet or JSON records with `prompt` and `metadata` fields. The metadata must provide the task identity and the task data required by the configured sandbox evaluator.

Training data must contain an `id` and a `messages` sequence:

```json
{
  "id": "example-0001",
  "messages": [
    {"role": "system", "content": "You are an ML agent."},
    {"role": "user", "content": "Task Description:\n..."},
    {"role": "assistant", "content": "<think>\n...\n</think>\n\n..."}
  ]
}
```

The data-selection commands accept JSONL. Commands that preserve a training table also accept Parquet when `pyarrow` is installed.

## Generate rollouts

### Parallel GLM-4.7 path

Set the model, task data, sandbox, and output locations:

```bash
export ZAI_API_KEY="<GLM_API_KEY>"
export ZAI_API_BASE="<OPENAI_COMPATIBLE_BASE_URL>"
export OPENMLE_PARALLEL_DATA="<DATA_ROOT>/parallel_tasks.parquet"
export OPENMLE_SANDBOX_GPU_URL="<SANDBOX_GPU_URL>"
export SANDBOX_GPU_API_KEY="<SANDBOX_API_KEY>"
export OPENMLE_OUTPUT_DIR="<OUTPUT_DIR>/parallel"
export OPENMLE_STORAGE_ROOT="<OUTPUT_DIR>/runtime"

bash scripts/evaluate_pass_k_glm-47_4-valid.sh
```

The second positional argument selects `full`, `gen_only`, or `eval_only`. Resume a prior run with `RESUME_FROM="<OUTPUT_DIR>/parallel"` or supply the existing run as the third argument in `eval_only` mode. Hydra overrides may be appended after the first three positional arguments.

The run directory contains the resolved configuration, generation and evaluation JSONL files, progress state, task summaries, and per-step artifacts.

### Evolutionary GLM-4.7 path

The evolutionary launcher converts the public Hydra configuration into the vendored AIRA-Evo/Dojo runner format:

```bash
export ZAI_API_KEY="<GLM_API_KEY>"
export ZAI_API_BASE="<OPENAI_COMPATIBLE_BASE_URL>"
export OPENMLE_EVOLUTIONARY_DATA="<DATA_ROOT>/evolutionary_tasks.parquet"
export OPENMLE_LEADERBOARD_DIR="<DATA_ROOT>/leaderboards"
export OPENMLE_TASK_DATA_ROOT="<DATA_ROOT>/task_data"
export OPENMLE_TOKENIZER_MODEL="<MODEL_PATH>"
export OPENMLE_SANDBOX_GPU_URL="<SANDBOX_GPU_URL>"
export OPENMLE_SANDBOX_CPU_URL="<SANDBOX_CPU_URL>"
export SANDBOX_GPU_API_KEY="<SANDBOX_API_KEY>"
export OPENMLE_OUTPUT_DIR="<OUTPUT_DIR>/evolutionary"
export OPENMLE_STORAGE_ROOT="<OUTPUT_DIR>/runtime"

bash scripts/run_evolutionary_rollout.sh
```

The representative configuration uses 64 search steps, five candidates per generation, Draft/Improve/Crossover operators, and Debug descendants. Override Hydra values after the optional config name, for example:

```bash
bash scripts/run_evolutionary_rollout.sh \
  experiment/evolutionary_glm47 \
  max_steps=32 \
  search.runner.task_concurrency=16
```

Use the same `OPENMLE_OUTPUT_DIR` to resume an interrupted evolutionary run.

## Select SFT data

### Parallel full responses

The first collection batch removes duplicate scores within each task before retaining the highest four GLM candidates:

```bash
python scripts/sft_data_selection/select_parallel.py \
  --input "<DATA_ROOT>/parallel_candidates.jsonl" \
  --output "<OUTPUT_DIR>/parallel_first_batch.jsonl" \
  --policy glm-top4-unique-score
```

The second batch ranks GLM and Qwen candidates together. It inspects only the joint Top-4, keeps GLM candidates in those positions, and keeps a Qwen candidate only when it is joint rank 1:

```bash
python scripts/sft_data_selection/select_parallel.py \
  --input "<DATA_ROOT>/mixed_model_candidates.jsonl" \
  --output "<OUTPUT_DIR>/parallel_second_batch.jsonl" \
  --policy joint-top4
```

This rule does not select GLM Top-4 and then add Qwen Top-1. A Qwen candidate at joint rank 2-4 is removed, and the candidate at rank 5 is not used to fill the empty position. Each task therefore contributes at most four responses.

### Evolutionary trajectory steps

```bash
python scripts/sft_data_selection/select_evolutionary.py \
  --stat-root "<DATA_ROOT>/evolutionary_run/program_ep_0" \
  --annotations "<DATA_ROOT>/causal_inheritance_annotations.jsonl" \
  --output-dir "<OUTPUT_DIR>/evolutionary_selection" \
  --require-complete-annotations
```

A segment starts at Draft, Improve, or Crossover and follows its Debug descendants. Draft endpoints require a positive score; Improve endpoints must beat their parent; Crossover endpoints must beat the stronger parent. The endpoint must also reach bronze, silver, or gold.

Single-step segments passing these gates are retained directly. Multi-step segments use causal-inheritance annotations. The corresponding system prompt is `scripts/sft_data_selection/causal_inheritance_prompt.txt`.

### Final gates

Deduplicate complete `messages` values, keeping the first occurrence:

```bash
python scripts/sft_data_selection/finalize_messages.py dedup \
  --input "<DATA_ROOT>/selected.jsonl" \
  --output "<OUTPUT_DIR>/deduplicated.jsonl"
```

Apply the target model's chat template and the 32,768-token limit:

```bash
python scripts/sft_data_selection/finalize_messages.py token-filter \
  --input "<OUTPUT_DIR>/deduplicated.jsonl" \
  --output "<OUTPUT_DIR>/train.jsonl" \
  --model-path "<MODEL_PATH>" \
  --max-tokens 32768
```

To remove task-description overlap with a reserved task set:

```bash
python scripts/sft_data_selection/exclude_reserved_tasks.py \
  --input "<DATA_ROOT>/train.parquet" \
  --reserved-prompts "<DATA_ROOT>/reserved_prompts.parquet" \
  --output-dir "<OUTPUT_DIR>/train_without_reserved_tasks"
```

This command extracts `Task Description:` from the user message, normalizes line endings and trailing whitespace, and compares exact SHA256 hashes. Missing task descriptions fail closed by default. Task IDs and names are audit fields, not matching keys.

## Train

Run training inside the existing SLIME image, following the upstream SLIME container setup. The following command mounts this release, the user-provided data and model roots, and a writable output root:

```bash
export SLIME_IMAGE="<PINNED_SLIME_IMAGE_TAG_OR_DIGEST>"
docker pull "$SLIME_IMAGE"

docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "<REPO_ROOT>:/workspace/openmle-sft" \
  -v "<DATA_ROOT>:/workspace/data:ro" \
  -v "<MODEL_ROOT>:/workspace/models:ro" \
  -v "<OUTPUT_ROOT>:/workspace/output" \
  -w /workspace/openmle-sft \
  -it "$SLIME_IMAGE" /bin/bash
```

Set `SLIME_IMAGE` to a reviewed immutable tag or digest for the target CUDA runtime. The commands below are run inside the container. Point the launcher at the mounted data, Hugging Face model files, converted Megatron weights, and a new output directory.

### Qwen3-30B

```bash
export DATA_PATH="/workspace/data/train.parquet"
export MODEL_PATH="/workspace/models/<HF_MODEL_DIR>"
export REF_LOAD_PATH="/workspace/models/<TORCH_DIST_DIR>"
export OUTPUT_DIR="/workspace/output/qwen3-30b"

bash slime_scripts/qwen3_30b/train.sh
```

### Qwen3.6-35B

```bash
export DATA_PATH="/workspace/data/train.parquet"
export MODEL_PATH="/workspace/models/<HF_MODEL_DIR>"
export REF_LOAD_PATH="/workspace/models/<TORCH_DIST_DIR>"
export OUTPUT_DIR="/workspace/output/qwen3-6-35b"

bash slime_scripts/qwen3_6_35b/train.sh
```

Both launchers derive `SLIME_ROOT` from the repository and run the vendored source. Qwen3.6 uses the compatible `qwen3_5` SLIME model specification shipped in this directory.

The launchers derive the number of training steps from the dataset row count, global batch size, and epoch count. They save only the final checkpoint by default. Common overrides include:

```bash
export NUM_EPOCH=3
export LR=3e-5
export GLOBAL_BATCH_SIZE=128
export ROLLOUT_BATCH_SIZE=128
export ROLLOUT_MAX_CONTEXT_LEN=32768
export LOG_PROBS_CHUNK_SIZE=4096
export ACTOR_NUM_NODES=1
export ACTOR_NUM_GPUS_PER_NODE=8
```

`LOG_PROBS_CHUNK_SIZE` is optional and disabled by default. A positive value
chunks the sequence axis of the vocabulary-parallel log-probability
calculation without dropping tokens. This bounds the transient FP32
cross-entropy buffer for long 32k examples on GPUs with less peak memory than
the H200 paper configuration.

When validating outside the pinned SLIME image, a host PyTorch build may spawn too many Inductor workers while compiling the first MoE step. If the GPUs are idle and many `torch._inductor.compile_worker` processes appear, set `TORCHINDUCTOR_COMPILE_THREADS=1`. The launcher forwards this setting to Ray workers only when it is explicitly set; the default training configuration is otherwise unchanged.

W&B is disabled by default. To enable it, set `USE_WANDB=1`, `WANDB_API_KEY`, `WANDB_PROJECT`, and `WANDB_GROUP`.

Before allocating GPUs, validate paths, dataset size, model profiles, and Ray ports with:

```bash
CHECK_CONFIG_ONLY=1 bash slime_scripts/qwen3_30b/train.sh
CHECK_CONFIG_ONLY=1 bash slime_scripts/qwen3_6_35b/train.sh
```

### External multi-node Ray

The common SFT launcher can submit training to a Ray cluster that was started
outside the driver process. Start one persistent process on each node:

```bash
# Head node.
MASTER_ADDR=10.0.0.10 NODE_IP=10.0.0.10 \
  bash slime_scripts/common/start_external_ray_node.sh head

# Worker node.
MASTER_ADDR=10.0.0.10 NODE_IP=10.0.0.11 \
  bash slime_scripts/common/start_external_ray_node.sh worker
```

Then invoke the model launcher on the head node:

```bash
export RAY_CLUSTER_MODE=external
export MASTER_ADDR=10.0.0.10
export SOCKET_IFNAME=eth0
export ACTOR_NUM_NODES=2
export ACTOR_NUM_GPUS_PER_NODE=8
bash slime_scripts/qwen3_6_35b/train.sh
```

External mode waits until Ray reports the requested node and GPU counts before
submitting the job. It does not use SSH or start remote processes itself, so
the node supervisor, container runtime, or cluster scheduler remains
responsible for process lifecycle.

Qwen3.6 defaults to the `flex` MoE dispatcher with DeepEP. To run the model
profile's standard all-to-all dispatcher instead, set both controls
explicitly:

```bash
export MOE_TOKEN_DISPATCHER_TYPE=alltoall
export MOE_ENABLE_DEEPEP=0
```

The first update on a newly provisioned node can be dominated by Triton
autotuning rather than training. Keep `TRITON_CACHE_DIR`,
`TORCHINDUCTOR_CACHE_DIR`, and `CUDA_CACHE_PATH` on persistent node-local
storage across diagnostic retries. The common launcher forwards these paths
to external Ray actors when they are set.

During a cold start, warm ranks may wait inside TP/EP/DP collectives while
another rank compiles or benchmarks a sequence-dependent kernel; low GPU power
on the waiting ranks alone is not evidence of a distributed deadlock. Confirm
progress by sampling Python stacks on more than one rank and by checking
cache-file activity before terminating the first update. If a PyTorch build
fails specifically while benchmarking an Inductor pointwise kernel, opt out of
that benchmark without disabling Inductor:

```bash
export SLIME_TORCHINDUCTOR_AUTOTUNE_POINTWISE=0
```

This selects a generated pointwise configuration without runtime benchmarking;
it can trade a small amount of kernel performance for a reliable cold start
but does not change the model, data, loss, or optimizer configuration. Leave
the variable unset for the PyTorch default.

## Output and resume behavior

Rollout output is resumable and includes progress files plus per-task artifacts. Data-selection commands write selected data and compact JSON summaries without embedding the private source corpus in the repository.

Training refuses to overwrite an existing output directory. Set `ALLOW_EXISTING_OUTPUT=1` only for an intentional resume and pass any required checkpoint controls through the documented environment variables in `slime_scripts/common/run_slime_sft.sh`.

SFT checkpoints remain model-only by default. Set both flags to retain complete
training state:

```bash
export SAVE_OPTIMIZER=1
export SAVE_RNG=1
```

Resume from a complete checkpoint with:

```bash
export LOAD_PATH=/workspace/output/previous-run
export LOAD_OPTIMIZER=1
export LOAD_RNG=1
export LOAD_ROLLOUT_STATE=1
export OUTPUT_DIR=/workspace/output/resumed-run
bash slime_scripts/qwen3_6_35b/train.sh
```

`LOAD_PATH` must contain `latest_checkpointed_iteration.txt`. When optimizer
loading is enabled, the launcher also restores the checkpointed optimizer
parameter scheduler. Exact SFT continuation also requires
`rollout/global_dataset_state_dict_<iteration>.pt`; this restores the shuffled
dataset cursor and sample indices. The launcher checks for that file by
default. Set `LOAD_ROLLOUT_STATE=0` only when deliberately restarting the data
sequence rather than continuing it.

In asynchronous SFT, checkpoint-boundary prefetch is deferred until after the
global dataset state is saved. This keeps the persisted cursor at the first
unconsumed batch. Megatron-restored scheduler progress is also used directly;
it is not advanced a second time during actor initialization.

A full-parameter Adam checkpoint can be several times larger and slower to
save than a model-only checkpoint, so measure the first save before selecting
a production save interval. For the closest continuation semantics, resume
with the same world size and parallel topology.

## Third-party code and license

`slime/` and `third_party/aira-evo/` retain their upstream license files and notices. OpenMLE material is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Models, datasets, SLIME, AIRA-Evo, Dojo, and their dependencies retain their respective terms.
