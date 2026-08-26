# Two-node H20 reproduction notes

This document records public-assets compatibility work for a two-node
OpenRSI reconstruction. It is not an exact replay of the unpublished paper
training run.

## Scope

- upstream OpenRSI commit:
  `1f477c48b6b464e196b651abc229dfe75247315a`;
- public OpenMLE SFT parquet revision:
  `dcb1d89f67c50660b2322efdb58f0769b0036395`;
- one 8 x NVIDIA H20 96 GB node for SFT;
- one 8 x NVIDIA H20 96 GB node split between TP4 inference and four
  Sandbox/NatureBench lanes.

The public release does not include the exact RL train/eval data, historical
optimizer/checkpoint state, Program Database, paper trajectories, image
digests, or seed-level execution state. Results from this setup must therefore
be labelled as a public-assets reconstruction.

## SFT compatibility changes

The vendored SFT tree references a Qwen3.5 Megatron Bridge package that is not
present in the released directory. The bridge in this branch is imported from
the SLIME revision named by the release, commit
`680824dd5e01a2e83750bf87fc366ec6fa98766c`.

The Megatron model-provider wrapper accepts the newer `config`,
`pg_collection`, and `vp_stage` callback arguments only when the underlying
provider supports them. HF/Megatron argument validation accepts the modern
`layernorm_epsilon` and `rope_parameters` spellings and distinguishes dense
from MoE intermediate sizes.

On H20, a 32,180-token sample caused the unchunked vocabulary-parallel
cross-entropy path to request an additional 14.88 GiB FP32 buffer. Set:

```bash
export LOG_PROBS_CHUNK_SIZE=4096
```

The adapted run passed the same sample while retaining every token in the SFT
objective.

## Observed training controls

The released `3e-5` learning-rate trajectory remained finite but developed
sustained divergence after step 222 and was stopped after 253 completed
updates. A separate `1e-5` 50-update systems gate completed with finite loss,
wrote a 65 GiB Megatron checkpoint, converted to an exact 1,045-key HF index,
served successfully, and closed the Agent/Sandbox and two-update RL control
paths.

The 50-update gate is not a resumable segment of the 615-update run: it uses a
50-update cosine schedule and saves neither optimizer nor RNG state. The
full-run adaptation must use its own 615-update schedule from the base
checkpoint.

Curves are retained in the `openrsi` W&B project:

- released-LR divergence:
  <https://wandb.ai/leviking98z-zhejiang-university/openrsi/runs/openrsi-sft-qwen36-h20-attempt7>
- stable 50-update gate:
  <https://wandb.ai/leviking98z-zhejiang-university/openrsi/runs/openrsi-sft-qwen36-h20-core50-lr1e5>
- 615-update adaptation:
  <https://wandb.ai/leviking98z-zhejiang-university/openrsi/runs/openrsi-sft-qwen36-h20-full-lr1e5>
- two-update RL systems control:
  <https://wandb.ai/leviking98z-zhejiang-university/openrsi/runs/bu3sbqbl>

## NatureBench local lanes

`experiment/naturebench_local_lite_v2` runs the ten pinned Lite-v2 tasks
against a local evaluator and task-specific Docker images. The data adapter
expands `docker_image_template` with each task ID, allowing one image per task
without maintaining ten nearly identical Hydra files.

Formal benchmark reports should retain resolved configuration, task/image
digests, evaluator revision, per-candidate score, terminal submit result,
wall time, and GPU-hours.
