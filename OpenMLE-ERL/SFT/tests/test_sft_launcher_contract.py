from __future__ import annotations

import subprocess
from pathlib import Path

SFT_ROOT = Path(__file__).resolve().parents[1]
COMMON = SFT_ROOT / "slime_scripts/common/run_slime_sft.sh"
RAY_NODE = SFT_ROOT / "slime_scripts/common/start_external_ray_node.sh"
MEGATRON_ACTOR = SFT_ROOT / "slime/slime/backends/megatron_utils/actor.py"
MEGATRON_MODEL = SFT_ROOT / "slime/slime/backends/megatron_utils/model.py"
ASYNC_TRAIN = SFT_ROOT / "slime/train_async.py"
TRAIN_METRICS = SFT_ROOT / "slime/slime/utils/train_metric_utils.py"
GEN_CONFIG = (
    SFT_ROOT / "tts_search/configs/experiment/evolutionary_generational_sft.yaml"
)
AIRAEVO_EVO = SFT_ROOT / "third_party/aira-evo/src/dojo/solvers/evo/evo.py"
TOKEN_FILTER = SFT_ROOT / "tts_search/data_produce/token_filter.py"


def test_launchers_have_valid_bash_syntax() -> None:
    launchers = [
        COMMON,
        RAY_NODE,
        SFT_ROOT / "slime_scripts/qwen3_30b/train.sh",
        SFT_ROOT / "slime_scripts/qwen3_6_35b/train.sh",
    ]
    for launcher in launchers:
        subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_complete_checkpoint_and_resume_controls_are_explicit() -> None:
    source = COMMON.read_text(encoding="utf-8")
    assert 'SAVE_OPTIMIZER="${SAVE_OPTIMIZER:-0}"' in source
    assert 'SAVE_RNG="${SAVE_RNG:-0}"' in source
    assert 'LOAD_ROLLOUT_STATE="${LOAD_ROLLOUT_STATE:-1}"' in source
    assert 'CKPT_ARGS+=(--load "${LOAD_PATH}")' in source
    assert "CKPT_ARGS+=(--no-save-optim)" in source
    assert "CKPT_ARGS+=(--no-save-rng)" in source
    assert "CKPT_ARGS+=(--use-checkpoint-opt_param-scheduler)" in source
    assert "global_dataset_state_dict_${LOAD_ITERATION}.pt" in source


def test_training_seeds_are_explicit() -> None:
    source = COMMON.read_text(encoding="utf-8")
    assert 'TRAINING_SEED="${TRAINING_SEED:-20260829}"' in source
    assert 'ROLLOUT_SEED="${ROLLOUT_SEED:-20260829}"' in source
    assert '--seed "${TRAINING_SEED}"' in source
    assert '--rollout-seed "${ROLLOUT_SEED}"' in source


def test_generational_rollout_uses_the_writable_runtime_and_real_packages() -> None:
    source = GEN_CONFIG.read_text(encoding="utf-8")
    assert "OPENMLE_AIRAEVO_PACKAGE_ROOT" in source
    assert "task_root: ${search.runner.package_root}/examples/mle_bench" in source
    assert "available_packages:" in source
    assert "        - scikit-learn" in source
    assert "        - lightgbm" not in source
    assert "        - xgboost" not in source


def test_airaevo_handles_an_all_invalid_search() -> None:
    source = AIRAEVO_EVO.read_text(encoding="utf-8")
    assert "best_node.code if best_node is not None else None" in source


def test_airaevo_enforces_the_execution_budget_during_debug() -> None:
    source = AIRAEVO_EVO.read_text(encoding="utf-8")
    assert "self.cfg.step_limit - self.state.current_step - 1" in source
    assert "if self.state.current_step >= self.cfg.step_limit:" in source


def test_token_filter_counts_batch_encoding_input_ids() -> None:
    source = TOKEN_FILTER.read_text(encoding="utf-8")
    assert "if isinstance(ids, Mapping):" in source
    assert 'ids = ids["input_ids"]' in source


def test_external_ray_contract_is_scheduler_neutral() -> None:
    common_source = COMMON.read_text(encoding="utf-8")
    node_source = RAY_NODE.read_text(encoding="utf-8")
    assert 'RAY_CLUSTER_MODE="${RAY_CLUSTER_MODE:-local}"' in common_source
    assert '"expected_nodes": expected_nodes' in common_source
    assert '"expected_gpus": expected_gpus' in common_source
    assert '--working-dir "${SLIME_ROOT}"' in common_source
    assert "ssh " not in node_source
    assert '--address "${MASTER_ADDR}:${RAY_PORT}"' in node_source


def test_external_ray_propagates_persistent_compiler_controls() -> None:
    common_source = COMMON.read_text(encoding="utf-8")
    actor_source = MEGATRON_ACTOR.read_text(encoding="utf-8")
    for variable in (
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "SLIME_TORCHINDUCTOR_AUTOTUNE_POINTWISE",
    ):
        assert f'env_vars["{variable}"]' in common_source or (
            f'"{variable}":' in common_source
        )
    assert "configure_torchinductor_from_env()" in actor_source


def test_performance_and_checkpoint_controls_are_explicit() -> None:
    source = COMMON.read_text(encoding="utf-8")
    for variable in (
        "ASYNC_SAVE",
        "SLIME_DISABLE_SAVE",
        "OVERLAP_GRAD_REDUCE",
        "OVERLAP_PARAM_GATHER",
    ):
        assert f'{variable}="${{{variable}:-0}}"' in source
    assert "CKPT_ARGS+=(--async-save)" in source
    assert "PERF_ARGS+=(--overlap-grad-reduce)" in source
    assert "PERF_ARGS+=(--overlap-param-gather)" in source
    assert '"SLIME_DISABLE_SAVE": sys.argv[13]' in source
    assert '"NCCL_IB_HCA"' in source
    assert '"NCCL_NET_GDR_LEVEL"' in source


def test_gpu_utilization_is_not_vram_occupancy() -> None:
    source = TRAIN_METRICS.read_text(encoding="utf-8")
    assert '"gpu_memory_utilization"' in source
    assert "torch.cuda.utilization(device)" in source


def test_async_checkpoint_precedes_next_dataset_prefetch() -> None:
    source = ASYNC_TRAIN.read_text(encoding="utf-8")
    checkpoint_guard = source.index("save_this_step = (")
    guarded_prefetch = source.index(
        "if not save_this_step and rollout_id + 1 < args.num_rollout:"
    )
    cursor_save = source.index("rollout_manager.save.remote(rollout_id)")
    resumed_prefetch = source.index(
        "# Resume asynchronous overlap after the cursor has been persisted."
    )
    assert checkpoint_guard < guarded_prefetch < cursor_save < resumed_prefetch


def test_loaded_scheduler_is_not_advanced_twice() -> None:
    source = MEGATRON_MODEL.read_text(encoding="utf-8")
    guard = source.index("if opt_param_scheduler.num_steps == 0:")
    advance = source.index(
        "opt_param_scheduler.step(increment=iteration * args.global_batch_size)",
        guard,
    )
    model_return = source.index(
        "return model, optimizer, opt_param_scheduler, iteration",
        advance,
    )
    assert guard < advance < model_return
