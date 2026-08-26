from __future__ import annotations

import subprocess
from pathlib import Path


SFT_ROOT = Path(__file__).resolve().parents[1]
COMMON = SFT_ROOT / "slime_scripts/common/run_slime_sft.sh"
RAY_NODE = SFT_ROOT / "slime_scripts/common/start_external_ray_node.sh"
MEGATRON_ACTOR = (
    SFT_ROOT / "slime/slime/backends/megatron_utils/actor.py"
)


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
    assert 'CKPT_ARGS+=(--load "${LOAD_PATH}")' in source
    assert "CKPT_ARGS+=(--no-save-optim)" in source
    assert "CKPT_ARGS+=(--no-save-rng)" in source
    assert "CKPT_ARGS+=(--use-checkpoint-opt_param-scheduler)" in source


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
