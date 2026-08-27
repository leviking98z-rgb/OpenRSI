#!/usr/bin/env bash
set -euo pipefail

SLIME_RUNTIME_ROOT="${SLIME_RUNTIME_ROOT:-/root/slime}"
TRAIN_FILE="${SLIME_RUNTIME_ROOT}/train.py"
ACTOR_FILE="${SLIME_RUNTIME_ROOT}/slime/backends/megatron_utils/actor.py"

test -w "${TRAIN_FILE}"
test -w "${ACTOR_FILE}"

python3 - "${TRAIN_FILE}" "${ACTOR_FILE}" <<'PY'
from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path


PINNED_SLIME_COMMIT = "680824dd5e01a2e83750bf87fc366ec6fa98766c"
train_path = Path(sys.argv[1])
actor_path = Path(sys.argv[2])

train_replacements = (
    (
        """import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
""",
        """import os

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
""",
    ),
    (
        """        if release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
""",
        """        checkpoint_due = release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        )
        defer_this_checkpoint = (
            os.environ.get("OPENRSI_DEFER_FINAL_CHECKPOINT", "0") == "1"
            and checkpoint_due
            and not release_train
            and rollout_id == args.num_rollout - 1
        )
        if checkpoint_due and not defer_this_checkpoint:
""",
    ),
    (
        """        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
""",
        """        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

        # On the terminal rollout, evaluate before checkpointing. The actor is
        # intentionally left resident after the final synchronous save, so it
        # never performs a second torch_memory_saver pinned-host backup merely
        # to exit the process.
        if defer_this_checkpoint:
            if args.offload_rollout:
                ray.get(rollout_manager.offload.remote())
            if actor_trains:
                actor_model.save_model(rollout_id, force_sync=True)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=True)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
""",
    ),
)

actor_replacements = (
    (
        """        if self.args.offload_train:
            self.sleep()

    @timer
    def update_weights(self) -> None:
""",
        """        if self.args.offload_train:
            defer_final_checkpoint = (
                force_sync and os.environ.get("OPENRSI_DEFER_FINAL_CHECKPOINT", "0") == "1"
            )
            if defer_final_checkpoint:
                logger.info(
                    "Final checkpoint completed after evaluation; leaving the actor resident until process exit "
                    "to avoid a redundant torch_memory_saver pause."
                )
            else:
                self.sleep()

    @timer
    def update_weights(self) -> None:
""",
    ),
)

specs = (
    (
        train_path,
        "8b1e231caaa59ac5970fdbcd1a1cd5d10bae6a293f27072781ce800df2baa0b0",
        "570ab6d2b6ace4506c61aa3760e42d320013271fc20fbcc2a831718175bab061",
        train_replacements,
    ),
    (
        actor_path,
        "34e275f039976c565df6b7ba2dc51180969932d79c1f850f35f57c1675e3bc24",
        "44a3f80ca295f4cfbfcb3bbed07f06088d1d7d34d2e29316269085a4c47f2d17",
        actor_replacements,
    ),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


pending: list[tuple[Path, bytes, int]] = []
already_patched = 0
for path, original_hash, patched_hash, replacements in specs:
    original_bytes = path.read_bytes()
    current_hash = sha256(original_bytes)
    if current_hash == patched_hash:
        compile(original_bytes.decode(), str(path), "exec")
        already_patched += 1
        continue
    if current_hash != original_hash:
        raise SystemExit(
            f"[OPENRSI PATCH] {path} does not match pristine or patched SLIME "
            f"{PINNED_SLIME_COMMIT}: sha256={current_hash}"
        )

    patched_text = original_bytes.decode()
    for old, new in replacements:
        count = patched_text.count(old)
        if count != 1:
            raise SystemExit(
                f"[OPENRSI PATCH] expected one exact replacement in {path}, found {count}"
            )
        patched_text = patched_text.replace(old, new, 1)

    compile(patched_text, str(path), "exec")
    patched_bytes = patched_text.encode()
    actual_patched_hash = sha256(patched_bytes)
    if actual_patched_hash != patched_hash:
        raise SystemExit(
            f"[OPENRSI PATCH] patched hash mismatch for {path}: "
            f"expected={patched_hash} actual={actual_patched_hash}"
        )
    pending.append((path, patched_bytes, stat.S_IMODE(path.stat().st_mode)))

for path, patched_bytes, mode in pending:
    temporary = path.with_name(f".{path.name}.openrsi-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(patched_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

for path, _, patched_hash, _ in specs:
    final_hash = sha256(path.read_bytes())
    if final_hash != patched_hash:
        raise SystemExit(
            f"[OPENRSI PATCH] post-write verification failed for {path}: sha256={final_hash}"
        )

if already_patched == len(specs):
    print("[OPENRSI PATCH] deferred final checkpoint patch already applied")
else:
    print(
        "[OPENRSI PATCH] deferred final checkpoint patch applied and verified "
        f"for SLIME {PINNED_SLIME_COMMIT}"
    )
PY
