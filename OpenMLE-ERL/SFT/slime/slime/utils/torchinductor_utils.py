import logging
import os


logger = logging.getLogger(__name__)


def configure_torchinductor_from_env() -> None:
    """Apply opt-in TorchInductor controls before the actor compiles kernels."""

    value = os.environ.get("SLIME_TORCHINDUCTOR_AUTOTUNE_POINTWISE")
    if value is None:
        return
    if value not in {"0", "1"}:
        raise ValueError(
            "SLIME_TORCHINDUCTOR_AUTOTUNE_POINTWISE must be 0 or 1, "
            f"got {value!r}"
        )

    from torch._inductor import config

    enabled = value == "1"
    config.triton.autotune_pointwise = enabled
    logger.info("Set TorchInductor pointwise autotuning to %s", enabled)
