"""Compatibility exports for the closing CBF-shielded environment."""

from envs.docking3dof.shields.closing_cbf import (
    CBF_TRAINING_VARIANTS,
    ClosingCBFShieldedGymEnv,
    build_closing_cbf_config,
)


__all__ = [
    "CBF_TRAINING_VARIANTS",
    "ClosingCBFShieldedGymEnv",
    "build_closing_cbf_config",
]
