"""Compatibility import for historical phase-balanced SAC checkpoints.

The retained checkpoint was serialized before the closing experiments were
grouped into family directories. Stable-Baselines3 therefore imports the
original top-level module name while loading it from the repository root.
Keep that serialized name available without duplicating the replay-buffer
implementation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_IMPLEMENTATION_PATH = (
    Path(__file__).resolve().parent
    / "src"
    / "controllers"
    / "SAC-3DoF"
    / "closing"
    / "phase_balanced_replay"
    / "phase_balanced_replay_buffer_implementation.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_phase_balanced_replay_buffer_implementation",
    _IMPLEMENTATION_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise ImportError(f"Cannot load phase-balanced replay buffer from {_IMPLEMENTATION_PATH}")
_IMPLEMENTATION = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_IMPLEMENTATION)

PhaseBalancedReplayBuffer = _IMPLEMENTATION.PhaseBalancedReplayBuffer
PhaseBalancedReplayBuffer.__module__ = __name__

__all__ = ["PhaseBalancedReplayBuffer"]
