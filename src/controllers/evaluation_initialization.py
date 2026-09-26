"""Shared evaluation-only initial-state range CLI and seeded sampling."""

from __future__ import annotations

import argparse

import numpy as np


RANGE_ARGUMENTS = {
    "x_o": ("x-o-min", "x-o-max", 280.0, 320.0),
    "y_o": ("y-o-min", "y-o-max", -3.0, 3.0),
    "xdot_o": ("xdot-o-min", "xdot-o-max", -0.05, -0.01),
    "ydot_o": ("ydot-o-min", "ydot-o-max", -0.01, 0.01),
    "psi_deg": ("psi-min-deg", "psi-max-deg", -1.0, 1.0),
    "omega_z_deg_s": (
        "omega-z-min-deg-s",
        "omega-z-max-deg-s",
        -0.05,
        0.05,
    ),
}


def add_initial_range_arguments(parser: argparse.ArgumentParser) -> None:
    for key, (min_flag, max_flag, default_min, default_max) in RANGE_ARGUMENTS.items():
        parser.add_argument(
            f"--{min_flag}", dest=f"{key}_min", type=float, default=default_min
        )
        parser.add_argument(
            f"--{max_flag}", dest=f"{key}_max", type=float, default=default_max
        )


def initial_ranges_from_args(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for key, (min_flag, max_flag, _, _) in RANGE_ARGUMENTS.items():
        low = float(getattr(args, f"{key}_min"))
        high = float(getattr(args, f"{key}_max"))
        if not np.isfinite([low, high]).all():
            raise ValueError(f"--{min_flag} and --{max_flag} must be finite")
        if high < low:
            raise ValueError(f"--{max_flag} must be >= --{min_flag}")
        ranges[key] = (low, high)
    return ranges


def sample_initial_state(seed: int, ranges: dict[str, tuple[float, float]]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array(
        [
            rng.uniform(*ranges["x_o"]),
            rng.uniform(*ranges["y_o"]),
            rng.uniform(*ranges["xdot_o"]),
            rng.uniform(*ranges["ydot_o"]),
            np.deg2rad(rng.uniform(*ranges["psi_deg"])),
            np.deg2rad(rng.uniform(*ranges["omega_z_deg_s"])),
        ],
        dtype=np.float64,
    )
