"""Shared utilities for the isolated two-stage closing-controller trial.

Nothing in this module mutates the existing closing experiments.  It defines
the capture-envelope predicate and the portable ``.npz`` handover dataset used
by the collector, reset wrapper, trainer, and evaluator in this directory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


TRIAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRIAL_DIR.parents[4]
SRC_ROOT = REPO_ROOT / "src"
CLOSING_DIR = TRIAL_DIR.parent

DEFAULT_LONG_RANGE_MODEL = (
    CLOSING_DIR
    / "models"
    / "multiseed"
    / "sac_closing_v15_trial16_terminal_phase_balanced_guarded600k_seed0"
    / "seed_0"
    / "model_0100000_steps.zip"
)
DEFAULT_LONG_RANGE_VECNORMALIZE = DEFAULT_LONG_RANGE_MODEL.with_name(
    "model_0100000_steps_vecnormalize.pkl"
)
DEFAULT_DATASET_DIR = TRIAL_DIR / "artifacts" / "handover_datasets"
DEFAULT_MODEL_DIR = TRIAL_DIR / "artifacts" / "models"
DEFAULT_RESULT_DIR = TRIAL_DIR / "artifacts" / "results"
DEFAULT_LOG_DIR = TRIAL_DIR / "artifacts" / "logs"


def wrap_to_pi(angle):
    """Wrap a scalar or array angle to ``[-pi, pi)``."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class CaptureEnvelope:
    """Component-specific outer envelope for long-range/capture handover."""

    radius_m: float
    lateral_error_m: float = 15.0
    velocity_error_mps: float = 0.03
    attitude_error_deg: float = 3.0
    rate_error_degs: float = 0.10
    maximum_radial_rate_mps: float = 0.0
    require_not_unsafe: bool = True

    def validate(self):
        numeric_positive = {
            "radius_m": self.radius_m,
            "lateral_error_m": self.lateral_error_m,
            "velocity_error_mps": self.velocity_error_mps,
            "attitude_error_deg": self.attitude_error_deg,
            "rate_error_degs": self.rate_error_degs,
        }
        for name, value in numeric_positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(self.maximum_radial_rate_mps):
            raise ValueError("maximum_radial_rate_mps must be finite.")

    def to_dict(self):
        return asdict(self)


def target_error(state, p3_target):
    """Return the six-component state error relative to an episode P3 target."""
    state = np.asarray(state, dtype=np.float64)
    target = np.asarray(p3_target, dtype=np.float64)
    if state.shape != (6,) or target.shape != (6,):
        raise ValueError("state and p3_target must both have shape (6,).")
    error = state - target
    error[4] = float(wrap_to_pi(error[4]))
    return error


def handover_metrics(state, p3_target, tolerance):
    """Compute the diagnostics needed by the capture-envelope predicate."""
    error = target_error(state, p3_target)
    tolerance = np.asarray(tolerance, dtype=np.float64)
    if tolerance.shape != (6,) or np.any(tolerance <= 0.0):
        raise ValueError("tolerance must contain six positive values.")

    position_error = float(np.linalg.norm(error[0:2]))
    velocity_error = float(np.linalg.norm(error[2:4]))
    if position_error > 1e-12:
        radial_rate = float(np.dot(error[0:2], error[2:4]) / position_error)
    else:
        radial_rate = 0.0
    ratios = np.abs(error) / tolerance

    return {
        "error": error,
        "target_ratios": ratios,
        "max_target_ratio": float(np.max(ratios)),
        "position_error_m": position_error,
        "lateral_error_m": float(abs(error[1])),
        "velocity_error_mps": velocity_error,
        "attitude_error_deg": float(abs(np.rad2deg(error[4]))),
        "rate_error_degs": float(abs(np.rad2deg(error[5]))),
        "radial_rate_mps": radial_rate,
    }


def is_capture_envelope_entry(metrics, envelope, safety=None):
    """Return whether diagnostics lie inside the requested capture envelope."""
    envelope.validate()
    if envelope.require_not_unsafe and safety is not None:
        if bool(safety.get("unsafe", False)):
            return False
    return bool(
        metrics["position_error_m"] <= envelope.radius_m
        and metrics["lateral_error_m"] <= envelope.lateral_error_m
        and metrics["velocity_error_mps"] <= envelope.velocity_error_mps
        and metrics["attitude_error_deg"] <= envelope.attitude_error_deg
        and metrics["rate_error_degs"] <= envelope.rate_error_degs
        and metrics["radial_rate_mps"] <= envelope.maximum_radial_rate_mps
    )


REQUIRED_DATASET_KEYS = (
    "states",
    "p3_targets",
    "radii_m",
    "seeds",
    "steps",
    "times_s",
    "errors",
    "target_ratios",
    "position_errors_m",
    "lateral_errors_m",
    "velocity_errors_mps",
    "attitude_errors_deg",
    "rate_errors_degs",
    "radial_rates_mps",
    "max_target_ratios",
    "metadata_json",
)


def save_handover_dataset(path, records, metadata):
    """Save handover records in a non-pickled, portable NumPy archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(records)

    def matrix(name):
        if n == 0:
            return np.empty((0, 6), dtype=np.float64)
        return np.asarray([record[name] for record in records], dtype=np.float64)

    def vector(name, dtype=np.float64):
        return np.asarray([record[name] for record in records], dtype=dtype)

    np.savez_compressed(
        path,
        states=matrix("state"),
        p3_targets=matrix("p3_target"),
        radii_m=vector("radius_m"),
        seeds=vector("seed", np.int64),
        steps=vector("step", np.int64),
        times_s=vector("time_s"),
        errors=matrix("error"),
        target_ratios=matrix("target_ratios"),
        position_errors_m=vector("position_error_m"),
        lateral_errors_m=vector("lateral_error_m"),
        velocity_errors_mps=vector("velocity_error_mps"),
        attitude_errors_deg=vector("attitude_error_deg"),
        rate_errors_degs=vector("rate_error_degs"),
        radial_rates_mps=vector("radial_rate_mps"),
        max_target_ratios=vector("max_target_ratio"),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_handover_dataset(path, radius_m=None):
    """Load and validate a handover dataset, optionally selecting one radius."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_DATASET_KEYS if key not in archive]
        if missing:
            raise ValueError(f"Handover dataset is missing keys: {missing}")
        dataset = {key: np.asarray(archive[key]) for key in REQUIRED_DATASET_KEYS}

    states = dataset["states"]
    targets = dataset["p3_targets"]
    if states.ndim != 2 or states.shape[1:] != (6,):
        raise ValueError("states must have shape (N, 6).")
    if targets.shape != states.shape:
        raise ValueError("p3_targets must have the same shape as states.")
    n = states.shape[0]
    for key, value in dataset.items():
        if key == "metadata_json":
            continue
        if value.shape[0] != n:
            raise ValueError(f"Dataset field {key!r} has inconsistent length.")

    dataset["metadata"] = json.loads(str(dataset.pop("metadata_json").item()))
    dataset["path"] = str(path.resolve())
    if radius_m is not None:
        mask = np.isclose(dataset["radii_m"], float(radius_m), rtol=0.0, atol=1e-9)
        if not np.any(mask):
            available = sorted(set(dataset["radii_m"].tolist()))
            raise ValueError(
                f"No handovers for radius {radius_m}. Available radii: {available}"
            )
        for key, value in list(dataset.items()):
            if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == n:
                dataset[key] = value[mask]
    return dataset


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
