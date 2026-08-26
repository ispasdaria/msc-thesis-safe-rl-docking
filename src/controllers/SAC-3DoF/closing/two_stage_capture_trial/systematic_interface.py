"""Single source of truth for the systematic long-range/capture handover.

The final-approach controller was validated from the *testing* P3 box.  The
capture controller must therefore enter that box, while the long-range
controller must enter the six-dimensional intermediate set ``X_H`` defined
relative to the same episode-specific P3 reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATE_LABELS = ("rx", "ry", "vx", "vy", "psi", "omega")


def wrap_to_pi(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class SystematicHandoverContract:
    """Validated P3 box and the candidate full-state intermediate handover."""

    handover_distance_m: float = 150.0
    approach_velocity_offset_mps: float = 0.08

    @property
    def p3_lower(self):
        return np.array(
            [280.0, -3.0, -0.05, -0.01, np.deg2rad(-1.0), np.deg2rad(-0.05)],
            dtype=np.float64,
        )

    @property
    def p3_upper(self):
        return np.array(
            [320.0, 3.0, -0.01, 0.01, np.deg2rad(1.0), np.deg2rad(0.05)],
            dtype=np.float64,
        )

    @property
    def handover_tolerance(self):
        return np.array(
            [20.0, 5.0, 0.015, 0.01, np.deg2rad(1.0), np.deg2rad(0.05)],
            dtype=np.float64,
        )

    @property
    def capture_shaping_scale(self):
        # These are shaping scales only. Capture success is membership in the
        # validated P3 box, not target +/- these values.
        return np.array(
            [20.0, 2.0, 0.01, 0.01, np.deg2rad(0.5), np.deg2rad(0.05)],
            dtype=np.float64,
        )

    def validate(self):
        if self.handover_distance_m <= 0.0:
            raise ValueError("handover_distance_m must be positive.")
        if self.approach_velocity_offset_mps < 0.0:
            raise ValueError("approach_velocity_offset_mps cannot be negative.")
        if np.any(self.p3_lower >= self.p3_upper):
            raise ValueError("Every P3 lower bound must be below its upper bound.")
        if np.any(self.handover_tolerance <= 0.0):
            raise ValueError("Every handover tolerance must be positive.")

    def sample_p3(self, rng):
        state = rng.uniform(self.p3_lower, self.p3_upper)
        state[4] = float(wrap_to_pi(state[4]))
        return state

    def intermediate_target(self, p3_target, distance_m=None):
        p3_target = _state(p3_target, "p3_target")
        distance = self.handover_distance_m if distance_m is None else float(distance_m)
        if not 0.0 < distance <= self.handover_distance_m:
            raise ValueError(
                f"distance_m must lie in (0, {self.handover_distance_m:g}]."
            )
        fraction = distance / self.handover_distance_m
        target = p3_target.copy()
        target[0] += distance
        target[2] -= self.approach_velocity_offset_mps * fraction
        target[4] = float(wrap_to_pi(target[4]))
        return target

    def sample_handover_state(self, rng, p3_target, distance_m=None):
        target = self.intermediate_target(p3_target, distance_m=distance_m)
        state = target + rng.uniform(
            -self.handover_tolerance, self.handover_tolerance
        )
        state[4] = float(wrap_to_pi(state[4]))
        return state, target

    def in_validated_p3(self, state):
        state = _state(state, "state").copy()
        state[4] = float(wrap_to_pi(state[4]))
        return bool(np.all(state >= self.p3_lower) and np.all(state <= self.p3_upper))

    def in_handover_set(self, state, p3_target, distance_m=None):
        target = self.intermediate_target(p3_target, distance_m=distance_m)
        return bool(np.all(np.abs(state_error(state, target)) <= self.handover_tolerance))

    def p3_box_ratios(self, state):
        """Zero inside validated P3; normalized distance outside each face."""
        state = _state(state, "state").copy()
        state[4] = float(wrap_to_pi(state[4]))
        lower = self.p3_lower
        upper = self.p3_upper
        half_width = 0.5 * (upper - lower)
        below = np.maximum(lower - state, 0.0)
        above = np.maximum(state - upper, 0.0)
        return (below + above) / half_width

    def as_dict(self):
        return {
            "state_order": list(STATE_LABELS),
            "validated_p3_lower": self.p3_lower.tolist(),
            "validated_p3_upper": self.p3_upper.tolist(),
            "handover_distance_m": self.handover_distance_m,
            "approach_velocity_offset_mps": self.approach_velocity_offset_mps,
            "handover_tolerance": self.handover_tolerance.tolist(),
            "angle_units": "radians in arrays",
        }


def _state(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite six-component state.")
    return value


def state_error(state, target):
    error = _state(state, "state") - _state(target, "target")
    error = error.copy()
    error[4] = float(wrap_to_pi(error[4]))
    return error


DEFAULT_CONTRACT = SystematicHandoverContract()
DEFAULT_CONTRACT.validate()
