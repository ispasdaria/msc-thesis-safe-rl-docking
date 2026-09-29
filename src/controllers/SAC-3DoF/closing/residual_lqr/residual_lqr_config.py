"""Canonical timing configuration for residual SAC around the closing LQR.

Historical residual experiments used the 1 Hz closing controller.  New
residual experiments use only the sampled-data-verified 10 Hz controller.  The
helpers in this module keep that choice out of individual training scripts and
make conversions between physical duration and interaction counts explicit.
"""

from __future__ import annotations

import math


CLOSING_LQR_DT_SECONDS = 0.1
CLOSING_LQR_CONTROL_HZ = 1.0 / CLOSING_LQR_DT_SECONDS
REWARD_REFERENCE_DT_SECONDS = 1.0

RESIDUAL_LQR_REWARD_VERSIONS = frozenset(
    {"lqr_residual", "lqr_reference_residual"}
)


def require_residual_lqr_dt(dt, *, context="residual SAC"):
    """Return ``dt`` after enforcing the canonical 0.1 s controller period."""
    value = float(dt)
    if not math.isfinite(value) or not math.isclose(
        value,
        CLOSING_LQR_DT_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{context} requires the dt={CLOSING_LQR_DT_SECONDS:g} s closing "
            f"LQR; received dt={dt!r}. Historical 1 Hz experiments must use "
            "their archived entry points and artifacts."
        )
    return value


def resolve_environment_dt(reward_version, dt=None):
    """Resolve an environment period without changing non-LQR HER defaults."""
    if reward_version in RESIDUAL_LQR_REWARD_VERSIONS:
        value = CLOSING_LQR_DT_SECONDS if dt is None else dt
        return require_residual_lqr_dt(value, context=reward_version)
    return 1.0 if dt is None else float(dt)


def steps_for_duration(duration_seconds, *, dt=CLOSING_LQR_DT_SECONDS):
    """Convert an exactly representable physical duration to simulator steps."""
    duration = float(duration_seconds)
    period = float(dt)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_seconds must be finite and positive.")
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("dt must be finite and positive.")
    steps = int(round(duration / period))
    if not math.isclose(steps * period, duration, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"duration_seconds={duration:g} is not an integer multiple of "
            f"dt={period:g}."
        )
    return steps


def discount_for_dt(
    discount_at_reference_dt,
    *,
    dt=CLOSING_LQR_DT_SECONDS,
    reference_dt=REWARD_REFERENCE_DT_SECONDS,
):
    """Preserve a physical-time discount rate when changing controller period."""
    discount = float(discount_at_reference_dt)
    period = float(dt)
    reference_period = float(reference_dt)
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount_at_reference_dt must lie in (0, 1].")
    if period <= 0.0 or reference_period <= 0.0:
        raise ValueError("dt and reference_dt must be positive.")
    return discount ** (period / reference_period)
