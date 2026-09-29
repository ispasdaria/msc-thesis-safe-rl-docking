"""Focused tests for P3 evaluator cone-containment diagnostics."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_p3_reach_nonterminal_safety_continuation import (  # noqa: E402
    ACTION_SATURATION_THRESHOLD,
    action_command_diagnostics,
    braking_envelope_diagnostics,
    cone_diagnostics,
    velocity_limit_diagnostics,
)


CORE = SimpleNamespace(
    cone_length=2000.0,
    cone_base_radius=200.0,
    cone_slope=0.1,
)


def test_cone_diagnostics_are_zero_inside_cone():
    result = cone_diagnostics(
        CORE,
        {
            "s": 1000.0,
            "rho": 75.0,
            "geometry": {"in_C_dock": True, "warn": False},
        },
    )

    assert not result["cone_exit"]
    assert result["cone_allowed_radius_m"] == pytest.approx(100.0)
    assert result["cone_exit_depth"] == 0.0
    assert result["cone_lateral_excess_m"] == 0.0


def test_cone_warning_depth_measures_lateral_excess_on_base_radius_scale():
    result = cone_diagnostics(
        CORE,
        {
            "s": 1000.0,
            "rho": 150.0,
            "geometry": {"in_C_dock": False, "warn": True},
        },
    )

    assert result["cone_exit"]
    assert result["cone_warning"]
    assert result["cone_lateral_exit"]
    assert not result["cone_axial_exit"]
    assert result["cone_lateral_excess_m"] == pytest.approx(50.0)
    assert result["cone_exit_depth"] == pytest.approx(0.25)


def test_cone_diagnostics_separate_and_clip_axial_exit():
    result = cone_diagnostics(
        CORE,
        {
            "s": 4500.0,
            "rho": 100.0,
            "geometry": {"in_C_dock": False, "warn": True},
        },
    )

    assert result["cone_axial_exit"]
    assert not result["cone_lateral_exit"]
    assert result["cone_axial_excess_m"] == pytest.approx(2500.0)
    assert result["cone_exit_depth"] == 1.0


@pytest.mark.parametrize(
    ("port_speed", "axial_speed", "expected_trigger"),
    [
        (0.36, 0.20, "port_speed"),
        (0.30, 0.26, "axial_speed"),
        (0.36, 0.26, "both"),
        (0.35, 0.25, "none"),
    ],
)
def test_velocity_limit_diagnostics_identify_exact_trigger(
    port_speed,
    axial_speed,
    expected_trigger,
):
    core = SimpleNamespace(
        vp_unsafe_close=0.35,
        sdot_unsafe_close=0.25,
    )

    result = velocity_limit_diagnostics(
        core,
        {
            "velocity": {
                "vp_norm": port_speed,
                "s_dot_abs": axial_speed,
            }
        },
    )

    assert result["velocity_failure_trigger"] == expected_trigger
    assert result["port_speed_unsafe"] is (port_speed > 0.35)
    assert result["axial_speed_unsafe"] is (axial_speed > 0.25)


def test_action_diagnostics_report_body_saturation_and_orbital_action():
    core = SimpleNamespace(
        transform_action_to_orbital_frame=(
            lambda action_body, psi: np.asarray(action_body) + psi
        )
    )
    normalized = np.array(
        [ACTION_SATURATION_THRESHOLD, -0.94, -1.0],
        dtype=float,
    )
    physical = np.array([0.1, -0.2, 0.03], dtype=float)

    result = action_command_diagnostics(
        core,
        normalized_action=normalized,
        physical_action=physical,
        pre_step_psi=0.5,
    )

    assert result["body_ax_saturated"] is True
    assert result["body_ay_saturated"] is False
    assert result["body_alpha_saturated"] is True
    assert result["orbital_ax_mps2"] == pytest.approx(0.6)
    assert result["orbital_ay_mps2"] == pytest.approx(0.3)


def test_braking_diagnostics_are_explicitly_unavailable_for_older_envs():
    result = braking_envelope_diagnostics(SimpleNamespace())

    assert result == {"available": False}
