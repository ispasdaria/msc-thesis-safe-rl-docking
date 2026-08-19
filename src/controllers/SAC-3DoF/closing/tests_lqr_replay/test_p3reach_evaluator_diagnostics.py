"""Focused tests for P3 evaluator cone-containment diagnostics."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_sac_closing_no_safety_termination import cone_diagnostics  # noqa: E402


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
