"""End-to-end behavioral validation tests for the two-phase mission runner.

Unlike ``test_runners.py``, these tests use the real nonlinear environment and
the project's representative LQR guidance controllers.  A module-scoped
fixture runs one deterministic nominal mission and the tests evaluate it
against mission-level acceptance criteria.
"""

import os
import sys

import numpy as np
import pytest


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")
LQR_PATH = os.path.join(SRC_PATH, "controllers", "LQR-3DoF")

for path in (SRC_PATH, LQR_PATH):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_lqr_full_mission import create_runner, run_full_mission


def test_full_mission_runner_supports_phase_specific_time_steps():
    runner = create_runner(
        seed=300,
        closing_max_time=12_000.0,
        final_max_time=5_000.0,
        closing_dt=0.1,
        final_dt=1.0,
    )

    assert runner.closing_env.dt == pytest.approx(0.1)
    assert runner.final_env.dt == pytest.approx(1.0)
    assert runner.closing_env.max_time == pytest.approx(12_000.0)
    assert runner.final_env.max_time == pytest.approx(5_000.0)


@pytest.fixture(scope="module")
def nominal_mission():
    """Run a reproducible, representative P2-to-docking mission."""
    runner, result = run_full_mission(
        seed=0,
        mode="nominal",
        closing_max_time=12_000.0,
        final_max_time=5_000.0,
        integrator="rk4",
    )
    return runner, result


def test_nominal_two_phase_mission_reaches_docking(nominal_mission):
    """The intended nominal mission must complete both phases successfully."""
    _, result = nominal_mission

    assert result["mission_success"] is True
    assert result["failed_phase"] is None
    assert result["closing_result"]["termination_reason"] == "success_P3_target"
    assert result["final_result"] is not None
    assert result["final_result"]["termination_reason"] == "success_docking"


def test_terminal_states_satisfy_phase_acceptance_sets(nominal_mission):
    """Reported successes must meet the physical P3 and docking criteria."""
    runner, result = nominal_mission
    closing_state = result["closing_result"]["terminal_state"]
    final_state = result["final_result"]["terminal_state"]
    final_info = result["final_result"]["final_info"]

    assert runner.closing_env.is_at_P3_target(closing_state)
    assert runner.final_env.is_in_X_dock(final_state)
    assert final_info["port_distance"] <= 0.1
    assert final_info["port_speed"] <= 0.1
    assert abs(final_info["psi_e"]) <= np.deg2rad(4.0)
    assert abs(final_info["omega"]) <= np.deg2rad(0.2)


def test_phase_handover_is_state_continuous(nominal_mission):
    """Final approach must start from the actual closing terminal state."""
    runner, result = nominal_mission
    closing_state = result["closing_result"]["terminal_state"]
    transition_state = result["transition_state"]
    final_initial_state = runner.final_env.history[0]["state"]

    assert np.array_equal(transition_state, closing_state)
    assert np.array_equal(final_initial_state, transition_state)


def test_nominal_mission_histories_are_complete_and_finite(nominal_mission):
    """The end-to-end run must produce usable telemetry for both phases."""
    runner, result = nominal_mission

    for env, phase_result in (
        (runner.closing_env, result["closing_result"]),
        (runner.final_env, result["final_result"]),
    ):
        times, states, actions_raw, actions_clipped, actions_orbital = (
            env.get_history_arrays()
        )

        assert phase_result["num_steps"] > 0
        assert len(times) == phase_result["num_steps"] + 1
        assert np.all(np.diff(times) >= 0.0)
        assert np.all(np.isfinite(states))
        assert np.all(np.isfinite(actions_raw))
        assert np.all(np.isfinite(actions_clipped))
        assert np.all(np.isfinite(actions_orbital))


def test_lightweight_saturation_tracking_matches_saved_history(nominal_mission):
    runner, _ = nominal_mission

    for phase, env in (
        ("closing", runner.closing_env),
        ("final", runner.final_env),
    ):
        _, _, actions_raw, actions_clipped, _ = env.get_history_arrays()
        saturated_channels = np.abs(actions_raw - actions_clipped) > 1e-12
        tracked = runner.phase_metric_trackers[phase].metrics()

        assert tracked["saturation_fraction_total"] == pytest.approx(
            np.mean(np.any(saturated_channels, axis=1))
        )
        assert np.allclose(
            tracked["saturation_fraction_channels"],
            np.mean(saturated_channels, axis=0),
        )
        assert np.all(actions_clipped <= env.action_high + 1e-12)
        assert np.all(actions_clipped >= env.action_low - 1e-12)
