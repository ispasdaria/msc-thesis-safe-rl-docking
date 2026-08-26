"""
Basic Gymnasium / Stable-Baselines3 compatibility test for Docking3DoFGymEnv.

This script checks:
    - environment creation
    - observation_space and action_space
    - reset() output format
    - step() output format
    - Stable-Baselines3 check_env compatibility
"""

import os
import sys
import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from stable_baselines3.common.env_checker import check_env
from envs.docking3dof.gym_env import Docking3DoFGymEnv


def expected_observation(env):
    if env.phase == "closing":
        q = env.core_env.get_derived_quantities(env.core_env.state)
        port_state = np.array(
            [
                q["r_p"][0],
                q["r_p"][1],
                q["v_p"][0],
                q["v_p"][1],
                q["psi_e"],
                q["omega"],
            ]
        )
        error = env.core_env.state - env.core_env.p3_target
        error[4] = env.core_env.wrap_to_pi(error[4])
        return np.concatenate((port_state, error)).astype(np.float32)

    q = env.core_env.get_derived_quantities(env.core_env.state)
    return np.array(
        [
            q["r_p"][0],
            q["r_p"][1],
            q["v_p"][0],
            q["v_p"][1],
            q["psi_e"],
            q["omega"],
        ],
        dtype=np.float32,
    )


def test_closing_observation_is_augmented():
    env = Docking3DoFGymEnv(phase="closing", mode="testing", seed=7)
    obs, info = env.reset(seed=7)

    expected = expected_observation(env)

    assert np.allclose(obs, expected)
    assert info["observation_labels"] == (
        "rpx",
        "rpy",
        "vpx",
        "vpy",
        "psi_e",
        "omegaz",
        "p3_rx_error",
        "p3_ry_error",
        "p3_vx_error",
        "p3_vy_error",
        "p3_psi_error",
        "p3_omega_error",
    )

    env.close()


def test_closing_observation_wraps_attitude_error():
    env = Docking3DoFGymEnv(phase="closing", mode="testing", seed=11)
    env.reset(seed=11)

    env.core_env.state[4] = np.pi - 0.1
    env.core_env.p3_target[4] = -np.pi + 0.1

    obs = env._get_obs()

    assert np.isclose(obs[10], -0.2, atol=1e-6)
    env.close()


def test_action_hold_repeats_action_without_coarsening_integration():
    held_env = Docking3DoFGymEnv(
        phase="final",
        mode="testing",
        dt=1.0,
        action_hold_steps=2,
        seed=19,
    )
    reference_env = Docking3DoFGymEnv(
        phase="final",
        mode="testing",
        dt=1.0,
        action_hold_steps=1,
        seed=19,
    )
    held_env.reset(seed=19)
    reference_env.reset(seed=19)
    action = np.array([0.2, -0.1, 0.05], dtype=np.float32)

    held_obs, held_reward, held_terminated, held_truncated, held_info = (
        held_env.step(action)
    )
    reference_reward = 0.0
    for _ in range(2):
        (
            reference_obs,
            reward,
            reference_terminated,
            reference_truncated,
            reference_info,
        ) = reference_env.step(action)
        reference_reward += reward
        if reference_terminated or reference_truncated:
            break

    assert np.allclose(held_env.core_env.state, reference_env.core_env.state)
    assert np.allclose(held_obs, reference_obs)
    assert np.isclose(held_reward, reference_reward)
    assert held_terminated == reference_terminated
    assert held_truncated == reference_truncated
    assert np.isclose(held_env.core_env.time, 2.0)
    assert np.isclose(reference_env.core_env.time, 2.0)
    assert held_info["integration_step_seconds"] == 1.0
    assert held_info["action_hold_steps"] == 2
    assert held_info["action_hold_steps_executed"] == 2
    assert held_info["action_hold_seconds_executed"] == 2.0
    assert held_info["control_interval_seconds"] == 2.0
    assert held_info["action_hold_reward_aggregation"] == "undiscounted_sum"
    assert np.isclose(
        held_info["reward_components"]["reward_total"],
        held_reward,
    )
    assert sum(
        held_info["action_hold_classification_counts"][level]
        for level in ("safe", "warning", "unsafe")
    ) == 2

    held_env.close()
    reference_env.close()


def test_action_hold_stops_at_first_terminal_substep():
    env = Docking3DoFGymEnv(
        phase="final",
        mode="testing",
        dt=1.0,
        max_time=1.0,
        action_hold_steps=2,
        seed=23,
    )
    env.reset(seed=23)

    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))

    assert not terminated
    assert truncated
    assert info["action_hold_steps_executed"] == 1
    assert info["action_hold_seconds_executed"] == 1.0
    assert env.core_env.time == 1.0
    env.close()


def test_action_hold_steps_must_be_a_positive_integer():
    for invalid_value in (0, -1, 1.5, True):
        try:
            Docking3DoFGymEnv(action_hold_steps=invalid_value)
        except ValueError as error:
            assert "action_hold_steps" in str(error)
        else:
            raise AssertionError(f"Accepted invalid action hold: {invalid_value!r}")


def test_adaptive_action_hold_uses_far_near_and_warning_cadences():
    env = Docking3DoFGymEnv(
        phase="closing",
        mode="testing",
        action_hold_steps=2,
        adaptive_action_hold_distance_m=400.0,
        seed=17,
    )
    env.reset(seed=17)

    requested, reason, distance = env._resolve_action_hold()
    assert requested == 2
    assert reason == "far_from_p3"
    assert distance > 400.0

    env.core_env.state = env.core_env.p3_target.copy()
    env.core_env.state[0] += 300.0
    requested, reason, distance = env._resolve_action_hold()
    assert requested == 1
    assert reason == "near_p3"
    assert np.isclose(distance, 300.0)

    env.reset(seed=17)
    env.core_env.state[2] = env.core_env.p3_target[2] + 0.20
    requested, reason, _ = env._resolve_action_hold()
    assert env.core_env.monitor_safety(env.core_env.state)["warning"]
    assert requested == 1
    assert reason == "safety_warning"
    env.close()


def test_adaptive_action_hold_returns_control_on_crossing_near_boundary():
    env = Docking3DoFGymEnv(
        phase="closing",
        mode="testing",
        action_hold_steps=2,
        adaptive_action_hold_distance_m=400.0,
        seed=17,
    )
    env.reset(seed=17)
    env.core_env.state = env.core_env.p3_target.copy()
    env.core_env.state[0] += 400.05
    env.core_env.state[2] -= 0.10

    _, _, terminated, truncated, info = env.step(
        np.zeros(3, dtype=np.float32)
    )

    assert not terminated
    assert not truncated
    assert info["action_hold_steps_requested"] == 2
    assert info["action_hold_steps_executed"] == 1
    assert info["action_hold_break_reason"] == "entered_near_p3_region"
    assert info["p3_position_error_m"] <= 400.0
    env.close()


def test_adaptive_action_hold_configuration_must_be_valid():
    for invalid_distance in (0.0, -1.0, np.inf):
        try:
            Docking3DoFGymEnv(
                phase="closing",
                adaptive_action_hold_distance_m=invalid_distance,
            )
        except ValueError as error:
            assert "adaptive_action_hold_distance_m" in str(error)
        else:
            raise AssertionError(
                f"Accepted invalid adaptive distance: {invalid_distance!r}"
            )

    try:
        Docking3DoFGymEnv(
            phase="final",
            adaptive_action_hold_distance_m=400.0,
        )
    except ValueError as error:
        assert "adaptive action hold" in str(error)
    else:
        raise AssertionError("Accepted adaptive hold for final approach.")


def test_closing_reward_v15_uses_symmetric_progress():
    env = Docking3DoFGymEnv(
        phase="closing",
        mode="testing",
        seed=13,
        closing_reward_version="v15",
    )
    env.reset(seed=13)

    state = env.core_env.state.copy()
    state[2] = -0.20
    state[5] = 0.0
    target_error = state - env.core_env.p3_target
    target_error[4] = env.core_env.wrap_to_pi(target_error[4])
    pos_error = np.linalg.norm(target_error[0:2])

    env.prev_closing_pos_error = pos_error + 0.1
    reward, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.array([0.0, 0.0, 0.4]),
        physical_action=np.zeros(3),
        info={"warning": False, "failure": False, "success": False},
        terminated=False,
        truncated=False,
    )

    assert np.isfinite(reward)
    assert np.isclose(components["progress_port"], 0.1)
    assert np.isclose(components["reward_progress"], 2.0)
    assert np.isclose(components["progress_gate"], 1.0)
    assert np.isclose(components["reward_speed_margin"], 0.0)
    assert np.isclose(components["reward_alpha"], 0.0)
    assert np.isclose(
        components["reward_lateral_speed"],
        -5.0 * abs(target_error[3]),
    )
    assert np.isclose(
        components["reward_lateral"],
        -0.01 * abs(target_error[1]),
    )
    assert np.isclose(components["s_dot"], -0.20)
    assert np.isclose(components["reward_s_dot"], -5.0)

    # Reward v15 uses symmetric distance progress, so a 0.1 m regression costs
    # the same magnitude that a 0.1 m improvement earns.
    env.prev_closing_pos_error = pos_error - 0.1
    _, regressing_components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"warning": False, "failure": False, "success": False},
        terminated=False,
        truncated=False,
    )

    assert np.isclose(regressing_components["progress_port"], -0.1)
    assert np.isclose(regressing_components["reward_progress"], -2.0)
    assert np.isclose(regressing_components["reward_axial_overshoot"], 0.0)

    # Overshoot remains measurable for evaluation, but reward v15 does not
    # attach a separate penalty to it.
    overshoot_state = state.copy()
    overshoot_state[0] = (
        env.core_env.p3_target[0]
        - env.core_env.p3_target_tolerance[0]
        - 50.0
    )
    overshoot_state[1] = env.core_env.p3_target[1]
    overshoot_pos_error = np.linalg.norm(
        overshoot_state[0:2] - env.core_env.p3_target[0:2]
    )
    env.prev_closing_pos_error = overshoot_pos_error
    _, overshoot_components = env._compute_sac_reward(
        state=overshoot_state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"warning": False, "failure": False, "success": False},
        terminated=False,
        truncated=False,
    )

    assert np.isclose(overshoot_components["axial_overshoot"], 50.0)
    assert np.isclose(overshoot_components["reward_axial_overshoot"], 0.0)
    assert np.isclose(overshoot_components["reward_progress"], 0.0)

    # Reducing the wrapped P3-relative attitude error by 2 deg earns the same
    # magnitude that increasing it by 2 deg loses.
    target_psi = env.core_env.p3_target[4]
    attitude_state = state.copy()
    env.prev_closing_attitude_error = np.deg2rad(10.0)
    attitude_state[4] = target_psi + np.deg2rad(8.0)
    _, improving_components = env._compute_sac_reward(
        state=attitude_state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"warning": False, "failure": False, "success": False},
        terminated=False,
        truncated=False,
    )

    expected_attitude_progress_reward = 50.0 * np.deg2rad(2.0)
    assert np.isclose(
        improving_components["attitude_progress"],
        np.deg2rad(2.0),
    )
    assert np.isclose(
        improving_components["reward_attitude_progress"],
        expected_attitude_progress_reward,
    )

    attitude_state[4] = target_psi + np.deg2rad(10.0)
    _, worsening_components = env._compute_sac_reward(
        state=attitude_state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"warning": False, "failure": False, "success": False},
        terminated=False,
        truncated=False,
    )

    assert np.isclose(
        worsening_components["attitude_progress"],
        -np.deg2rad(2.0),
    )
    assert np.isclose(
        worsening_components["reward_attitude_progress"],
        -expected_attitude_progress_reward,
    )

    # A 100 m P3-relative lateral offset has a persistent -1 reward. Keep the
    # lateral velocity at its sampled P3 target to isolate the position term.
    lateral_position_state = state.copy()
    lateral_position_state[1] = env.core_env.p3_target[1] + 100.0
    lateral_position_state[3] = env.core_env.p3_target[3]
    env.prev_closing_pos_error = np.linalg.norm(
        lateral_position_state[0:2] - env.core_env.p3_target[0:2]
    )
    _, lateral_position_components = env._compute_sac_reward(
        state=lateral_position_state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"warning": False, "failure": False, "success": False},
        terminated=False,
        truncated=False,
    )

    assert np.isclose(lateral_position_components["v_lateral"], 100.0)
    assert np.isclose(lateral_position_components["reward_lateral"], -1.0)
    assert np.isclose(
        lateral_position_components["reward_lateral_speed"],
        0.0,
    )

    # Isolate total port-speed shaping from axial-speed shaping: pure lateral
    # motion at 0.30 m/s is halfway through the [0.25, 0.35] m/s warning band.
    lateral_state = state.copy()
    lateral_state[2] = 0.0
    lateral_state[3] = 0.30
    lateral_state[5] = 0.0
    lateral_target_error = lateral_state - env.core_env.p3_target
    lateral_target_error[4] = env.core_env.wrap_to_pi(lateral_target_error[4])
    env.prev_closing_pos_error = np.linalg.norm(lateral_target_error[0:2])
    _, lateral_components = env._compute_sac_reward(
        state=lateral_state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"warning": False, "failure": False, "success": False},
        terminated=False,
        truncated=False,
    )

    assert np.isclose(lateral_components["port_speed"], 0.30)
    assert np.isclose(lateral_components["s_dot"], 0.0)
    assert np.isclose(lateral_components["reward_s_dot"], 0.0)
    assert np.isclose(lateral_components["reward_speed_margin"], -5.0)

    env.prev_closing_pos_error = pos_error
    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"warning": False, "failure": False, "success": False},
        terminated=False,
        truncated=False,
    )

    assert np.isclose(components["no_progress"], 1.0)
    assert np.isclose(components["reward_no_progress"], -0.05)

    env.prev_closing_pos_error = pos_error + 0.1
    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"warning": True, "failure": True, "success": False},
        terminated=True,
        truncated=False,
    )

    assert np.isclose(components["reward_warning"], -10.0)
    assert np.isclose(components["reward_failure"], -1000.0)
    env.close()


def test_closing_reward_v16_normalizes_terminal_errors():
    env = Docking3DoFGymEnv(
        phase="closing",
        mode="testing",
        seed=17,
        closing_reward_version="v16",
    )
    env.reset(seed=17)

    tolerance = env.core_env.p3_target_tolerance
    state = env.core_env.p3_target + tolerance
    state[4] = env.core_env.p3_target[4] + tolerance[4]
    pos_error = np.linalg.norm(tolerance[0:2])
    env.prev_closing_pos_error = pos_error
    env.prev_closing_lateral_error = tolerance[1]
    env.prev_closing_attitude_error = tolerance[4]

    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"warning": False, "failure": False, "success": False},
        terminated=False,
        truncated=False,
    )

    expected_terminal_weight = 1.0
    assert components["closing_reward_version"] == "v16"
    assert np.isclose(components["p3_axial_ratio"], 1.0)
    assert np.isclose(components["p3_lateral_ratio"], 1.0)
    assert np.isclose(components["p3_vx_ratio"], 1.0)
    assert np.isclose(components["p3_vy_ratio"], 1.0)
    assert np.isclose(components["p3_attitude_ratio"], 1.0)
    assert np.isclose(components["p3_rate_ratio"], 1.0)
    assert np.isclose(components["p3_normalized_error_mean"], 1.0)
    assert np.isclose(components["p3_normalized_error_max"], 1.0)
    assert np.isclose(
        components["p3_terminal_weight"],
        expected_terminal_weight,
    )
    assert np.isclose(
        components["reward_tolerance_aggregate"],
        -0.05 * expected_terminal_weight,
    )
    assert np.isclose(
        components["reward_tolerance_worst"],
        -0.10 * expected_terminal_weight,
    )

    # Ratios are clipped rather than distance-gated, avoiding an incentive to
    # remain just outside a terminal shaping boundary.
    far_state = env.core_env.p3_target.copy()
    far_state[0] += 301.0
    env.prev_closing_pos_error = 301.0
    _, far_components = env._compute_sac_reward(
        state=far_state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"warning": False, "failure": False, "success": False},
        terminated=False,
        truncated=False,
    )
    assert np.isclose(far_components["p3_axial_ratio"], 301.0 / 20.0)
    assert np.isclose(far_components["p3_terminal_weight"], 1.0)
    assert np.isclose(
        far_components["p3_normalized_error_mean"],
        10.0 / 6.0,
    )
    assert np.isclose(far_components["p3_normalized_error_max"], 10.0)
    assert np.isclose(
        far_components["reward_tolerance_aggregate"],
        -0.05 * 10.0 / 6.0,
    )
    assert np.isclose(far_components["reward_tolerance_worst"], -1.0)
    env.close()


def test_closing_reward_v17_uses_smooth_terminal_progress():
    env = Docking3DoFGymEnv(
        phase="closing",
        mode="testing",
        seed=19,
        closing_reward_version="v17",
    )
    env.reset(seed=19)

    tolerance = env.core_env.p3_target_tolerance
    previous_error = 2.0 * tolerance
    previous_terminal = env._closing_v17_terminal_cost(
        previous_error,
        tolerance,
    )
    env.prev_closing_terminal_cost = previous_terminal["cost"]

    state = env.core_env.p3_target + tolerance
    state[4] = env.core_env.p3_target[4] + tolerance[4]
    env.prev_closing_pos_error = np.linalg.norm(tolerance[0:2])
    env.prev_closing_lateral_error = tolerance[1]
    env.prev_closing_attitude_error = tolerance[4]
    _, improving = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"warning": False, "failure": False, "success": False},
        terminated=False,
        truncated=False,
    )

    expected_progress = np.log(3.0) - np.log(2.0)
    assert improving["closing_reward_version"] == "v17"
    assert np.isclose(improving["p3_log_error_mean"], np.log(2.0))
    assert np.isclose(
        improving["p3_log_error_smooth_max"],
        np.log(2.0),
    )
    assert np.isclose(improving["p3_terminal_cost"], np.log(2.0))
    assert np.isclose(
        improving["p3_terminal_cost_progress"],
        expected_progress,
    )
    assert np.isclose(
        improving["reward_tolerance_progress"],
        20.0 * expected_progress,
    )
    assert np.isclose(improving["p3_terminal_progress_scale"], 20.0)
    assert np.isclose(improving["reward_tolerance_aggregate"], 0.0)
    assert np.isclose(improving["reward_tolerance_worst"], 0.0)

    # Holding the same terminal error has no persistent v17 cost.
    _, stationary = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"warning": False, "failure": False, "success": False},
        terminated=False,
        truncated=False,
    )
    assert np.isclose(stationary["p3_terminal_cost_progress"], 0.0)
    assert np.isclose(stationary["reward_tolerance_progress"], 0.0)

    # log1p remains sensitive above v16's former ratio-10 clipping boundary.
    far_error = np.zeros(6)
    far_error[0] = 100.0 * tolerance[0]
    farther_error = np.zeros(6)
    farther_error[0] = 101.0 * tolerance[0]
    far_cost = env._closing_v17_terminal_cost(far_error, tolerance)["cost"]
    farther_cost = env._closing_v17_terminal_cost(
        farther_error,
        tolerance,
    )["cost"]
    assert farther_cost > far_cost
    env.close()


def test_closing_reward_v18_scales_v17_progress():
    env = Docking3DoFGymEnv(
        phase="closing",
        mode="testing",
        seed=23,
        closing_reward_version="v18",
    )
    env.reset(seed=23)

    tolerance = env.core_env.p3_target_tolerance
    previous_error = 2.0 * tolerance
    env.prev_closing_terminal_cost = env._closing_v17_terminal_cost(
        previous_error,
        tolerance,
    )["cost"]

    state = env.core_env.p3_target + tolerance
    state[4] = env.core_env.p3_target[4] + tolerance[4]
    env.prev_closing_pos_error = np.linalg.norm(tolerance[0:2])
    env.prev_closing_lateral_error = tolerance[1]
    env.prev_closing_attitude_error = tolerance[4]
    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"warning": False, "failure": False, "success": False},
        terminated=False,
        truncated=False,
    )

    expected_progress = np.log(3.0) - np.log(2.0)
    assert components["closing_reward_version"] == "v18"
    assert np.isclose(components["p3_terminal_progress_scale"], 500.0)
    assert np.isclose(
        components["p3_terminal_cost_progress"],
        expected_progress,
    )
    assert np.isclose(
        components["reward_tolerance_progress"],
        500.0 * expected_progress,
    )
    assert np.isclose(components["reward_tolerance_aggregate"], 0.0)
    assert np.isclose(components["reward_tolerance_worst"], 0.0)
    env.close()


def test_closing_reward_v19_uses_persistent_terminal_state_cost():
    env = Docking3DoFGymEnv(
        phase="closing",
        mode="testing",
        seed=29,
        closing_reward_version="v19",
    )
    env.reset(seed=29)

    tolerance = env.core_env.p3_target_tolerance
    state = env.core_env.p3_target + tolerance
    state[4] = env.core_env.p3_target[4] + tolerance[4]
    env.prev_closing_pos_error = np.linalg.norm(tolerance[0:2])
    env.prev_closing_lateral_error = tolerance[1]
    env.prev_closing_attitude_error = tolerance[4]

    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"warning": False, "failure": False, "success": False},
        terminated=False,
        truncated=False,
    )

    assert components["closing_reward_version"] == "v19"
    assert np.isclose(components["p3_terminal_cost"], np.log(2.0))
    assert np.isclose(
        components["reward_tolerance_state"],
        -10.0 * np.log(2.0),
    )
    assert np.isclose(components["reward_tolerance_progress"], 0.0)
    assert np.isclose(components["reward_progress"], 0.0)
    assert np.isclose(components["reward_position"], 0.0)
    assert np.isclose(
        components["reward_total"],
        components["reward_tolerance_state"],
    )

    # The state cost remains active without requiring progress on this step.
    _, stationary = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"warning": False, "failure": False, "success": False},
        terminated=False,
        truncated=False,
    )
    assert np.isclose(
        stationary["reward_tolerance_state"],
        -10.0 * np.log(2.0),
    )
    env.close()


def test_closing_reward_v20_restores_only_velocity_margins():
    env = Docking3DoFGymEnv(
        phase="closing",
        mode="testing",
        seed=31,
        closing_reward_version="v20",
    )
    env.reset(seed=31)

    tolerance = env.core_env.p3_target_tolerance
    state = env.core_env.p3_target + tolerance
    state[4] = env.core_env.p3_target[4] + tolerance[4]
    env.prev_closing_pos_error = np.linalg.norm(tolerance[0:2])
    env.prev_closing_lateral_error = tolerance[1]
    env.prev_closing_attitude_error = tolerance[4]

    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"warning": False, "failure": False, "success": False},
        terminated=False,
        truncated=False,
    )

    assert components["closing_reward_version"] == "v20"
    assert np.isclose(
        components["reward_tolerance_state"],
        -10.0 * np.log(2.0),
    )
    assert np.isclose(components["reward_progress"], 0.0)
    assert np.isclose(components["reward_position"], 0.0)
    expected_total = (
        components["reward_tolerance_state"]
        + components["reward_action"]
        + components["reward_s_dot"]
        + components["reward_speed_margin"]
    )
    assert np.isclose(components["reward_total"], expected_total)
    env.close()


def check_single_phase(phase):
    print("=" * 80)
    print(f"Testing Gym wrapper for phase: {phase}")
    print("=" * 80)

    env = Docking3DoFGymEnv(
        phase=phase,
        mode="training",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=0,
    )

    print("\nObservation space:")
    print(env.observation_space)

    print("\nAction space:")
    print(env.action_space)

    obs, info = env.reset(seed=0)

    print("\nReset output:")
    print("obs:", obs)
    print("obs shape:", obs.shape)
    print("obs dtype:", obs.dtype)
    print("info keys:", list(info.keys()))

    assert isinstance(obs, np.ndarray), "reset() observation must be a NumPy array."
    expected_shape = (12,) if phase == "closing" else (6,)
    assert obs.shape == expected_shape, (
        f"{phase} observation must have shape {expected_shape}."
    )
    assert obs.dtype == np.float32, "Observation must have dtype float32."
    assert isinstance(info, dict), "reset() info must be a dictionary."

    assert env.observation_space.contains(obs), (
        "Initial observation is not inside observation_space."
    )

    expected_obs = expected_observation(env)
    assert np.allclose(obs, expected_obs), (
        "Observation does not match the phase-specific observation definition."
    )

    action = env.action_space.sample()
    step_output = env.step(action)

    assert len(step_output) == 5, (
        "step() must return 5 values: obs, reward, terminated, truncated, info."
    )

    obs, reward, terminated, truncated, info = step_output

    print("\nOne-step output:")
    print("sampled normalized action:", action)
    print("obs:", obs)
    print("reward:", reward)
    print("terminated:", terminated)
    print("truncated:", truncated)
    print("termination reason:", info.get("termination_reason"))
    print("info keys:", list(info.keys()))

    assert isinstance(obs, np.ndarray), "step() observation must be a NumPy array."
    assert obs.shape == expected_shape, (
        f"{phase} step observation must have shape {expected_shape}."
    )
    assert obs.dtype == np.float32, "Step observation must have dtype float32."
    assert isinstance(reward, float), "Reward must be a Python float."
    assert isinstance(terminated, bool), "terminated must be a bool."
    assert isinstance(truncated, bool), "truncated must be a bool."
    assert isinstance(info, dict), "step() info must be a dictionary."

    expected_obs = expected_observation(env)
    assert np.allclose(obs, expected_obs), (
        "Step observation does not match the phase-specific observation definition."
    )

    assert "reward_components" in info, (
        "info must contain reward_components for diagnostics."
    )

    print("\nReward components:")
    for key, value in info["reward_components"].items():
        print(f"  {key}: {value}")

    print("\nRunning Stable-Baselines3 check_env...")
    check_env(env, warn=True)

    env.close()

    print(f"\nGym wrapper test passed for phase: {phase}\n")


def test_single_phase():
    """Exercise both supported mission phases under pytest."""
    check_single_phase("closing")
    check_single_phase("final")


def main():
    test_closing_observation_is_augmented()
    test_closing_observation_wraps_attitude_error()
    test_closing_reward_v15_uses_symmetric_progress()
    test_closing_reward_v16_normalizes_terminal_errors()
    test_closing_reward_v17_uses_smooth_terminal_progress()
    test_closing_reward_v18_scales_v17_progress()
    test_closing_reward_v19_uses_persistent_terminal_state_cost()
    test_closing_reward_v20_restores_only_velocity_margins()
    check_single_phase("closing")
    check_single_phase("final")

    print("=" * 80)
    print("All Gym wrapper tests passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
