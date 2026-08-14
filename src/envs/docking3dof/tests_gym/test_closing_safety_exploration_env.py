import numpy as np

from envs.docking3dof.closing_safety_exploration_env import (
    ClosingSafetyExplorationGymEnv,
    ClosingSafetyExplorationV2GymEnv,
    ClosingSafetyExplorationV3GymEnv,
    ClosingSafetyExplorationV4GymEnv,
    ClosingSafetyExplorationV5GymEnv,
)
from envs.docking3dof.gym_env import Docking3DoFGymEnv


def _set_velocity_unsafe_state(env):
    # Keep the sampled position/target, but exceed both closing hard velocity
    # limits.  A zero command leaves it unsafe after one one-second step.
    env.core_env.state[2] = -0.40
    env.core_env.state[3] = 0.0
    env.core_env.state[5] = 0.0


def test_standard_closing_environment_still_terminates_on_safety_violation():
    env = Docking3DoFGymEnv(
        phase="closing",
        mode="testing",
        closing_reward_version="v15",
        seed=7,
    )
    env.reset(seed=7)
    _set_velocity_unsafe_state(env)

    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert info["failure"] is True
    assert info["failure_reason"] == "safety_violation"
    assert "safety_termination_disabled" not in info


def test_experiment_continues_and_preserves_hard_safety_penalty():
    env = ClosingSafetyExplorationGymEnv(
        mode="testing",
        closing_reward_version="v15",
        seed=7,
    )
    env.reset(seed=7)
    _set_velocity_unsafe_state(env)

    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))

    assert terminated is False
    assert truncated is False
    assert info["failure"] is False
    assert bool(info["unsafe"]) is True
    assert bool(info["velocity"]["unsafe"]) is True
    assert info["safety_violation_observed"] is True
    assert info["safety_violation_ignored_for_termination"] is True
    assert info["observed_failure_reason"] == "safety_violation"
    assert info["termination_reason"] is None
    assert info["reward_components"]["reward_failure"] == -1000.0
    assert info["reward_components"]["safety_penalty_preserved"] is True


def test_experiment_still_terminates_on_p3_success():
    env = ClosingSafetyExplorationGymEnv(
        mode="testing",
        closing_reward_version="v15",
        seed=11,
    )
    env.reset(seed=11)
    env.core_env.state = env.core_env.p3_target.copy()

    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["termination_reason"] == "success_P3_target"


def test_experiment_still_truncates_at_time_limit():
    env = ClosingSafetyExplorationGymEnv(
        mode="testing",
        max_time=1.0,
        closing_reward_version="v15",
        seed=13,
    )
    env.reset(seed=13)

    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))

    assert terminated is False
    assert truncated is True
    assert info["time_limit"] is True
    assert info["termination_reason"] == "timeout"
    assert info["reward_components"]["reward_timeout"] == -200.0


def test_custom_reward_is_isolated_from_standard_v15_reward():
    standard = Docking3DoFGymEnv(
        phase="closing",
        mode="testing",
        closing_reward_version="v15",
        seed=17,
    )
    experiment = ClosingSafetyExplorationGymEnv(
        mode="testing",
        closing_reward_version="v15",
        seed=17,
    )
    standard.reset(seed=17)
    experiment.reset(seed=17)

    action = np.array([0.01, -0.02, 0.03], dtype=np.float32)
    standard_step = standard.step(action)
    experiment_step = experiment.step(action)

    np.testing.assert_allclose(standard_step[0], experiment_step[0])
    assert standard_step[2:4] == experiment_step[2:4]
    assert standard_step[4]["reward_components"]["closing_reward_version"] == "v15"
    assert "reward_scheme" not in standard_step[4]["reward_components"]
    assert experiment_step[4]["reward_components"]["reward_scheme"] == "p3_reach_v1"
    assert (
        experiment_step[4]["reward_components"]["shared_gym_reward_version_used"]
        is False
    )
    assert standard_step[1] != experiment_step[1]


def test_v2_capture_shaping_is_active_at_v1_stall_distance():
    env = ClosingSafetyExplorationV2GymEnv(mode="testing", seed=19)
    env.reset(seed=19)
    state = env.core_env.p3_target.copy()
    state[0] += 164.0

    reward_state = env._p3_reward_state(state)

    assert env.reward_scheme == "p3_reach_v2"
    assert reward_state["precision_gate"] > 0.90
    np.testing.assert_allclose(
        env.capture_scales,
        np.array(
            [
                100.0,
                20.0,
                0.05,
                0.02,
                np.deg2rad(10.0),
                np.deg2rad(0.1),
            ]
        ),
    )


def test_v2_unsafe_step_is_penalized_but_nonterminal():
    env = ClosingSafetyExplorationV2GymEnv(mode="testing", seed=23)
    env.reset(seed=23)
    _set_velocity_unsafe_state(env)

    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))

    assert terminated is False
    assert truncated is False
    assert info["reward_components"]["reward_scheme"] == "p3_reach_v2"
    assert info["reward_components"]["reward_failure"] == -1000.0
    assert info["reward_components"]["safety_penalty_preserved"] is True
    assert "capture_max_ratio" in info["reward_components"]


def test_v3_velocity_and_attitude_progress_are_active_at_p2():
    env = ClosingSafetyExplorationV3GymEnv(mode="testing", seed=29)
    env.reset(seed=29)
    state = env.core_env.state.copy()
    target = env.core_env.p3_target
    initial_position_error = np.linalg.norm(state[:2] - target[:2])

    state[2:4] = target[2:4] + 0.5 * (state[2:4] - target[2:4])
    attitude_error = env.core_env.wrap_to_pi(state[4] - target[4])
    state[4] = env.core_env.wrap_to_pi(target[4] + 0.5 * attitude_error)
    state[5] = target[5] + 0.5 * (state[5] - target[5])
    monitor = env.core_env.monitor_safety(state)
    info = {
        "success": False,
        "geometry": monitor["geometry"],
        "safety_violation_observed": False,
    }

    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info=info,
        terminated=False,
        truncated=False,
    )

    assert initial_position_error > 1000.0
    assert components["reward_velocity_progress"] > 0.0
    assert components["reward_attitude_progress"] > 0.0
    assert components["reward_rate_progress"] > 0.0


def test_v3_speed_shaping_is_zero_in_safe_set_and_smooth_in_warning_band():
    safe_env = ClosingSafetyExplorationV3GymEnv(mode="testing", seed=31)
    safe_env.reset(seed=31)
    safe_state = safe_env.core_env.state.copy()
    safe_state[2:4] = safe_env.core_env.p3_target[2:4]
    safe_state[5] = 0.0
    safe_monitor = safe_env.core_env.monitor_safety(safe_state)
    _, safe_components = safe_env._compute_sac_reward(
        state=safe_state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={
            "success": False,
            "geometry": safe_monitor["geometry"],
            "safety_violation_observed": False,
        },
        terminated=False,
        truncated=False,
    )

    warning_env = ClosingSafetyExplorationV3GymEnv(mode="testing", seed=31)
    warning_env.reset(seed=31)
    warning_state = warning_env.core_env.state.copy()
    warning_state[2] = -0.20
    warning_state[3] = 0.0
    warning_state[5] = 0.0
    warning_monitor = warning_env.core_env.monitor_safety(warning_state)
    _, warning_components = warning_env._compute_sac_reward(
        state=warning_state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={
            "success": False,
            "geometry": warning_monitor["geometry"],
            "safety_violation_observed": False,
        },
        terminated=False,
        truncated=False,
    )

    assert safe_components["reward_total_speed_warning"] == 0.0
    assert safe_components["reward_axial_speed_warning"] == 0.0
    assert warning_components["reward_total_speed_warning"] == 0.0
    assert warning_components["reward_axial_speed_warning"] < 0.0
    assert -20.0 < warning_components["reward_axial_speed_warning"] < 0.0


def test_v3_unsafe_step_is_penalized_but_nonterminal():
    env = ClosingSafetyExplorationV3GymEnv(mode="testing", seed=37)
    env.reset(seed=37)
    _set_velocity_unsafe_state(env)

    _, _, terminated, truncated, info = env.step(np.zeros(3))

    assert terminated is False
    assert truncated is False
    assert info["safety_violation_observed"] is True
    assert info["reward_components"]["reward_failure"] == -1000.0
    assert info["reward_components"]["safety_penalty_preserved"] is True


def test_v4_characteristic_scales_balance_physical_units_without_tolerances():
    env = ClosingSafetyExplorationV4GymEnv(mode="testing", seed=41)
    env.reset(seed=41)
    state = env.core_env.p3_target.copy() + env.goal_scales
    state[4] = env.core_env.wrap_to_pi(state[4])

    goal = env._balanced_goal_state(state)

    np.testing.assert_allclose(goal["potentials"], np.log(2.0))
    assert not np.allclose(env.goal_scales, env.core_env.p3_target_tolerance)


def test_v4_all_component_progress_terms_are_active_at_p2():
    env = ClosingSafetyExplorationV4GymEnv(mode="testing", seed=43)
    env.reset(seed=43)
    initial = env.core_env.state.copy()
    target = env.core_env.p3_target
    state = target + 0.5 * (initial - target)
    attitude_error = env.core_env.wrap_to_pi(initial[4] - target[4])
    state[4] = env.core_env.wrap_to_pi(target[4] + 0.5 * attitude_error)
    monitor = env.core_env.monitor_safety(state)

    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={
            "success": False,
            "geometry": monitor["geometry"],
            "safety_violation_observed": False,
        },
        terminated=False,
        truncated=False,
    )

    assert np.linalg.norm(initial[:2] - target[:2]) > 1000.0
    for component in env.goal_component_names:
        assert components[f"reward_{component}_progress"] > 0.0


def test_v4_speed_warning_and_hard_safety_behavior_are_preserved():
    safe_env = ClosingSafetyExplorationV4GymEnv(mode="testing", seed=47)
    safe_env.reset(seed=47)
    safe_state = safe_env.core_env.state.copy()
    safe_state[2:4] = safe_env.core_env.p3_target[2:4]
    safe_state[5] = 0.0
    safe_monitor = safe_env.core_env.monitor_safety(safe_state)
    _, safe_components = safe_env._compute_sac_reward(
        state=safe_state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={
            "success": False,
            "geometry": safe_monitor["geometry"],
            "safety_violation_observed": False,
        },
        terminated=False,
        truncated=False,
    )
    assert safe_components["reward_total_speed_warning"] == 0.0
    assert safe_components["reward_axial_speed_warning"] == 0.0

    unsafe_env = ClosingSafetyExplorationV4GymEnv(mode="testing", seed=47)
    unsafe_env.reset(seed=47)
    _set_velocity_unsafe_state(unsafe_env)
    _, _, terminated, truncated, info = unsafe_env.step(np.zeros(3))
    assert terminated is False
    assert truncated is False
    assert info["safety_violation_observed"] is True
    assert info["reward_components"]["reward_failure"] == -1000.0


def test_v4_geometry_warning_is_smooth_instead_of_flat():
    env = ClosingSafetyExplorationV4GymEnv(mode="testing", seed=53)
    env.reset(seed=53)
    state = env.core_env.state.copy()
    q = env.core_env.get_derived_quantities(state)
    cone_radius = env.core_env.cone_slope * q["s"]
    state[1] += 1.1 * cone_radius - q["r_p"][1]
    monitor = env.core_env.monitor_safety(state)
    assert monitor["geometry"]["warn"] is True

    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={
            "success": False,
            "geometry": monitor["geometry"],
            "safety_violation_observed": False,
        },
        terminated=False,
        truncated=False,
    )

    assert 0.0 < components["geometry_warning_depth"] < 1.0
    assert -1.0 < components["reward_geometry_warning"] < 0.0


def test_v5_terminal_velocity_gate_is_off_at_p2_and_on_near_p3():
    env = ClosingSafetyExplorationV5GymEnv(mode="testing", seed=59)
    env.reset(seed=59)

    far_goal = env._v5_goal_state(env.core_env.state)
    near_state = env.core_env.p3_target.copy()
    near_state[0] += 100.0
    near_goal = env._v5_goal_state(near_state)

    assert far_goal["position_error"] > 1000.0
    assert far_goal["terminal_velocity_gate"] < 0.001
    assert near_goal["terminal_velocity_gate"] > 0.95


def test_v5_translation_and_attitude_progress_are_active_at_p2():
    env = ClosingSafetyExplorationV5GymEnv(mode="testing", seed=61)
    env.reset(seed=61)
    initial = env.core_env.state.copy()
    target = env.core_env.p3_target
    state = initial.copy()
    state[0:2] = target[0:2] + 0.5 * (initial[0:2] - target[0:2])
    attitude_error = env.core_env.wrap_to_pi(initial[4] - target[4])
    state[4] = env.core_env.wrap_to_pi(target[4] + 0.5 * attitude_error)
    state[5] = target[5] + 0.5 * (initial[5] - target[5])
    monitor = env.core_env.monitor_safety(state)

    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={
            "success": False,
            "geometry": monitor["geometry"],
            "safety_violation_observed": False,
        },
        terminated=False,
        truncated=False,
    )

    assert components["terminal_velocity_gate"] < 0.10
    assert components["reward_axial_progress"] > 0.0
    assert components["reward_lateral_progress"] > 0.0
    assert components["reward_attitude_progress"] > 0.0
    assert components["reward_rate_progress"] > 0.0


def test_v5_velocity_progress_is_available_inside_terminal_region():
    env = ClosingSafetyExplorationV5GymEnv(mode="testing", seed=67)
    env.reset(seed=67)
    target = env.core_env.p3_target
    state = target.copy()
    state[0] += 100.0
    state[2] += 0.05
    env._previous_axial_error = 100.0
    env._previous_lateral_error = 0.0
    env._previous_velocity_error = 0.10
    env._previous_attitude_error = 0.0
    env._previous_rate_error = 0.0
    monitor = env.core_env.monitor_safety(state)

    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={
            "success": False,
            "geometry": monitor["geometry"],
            "safety_violation_observed": False,
        },
        terminated=False,
        truncated=False,
    )

    assert components["terminal_velocity_gate"] > 0.95
    assert components["reward_terminal_velocity_progress"] > 4.5


def test_v5_unsafe_step_is_penalized_but_nonterminal():
    env = ClosingSafetyExplorationV5GymEnv(mode="testing", seed=71)
    env.reset(seed=71)
    _set_velocity_unsafe_state(env)

    _, _, terminated, truncated, info = env.step(np.zeros(3))

    assert terminated is False
    assert truncated is False
    assert info["safety_violation_observed"] is True
    assert info["reward_components"]["reward_failure"] == -1000.0
    assert info["reward_components"]["safety_penalty_preserved"] is True
