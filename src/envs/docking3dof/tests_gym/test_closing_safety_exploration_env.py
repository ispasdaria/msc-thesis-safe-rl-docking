import numpy as np

from envs.docking3dof.closing_safety_exploration_env import (
    ClosingSafetyExplorationGymEnv,
    ClosingSafetyExplorationV2GymEnv,
    ClosingSafetyExplorationV3GymEnv,
    ClosingSafetyExplorationV4GymEnv,
    ClosingSafetyExplorationV5GymEnv,
    ClosingSafetyExplorationV6GymEnv,
    ClosingSafetyExplorationV7GymEnv,
    ClosingSafetyExplorationV8GymEnv,
    ClosingSafetyExplorationV9GymEnv,
    ClosingSafetyExplorationV10GymEnv,
    ClosingSafetyExplorationV11GymEnv,
    ClosingSafetyExplorationV12GymEnv,
    ClosingSafetyExplorationV13GymEnv,
    ClosingSafetyExplorationV14GymEnv,
    ClosingSafetyExplorationV15GymEnv,
    ClosingSafetyExplorationV16GymEnv,
    ClosingSafetyExplorationV17GymEnv,
    ClosingSafetyExplorationV18GymEnv,
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


def test_v6_smooth_max_uses_p3_tolerances_and_is_zero_at_target():
    env = ClosingSafetyExplorationV6GymEnv(mode="testing", seed=73)
    env.reset(seed=73)

    at_target = env._v6_goal_state(env.core_env.p3_target.copy())
    one_tolerance_away = env._v6_goal_state(
        env.core_env.p3_target + env.core_env.p3_target_tolerance
    )

    np.testing.assert_allclose(at_target["ratios"], np.zeros(6), atol=1e-12)
    assert abs(at_target["smooth_max_cost"]) < 1e-12
    np.testing.assert_allclose(one_tolerance_away["ratios"], np.ones(6))
    np.testing.assert_allclose(one_tolerance_away["smooth_max_cost"], 1.0)


def test_v6_progress_and_action_terms_follow_compact_reward_formula():
    env = ClosingSafetyExplorationV6GymEnv(mode="testing", seed=79)
    env.reset(seed=79)
    state = env.core_env.p3_target.copy()
    state[0] += env.core_env.p3_target_tolerance[0]
    goal = env._v6_goal_state(state)
    env._previous_smooth_max_cost = goal["smooth_max_cost"] + 2.0

    reward, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.array([1.0, 0.5, 0.0]),
        physical_action=np.zeros(3),
        info={"success": False, "safety_violation_observed": False},
        terminated=False,
        truncated=False,
    )

    assert components["smooth_max_progress_raw"] == 2.0
    assert components["smooth_max_progress_clipped"] == 1.0
    assert components["reward_smoothmax_progress"] == 10.0
    np.testing.assert_allclose(components["action_norm_squared"], 1.25)
    np.testing.assert_allclose(components["reward_action"], -0.00125)
    np.testing.assert_allclose(reward, components["reward_total"])


def test_v6_unsafe_step_has_smaller_penalty_and_remains_nonterminal():
    env = ClosingSafetyExplorationV6GymEnv(mode="testing", seed=83)
    env.reset(seed=83)
    _set_velocity_unsafe_state(env)

    _, _, terminated, truncated, info = env.step(np.zeros(3))

    assert terminated is False
    assert truncated is False
    assert info["safety_violation_observed"] is True
    assert info["reward_components"]["reward_scheme"] == "p3_reach_v6"
    assert info["reward_components"]["reward_failure"] == -100.0
    assert info["reward_components"]["safety_penalty_preserved"] is True


def test_v7_goal_cost_is_zero_at_target_and_balanced_at_one_tolerance():
    env = ClosingSafetyExplorationV7GymEnv(mode="testing", seed=89)
    env.reset(seed=89)

    at_target = env._v7_goal_state(env.core_env.p3_target.copy())
    one_tolerance_away = env._v7_goal_state(
        env.core_env.p3_target + env.core_env.p3_target_tolerance
    )

    assert abs(at_target["goal_cost"]) < 1e-12
    np.testing.assert_allclose(one_tolerance_away["ratios"], np.ones(6))
    np.testing.assert_allclose(
        one_tolerance_away["goal_cost"],
        np.log(2.0),
    )


def test_v7_nonworst_component_still_changes_goal_cost():
    env = ClosingSafetyExplorationV7GymEnv(mode="testing", seed=97)
    env.reset(seed=97)
    axial_only = env.core_env.p3_target.copy()
    axial_only[0] += 50.0 * env.core_env.p3_target_tolerance[0]
    axial_and_lateral = axial_only.copy()
    axial_and_lateral[1] += env.core_env.p3_target_tolerance[1]

    axial_cost = env._v7_goal_state(axial_only)["goal_cost"]
    combined_cost = env._v7_goal_state(axial_and_lateral)["goal_cost"]

    assert combined_cost > axial_cost


def test_v7_reward_uses_clipped_progress_and_unclipped_goal_state_cost():
    env = ClosingSafetyExplorationV7GymEnv(mode="testing", seed=101)
    env.reset(seed=101)
    state = env.core_env.p3_target + env.core_env.p3_target_tolerance
    goal = env._v7_goal_state(state)
    env._previous_goal_cost = goal["goal_cost"] + 0.5

    reward, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.array([1.0, 0.5, 0.0]),
        physical_action=np.zeros(3),
        info={"success": False, "safety_violation_observed": False},
        terminated=False,
        truncated=False,
    )

    np.testing.assert_allclose(components["goal_progress_raw"], 0.5)
    np.testing.assert_allclose(components["goal_progress_clipped"], 0.2)
    np.testing.assert_allclose(components["reward_goal_progress"], 2.0)
    np.testing.assert_allclose(
        components["reward_goal_state"],
        -0.05 * goal["goal_cost"],
    )
    np.testing.assert_allclose(components["reward_action"], -0.00125)
    np.testing.assert_allclose(reward, components["reward_total"])


def test_v7_penalizes_only_entry_into_nonterminal_unsafe_set():
    env = ClosingSafetyExplorationV7GymEnv(mode="testing", seed=103)
    env.reset(seed=103)
    state = env.core_env.state.copy()

    def reward_components(unsafe):
        _, components = env._compute_sac_reward(
            state=state,
            normalized_action=np.zeros(3),
            physical_action=np.zeros(3),
            info={
                "success": False,
                "safety_violation_observed": unsafe,
            },
            terminated=False,
            truncated=False,
        )
        return components

    first_unsafe = reward_components(True)
    continuing_unsafe = reward_components(True)
    reward_components(False)
    second_entry = reward_components(True)

    assert first_unsafe["safety_unsafe_entry"] is True
    assert first_unsafe["reward_failure"] == -25.0
    assert continuing_unsafe["safety_unsafe_entry"] is False
    assert continuing_unsafe["reward_failure"] == 0.0
    assert second_entry["safety_unsafe_entry"] is True
    assert second_entry["reward_failure"] == -25.0


def test_v8_progress_is_not_clipped():
    env = ClosingSafetyExplorationV8GymEnv(mode="testing", seed=107)
    env.reset(seed=107)
    state = env.core_env.p3_target + env.core_env.p3_target_tolerance
    goal = env._v8_goal_state(state)
    env._previous_goal_cost = goal["goal_cost"] + 0.5

    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"success": False, "safety_violation_observed": False},
        terminated=False,
        truncated=False,
    )

    assert env.progress_clip is None
    np.testing.assert_allclose(components["goal_progress"], 0.5)
    np.testing.assert_allclose(components["reward_goal_progress"], 5.0)
    assert "goal_progress_clipped" not in components


def test_v8_penalizes_every_nonterminal_unsafe_step():
    env = ClosingSafetyExplorationV8GymEnv(mode="testing", seed=109)
    env.reset(seed=109)
    state = env.core_env.state.copy()

    def reward_components(unsafe):
        _, components = env._compute_sac_reward(
            state=state,
            normalized_action=np.zeros(3),
            physical_action=np.zeros(3),
            info={
                "success": False,
                "safety_violation_observed": unsafe,
            },
            terminated=False,
            truncated=False,
        )
        return components

    first_unsafe = reward_components(True)
    continuing_unsafe = reward_components(True)
    safe = reward_components(False)

    assert first_unsafe["unsafe_occupancy_penalty_active"] is True
    assert first_unsafe["reward_failure"] == -0.1
    assert continuing_unsafe["unsafe_occupancy_penalty_active"] is True
    assert continuing_unsafe["reward_failure"] == -0.1
    assert safe["unsafe_occupancy_penalty_active"] is False
    assert safe["reward_failure"] == 0.0


def test_v9_terminal_velocity_gate_is_off_at_p2_and_on_near_p3():
    env = ClosingSafetyExplorationV9GymEnv(mode="testing", seed=113)
    env.reset(seed=113)

    far_goal = env._v9_goal_state(env.core_env.state)
    near_state = env.core_env.p3_target.copy()
    near_state[0] += 100.0
    near_goal = env._v9_goal_state(near_state)

    assert far_goal["position_error"] > 1000.0
    assert far_goal["terminal_velocity_gate"] < 0.001
    assert near_goal["terminal_velocity_gate"] > 0.95


def test_v9_position_and_attitude_progress_remain_active_far_from_p3():
    env = ClosingSafetyExplorationV9GymEnv(mode="testing", seed=127)
    env.reset(seed=127)
    initial = env.core_env.state.copy()
    target = env.core_env.p3_target
    state = target + 0.5 * (initial - target)
    attitude_error = env.core_env.wrap_to_pi(initial[4] - target[4])
    state[4] = env.core_env.wrap_to_pi(target[4] + 0.5 * attitude_error)

    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"success": False, "safety_violation_observed": False},
        terminated=False,
        truncated=False,
    )

    assert components["position_error"] > 500.0
    assert components["reward_position_progress"] > 0.0
    assert components["reward_attitude_progress"] > 0.0


def test_v9_velocity_progress_is_computed_before_distance_gate():
    def velocity_progress_components(position_error):
        env = ClosingSafetyExplorationV9GymEnv(mode="testing", seed=131)
        env.reset(seed=131)
        state = env.core_env.p3_target.copy()
        state[0] += position_error
        state[2] += env.core_env.p3_target_tolerance[2]
        goal = env._v9_goal_state(state)
        env._previous_position_cost = goal["position_cost"]
        env._previous_attitude_cost = goal["attitude_cost"]
        env._previous_velocity_cost = goal["velocity_cost"] + 1.0
        _, components = env._compute_sac_reward(
            state=state,
            normalized_action=np.zeros(3),
            physical_action=np.zeros(3),
            info={"success": False, "safety_violation_observed": False},
            terminated=False,
            truncated=False,
        )
        return components

    far = velocity_progress_components(1000.0)
    near = velocity_progress_components(100.0)

    np.testing.assert_allclose(far["velocity_progress"], 1.0)
    np.testing.assert_allclose(near["velocity_progress"], 1.0)
    assert far["terminal_velocity_gate"] < 0.01
    assert near["terminal_velocity_gate"] > 0.95
    assert far["reward_terminal_velocity_progress"] < 0.01
    assert near["reward_terminal_velocity_progress"] > 2.0


def test_v10_dense_capture_cost_is_stronger_near_p3_and_zero_at_target():
    env = ClosingSafetyExplorationV10GymEnv(mode="testing", seed=137)
    env.reset(seed=137)

    at_target = env._v10_goal_state(env.core_env.p3_target.copy())
    near_state = env.core_env.p3_target + env.core_env.p3_target_tolerance
    near = env._v10_goal_state(near_state)
    far_state = near_state.copy()
    far_state[0] += 1000.0
    far = env._v10_goal_state(far_state)

    np.testing.assert_allclose(at_target["near_target_capture_cost"], 0.0)
    assert near["near_target_gate"] > 0.95
    assert near["near_target_capture_cost"] > 0.0
    assert far["near_target_gate"] < 0.01

    _, near_components = env._compute_sac_reward(
        state=near_state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"success": False, "safety_violation_observed": False},
        terminated=False,
        truncated=False,
    )
    assert near_components["reward_near_target_capture"] < 0.0


def test_v10_penalizes_only_axial_error_beyond_p3_overshoot_boundary():
    env = ClosingSafetyExplorationV10GymEnv(mode="testing", seed=139)
    env.reset(seed=139)
    tolerance = env.core_env.p3_target_tolerance[0]

    inside_state = env.core_env.p3_target.copy()
    inside_state[0] -= tolerance
    overshot_state = env.core_env.p3_target.copy()
    overshot_state[0] -= 2.0 * tolerance

    inside = env._v10_goal_state(inside_state)
    overshot = env._v10_goal_state(overshot_state)

    assert inside["axial_overshoot_m"] == 0.0
    assert inside["axial_overshoot_cost"] == 0.0
    np.testing.assert_allclose(overshot["axial_overshoot_ratio"], 1.0)
    np.testing.assert_allclose(overshot["axial_overshoot_cost"], np.log(2.0))

    _, components = env._compute_sac_reward(
        state=overshot_state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"success": False, "safety_violation_observed": False},
        terminated=False,
        truncated=False,
    )
    np.testing.assert_allclose(components["reward_axial_overshoot"], -np.log(2.0))


def test_v10_penalizes_near_target_flyaway_but_not_approach():
    env = ClosingSafetyExplorationV10GymEnv(mode="testing", seed=149)
    env.reset(seed=149)

    def goal_with_radial_velocity(velocity_mps):
        state = env.core_env.p3_target.copy()
        state[0] += 100.0
        state[2] += velocity_mps
        return env._v10_goal_state(state)

    approaching = goal_with_radial_velocity(-0.05)
    stationary = goal_with_radial_velocity(0.0)
    departing = goal_with_radial_velocity(0.05)

    assert approaching["radial_velocity_mps"] < 0.0
    assert approaching["flyaway_cost"] == 0.0
    assert stationary["flyaway_cost"] == 0.0
    assert departing["radial_velocity_mps"] > 0.0
    assert departing["flyaway_cost"] > 0.0

    _, components = env._compute_sac_reward(
        state=(
            env.core_env.p3_target
            + np.array([100.0, 0.0, 0.05, 0.0, 0.0, 0.0])
        ),
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"success": False, "safety_violation_observed": False},
        terminated=False,
        truncated=False,
    )
    assert components["reward_flyaway"] < 0.0
    np.testing.assert_allclose(
        components["reward_total"],
        sum(
            value
            for key, value in components.items()
            if key.startswith("reward_")
            and key not in {"reward_scheme", "reward_total"}
        ),
    )


def test_v11_stationary_near_target_has_no_additional_occupancy_penalty():
    env = ClosingSafetyExplorationV11GymEnv(mode="testing", seed=151)
    env.reset(seed=151)
    state = env.core_env.p3_target + env.core_env.p3_target_tolerance
    goal = env._v11_goal_state(state)
    env._previous_capture_cost = goal["capture_cost"]

    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"success": False, "safety_violation_observed": False},
        terminated=False,
        truncated=False,
    )

    assert components["near_target_gate"] > 0.95
    assert components["capture_progress_raw"] == 0.0
    assert components["reward_capture_progress"] == 0.0
    assert components["near_target_occupancy_penalty_active"] is False
    assert "reward_near_target_capture" not in components


def test_v11_capture_progress_and_regression_are_symmetrically_bounded():
    def capture_reward(previous_cost_offset):
        env = ClosingSafetyExplorationV11GymEnv(mode="testing", seed=157)
        env.reset(seed=157)
        state = env.core_env.p3_target + env.core_env.p3_target_tolerance
        goal = env._v11_goal_state(state)
        env._previous_capture_cost = goal["capture_cost"] + previous_cost_offset
        _, components = env._compute_sac_reward(
            state=state,
            normalized_action=np.zeros(3),
            physical_action=np.zeros(3),
            info={"success": False, "safety_violation_observed": False},
            terminated=False,
            truncated=False,
        )
        return components

    improvement = capture_reward(1.0)
    regression = capture_reward(-1.0)

    assert improvement["capture_progress_clipped"] == 0.1
    assert regression["capture_progress_clipped"] == -0.1
    assert 0.0 < improvement["reward_capture_progress"] <= 0.1
    assert -0.1 <= regression["reward_capture_progress"] < 0.0
    np.testing.assert_allclose(
        improvement["reward_capture_progress"],
        -regression["reward_capture_progress"],
    )


def test_v11_axial_overshoot_is_diagnostic_only():
    env = ClosingSafetyExplorationV11GymEnv(mode="testing", seed=163)
    env.reset(seed=163)
    state = env.core_env.p3_target.copy()
    state[0] -= 2.0 * env.core_env.p3_target_tolerance[0]

    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"success": False, "safety_violation_observed": False},
        terminated=False,
        truncated=False,
    )

    assert components["axial_overshoot_m"] > 0.0
    assert components["axial_overshoot_penalty_active"] is False
    assert "reward_axial_overshoot" not in components


def test_v11_flyaway_penalty_is_bounded_to_unsafe_step_scale():
    env = ClosingSafetyExplorationV11GymEnv(mode="testing", seed=167)
    env.reset(seed=167)
    state = env.core_env.p3_target.copy()
    state[0] += 100.0
    state[2] += 10.0
    goal = env._v11_goal_state(state)
    env._previous_capture_cost = goal["capture_cost"]

    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"success": False, "safety_violation_observed": False},
        terminated=False,
        truncated=False,
    )

    assert components["flyaway_cost_raw"] > 0.1
    assert components["flyaway_cost_clipped"] == 0.1
    assert components["reward_flyaway"] == -0.1
    assert abs(components["reward_flyaway"]) == env.reward_weights["unsafe_step"]


def test_v12_balanced_capture_progress_uses_requested_group_weights():
    env = ClosingSafetyExplorationV12GymEnv(mode="testing", seed=173)
    env.reset(seed=173)
    state = env.core_env.p3_target + env.core_env.p3_target_tolerance
    goal = env._v12_goal_state(state)
    env._previous_capture_position_cost = goal["position_cost"] + 1.0
    env._previous_capture_attitude_cost = goal["attitude_cost"] + 1.0
    env._previous_capture_velocity_cost = goal["velocity_cost"] + 1.0

    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"success": False, "safety_violation_observed": False},
        terminated=False,
        truncated=False,
    )

    assert env.position_group_weight == 0.4
    assert env.attitude_group_weight == 0.4
    assert env.velocity_group_weight == 0.2
    np.testing.assert_allclose(
        components["capture_balanced_progress_raw"],
        1.0,
    )
    assert components["capture_balanced_progress_clipped"] == 0.1
    assert 0.0 < components["reward_capture_progress"] <= 0.1
    assert "reward_near_target_capture" not in components
    assert "reward_axial_overshoot" not in components


def test_v12_combined_capture_regression_is_bounded_to_unsafe_step_scale():
    env = ClosingSafetyExplorationV12GymEnv(mode="testing", seed=179)
    env.reset(seed=179)
    state = env.core_env.p3_target + env.core_env.p3_target_tolerance
    goal = env._v12_goal_state(state)
    env._previous_capture_position_cost = goal["position_cost"] - 1.0
    env._previous_capture_attitude_cost = goal["attitude_cost"] - 1.0
    env._previous_capture_velocity_cost = goal["velocity_cost"] - 1.0

    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"success": False, "safety_violation_observed": False},
        terminated=False,
        truncated=False,
    )

    assert components["capture_balanced_progress_clipped"] == -0.1
    assert -0.1 <= components["reward_capture_progress"] < 0.0
    assert abs(components["reward_capture_progress"]) <= env.reward_weights[
        "unsafe_step"
    ]


def test_v12_flyaway_penalty_uses_reduced_bound():
    env = ClosingSafetyExplorationV12GymEnv(mode="testing", seed=181)
    env.reset(seed=181)
    state = env.core_env.p3_target.copy()
    state[0] += 100.0
    state[2] += 10.0
    goal = env._v12_goal_state(state)
    env._previous_capture_position_cost = goal["position_cost"]
    env._previous_capture_attitude_cost = goal["attitude_cost"]
    env._previous_capture_velocity_cost = goal["velocity_cost"]

    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"success": False, "safety_violation_observed": False},
        terminated=False,
        truncated=False,
    )

    assert components["flyaway_cost_raw"] > 0.05
    assert components["flyaway_cost_clipped"] == 0.05
    assert components["reward_flyaway"] == -0.05


def test_v12_failed_approach_requires_entry_loss_and_patience():
    env = ClosingSafetyExplorationV12GymEnv(mode="testing", seed=191)
    env.reset(seed=191)
    near_state = env.core_env.p3_target.copy()
    near_state[0] += 100.0
    assert env._update_failed_approach_state(
        env._v12_goal_state(near_state)
    ) is False
    assert env._entered_near_target is True

    departing_state = env.core_env.p3_target.copy()
    departing_state[0] += 301.0
    departing_state[2] += 0.02
    departing_goal = env._v12_goal_state(departing_state)
    for _ in range(env.failed_approach_patience_steps - 1):
        assert env._update_failed_approach_state(departing_goal) is False
    assert env._update_failed_approach_state(departing_goal) is True


def test_v12_failed_approach_is_zero_penalty_truncation():
    env = ClosingSafetyExplorationV12GymEnv(mode="testing", seed=193)
    env.reset(seed=193)
    env._update_failed_approach_state = lambda goal: True

    _, _, terminated, truncated, info = env.step(np.zeros(3))

    assert terminated is False
    assert truncated is True
    assert info["failed_approach_reset"] is True
    assert info["termination_reason"] == "failed_approach_reset"
    assert info["reward_components"]["failed_approach_reset_reward"] == 0.0


def test_v13_components_inside_tolerance_have_zero_capture_deficit():
    env = ClosingSafetyExplorationV13GymEnv(mode="testing", seed=197)
    env.reset(seed=197)
    state = env.core_env.p3_target + env.core_env.p3_target_tolerance

    goal = env._v13_goal_state(state)

    np.testing.assert_allclose(goal["capture_component_deficits"], 0.0)
    assert goal["capture_position_deficit"] == 0.0
    assert goal["capture_velocity_deficit"] == 0.0
    assert goal["capture_attitude_deficit"] == 0.0
    assert goal["capture_cost"] == 0.0


def test_v13_nominal_weights_shift_capture_away_from_alignment():
    env = ClosingSafetyExplorationV13GymEnv(mode="testing", seed=199)
    env.reset(seed=199)
    target = env.core_env.p3_target
    tolerance = env.core_env.p3_target_tolerance

    def capture_cost(component_index):
        state = target.copy()
        state[component_index] += 10.0 * tolerance[component_index]
        return env._v13_goal_state(state)["capture_cost"]

    position_cost = capture_cost(0)
    velocity_cost = capture_cost(2)
    attitude_cost = capture_cost(4)

    assert env.capture_position_group_weight == 0.60
    assert env.capture_velocity_group_weight == 0.30
    assert env.capture_attitude_group_weight == 0.10
    assert position_cost > velocity_cost > attitude_cost > 0.0
    # The unweighted smooth-worst contribution keeps even the low-nominal-
    # weight alignment group material when it is the sole violated group.
    assert attitude_cost > 0.5 * position_cost


def test_v13_worst_unsatisfied_capture_progress_is_symmetrically_bounded():
    def capture_components(previous_cost_offset):
        env = ClosingSafetyExplorationV13GymEnv(mode="testing", seed=211)
        env.reset(seed=211)
        state = env.core_env.p3_target.copy()
        state[0] += 10.0 * env.core_env.p3_target_tolerance[0]
        goal = env._v13_goal_state(state)
        env._previous_v13_capture_cost = (
            goal["capture_cost"] + previous_cost_offset
        )
        _, components = env._compute_sac_reward(
            state=state,
            normalized_action=np.zeros(3),
            physical_action=np.zeros(3),
            info={"success": False, "safety_violation_observed": False},
            terminated=False,
            truncated=False,
        )
        return components

    improvement = capture_components(1.0)
    regression = capture_components(-1.0)

    assert improvement["capture_progress_clipped"] == 0.1
    assert regression["capture_progress_clipped"] == -0.1
    assert 0.0 < improvement["reward_capture_progress"] <= 0.1
    assert -0.1 <= regression["reward_capture_progress"] < 0.0
    np.testing.assert_allclose(
        improvement["reward_capture_progress"],
        -regression["reward_capture_progress"],
    )
    assert "capture_balanced_progress_raw" not in improvement
    assert improvement["near_target_occupancy_penalty_active"] is False
    assert improvement["axial_overshoot_penalty_active"] is False


def test_v13_preserves_v12_flyaway_and_failed_approach_bounds():
    env = ClosingSafetyExplorationV13GymEnv(mode="testing", seed=223)
    env.reset(seed=223)
    state = env.core_env.p3_target.copy()
    state[0] += 100.0
    state[2] += 10.0
    goal = env._v13_goal_state(state)
    env._previous_v13_capture_cost = goal["capture_cost"]

    _, components = env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"success": False, "safety_violation_observed": False},
        terminated=False,
        truncated=False,
    )

    assert env.reward_scheme == "p3_reach_v13"
    assert components["reward_scheme"] == "p3_reach_v13"
    assert components["flyaway_cost_clipped"] == 0.05
    assert components["reward_flyaway"] == -0.05
    assert env.failed_approach_entry_distance_m == 400.0
    assert env.failed_approach_distance_growth_m == 200.0


def test_v14_matches_v12_reward_inside_angle_rate_safe_band():
    v12 = ClosingSafetyExplorationV12GymEnv(mode="testing", seed=227)
    v14 = ClosingSafetyExplorationV14GymEnv(mode="testing", seed=227)
    v12.reset(seed=227)
    v14.reset(seed=227)
    state = v12.core_env.p3_target.copy()
    state[0] += 100.0
    state[4] = np.deg2rad(3.0)
    state[5] = np.deg2rad(0.15)
    v12_goal = v12._v12_goal_state(state)
    v14_goal = v14._v12_goal_state(state)
    for env, goal in ((v12, v12_goal), (v14, v14_goal)):
        env._previous_capture_position_cost = goal["position_cost"]
        env._previous_capture_attitude_cost = goal["attitude_cost"]
        env._previous_capture_velocity_cost = goal["velocity_cost"]

    v12_reward, _ = v12._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"success": False, "safety_violation_observed": False},
        terminated=False,
        truncated=False,
    )
    v14_reward, components = v14._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"success": False, "safety_violation_observed": False},
        terminated=False,
        truncated=False,
    )

    np.testing.assert_allclose(v14_reward, v12_reward)
    assert components["reward_angle_constraint"] == 0.0
    assert components["angle_constraint_combined_warning_depth"] == 0.0


def test_v14_angle_penalty_rises_through_warning_band_and_is_bounded():
    env = ClosingSafetyExplorationV14GymEnv(mode="testing", seed=229)
    env.reset(seed=229)

    def components_at(psi_deg, omega_degs=0.0):
        state = env.core_env.p3_target.copy()
        state[4] = np.deg2rad(psi_deg)
        state[5] = np.deg2rad(omega_degs)
        goal = env._v12_goal_state(state)
        env._previous_capture_position_cost = goal["position_cost"]
        env._previous_capture_attitude_cost = goal["attitude_cost"]
        env._previous_capture_velocity_cost = goal["velocity_cost"]
        _, components = env._compute_sac_reward(
            state=state,
            normalized_action=np.zeros(3),
            physical_action=np.zeros(3),
            info={"success": False, "safety_violation_observed": False},
            terminated=False,
            truncated=False,
        )
        return components

    safe = components_at(3.0)
    warning = components_at(3.5)
    unsafe = components_at(4.0)
    beyond_unsafe = components_at(20.0)

    assert safe["reward_angle_constraint"] == 0.0
    np.testing.assert_allclose(
        warning["angle_constraint_psi_warning_depth"],
        0.5,
    )
    np.testing.assert_allclose(warning["reward_angle_constraint"], -0.05)
    assert unsafe["reward_angle_constraint"] == -0.1
    assert beyond_unsafe["reward_angle_constraint"] == -0.1


def test_v14_angle_and_rate_depths_use_bounded_union():
    env = ClosingSafetyExplorationV14GymEnv(mode="testing", seed=233)
    env.reset(seed=233)
    state = env.core_env.p3_target.copy()
    state[4] = np.deg2rad(3.5)
    state[5] = np.deg2rad(0.175)
    constraint = env._v14_angle_constraint(state)

    np.testing.assert_allclose(constraint["psi_warning_depth"], 0.5)
    np.testing.assert_allclose(constraint["omega_warning_depth"], 0.5)
    np.testing.assert_allclose(constraint["combined_warning_depth"], 0.75)
    assert env.reward_weights["angle_constraint"] == 0.1
    assert env.reward_scheme == "p3_reach_v14"


def _prepare_v12_capture_history(env, state):
    goal = env._v12_goal_state(state)
    env._previous_capture_position_cost = goal["position_cost"]
    env._previous_capture_attitude_cost = goal["attitude_cost"]
    env._previous_capture_velocity_cost = goal["velocity_cost"]


def _reward_at_state(env, state):
    _prepare_v12_capture_history(env, state)
    return env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"success": False, "safety_violation_observed": False},
        terminated=False,
        truncated=False,
    )


def test_v15_matches_v12_reward_inside_docking_cone():
    v12 = ClosingSafetyExplorationV12GymEnv(mode="testing", seed=239)
    v15 = ClosingSafetyExplorationV15GymEnv(mode="testing", seed=239)
    v12.reset(seed=239)
    v15.reset(seed=239)
    state = v12.core_env.p3_target.copy()
    state[0] += 100.0

    v12_reward, _ = _reward_at_state(v12, state)
    v15_reward, components = _reward_at_state(v15, state)

    np.testing.assert_allclose(v15_reward, v12_reward)
    assert components["cone_constraint_warning_depth"] == 0.0
    assert components["reward_cone_constraint"] == 0.0


def test_v15_cone_penalty_uses_bounded_lateral_exit_depth():
    v12 = ClosingSafetyExplorationV12GymEnv(mode="testing", seed=241)
    v15 = ClosingSafetyExplorationV15GymEnv(mode="testing", seed=241)
    v12.reset(seed=241)
    v15.reset(seed=241)
    state = v12.core_env.p3_target.copy()
    q = v12.core_env.get_derived_quantities(state)
    cone_radius = v12.core_env.cone_slope * q["s"]
    state[1] += cone_radius + 50.0 - q["r_p"][1]

    v12_reward, _ = _reward_at_state(v12, state)
    v15_reward, components = _reward_at_state(v15, state)

    np.testing.assert_allclose(
        components["cone_constraint_warning_depth"],
        0.25,
    )
    np.testing.assert_allclose(components["reward_cone_constraint"], -0.025)
    np.testing.assert_allclose(v15_reward, v12_reward - 0.025)
    assert components["cone_constraint_non_terminal"] is True
    assert components["cone_constraint_lateral_only"] is True

    state[1] += 300.0
    _, clipped = _reward_at_state(v15, state)
    assert clipped["cone_constraint_warning_depth"] == 1.0
    assert clipped["reward_cone_constraint"] == -0.1


def _state_at_cone_lateral_excess(env, lateral_excess_m):
    state = env.core_env.p3_target.copy()
    q = env.core_env.get_derived_quantities(state)
    cone_radius = env.core_env.cone_slope * q["s"]
    state[1] += cone_radius + lateral_excess_m - q["r_p"][1]
    return state


def test_v16_matches_v12_when_cone_depth_is_unchanged():
    v12 = ClosingSafetyExplorationV12GymEnv(mode="testing", seed=251)
    v16 = ClosingSafetyExplorationV16GymEnv(mode="testing", seed=251)
    v12.reset(seed=251)
    _, reset_info = v16.reset(seed=251)
    state = _state_at_cone_lateral_excess(v16, 50.0)
    v16._previous_cone_exit_depth = 0.25

    v12_reward, _ = _reward_at_state(v12, state)
    v16_reward, components = _reward_at_state(v16, state)

    np.testing.assert_allclose(v16_reward, v12_reward)
    assert reset_info["reward_scheme"] == "p3_reach_v16"
    assert components["cone_current_exit_depth"] == 0.25
    assert components["cone_depth_improvement"] == 0.0
    assert components["reward_cone_progress"] == 0.0
    assert components["cone_occupancy_penalty_active"] is False


def test_v16_penalizes_worsening_and_rewards_recovery_symmetrically():
    v12 = ClosingSafetyExplorationV12GymEnv(mode="testing", seed=257)
    v16 = ClosingSafetyExplorationV16GymEnv(mode="testing", seed=257)
    v12.reset(seed=257)
    v16.reset(seed=257)
    state = _state_at_cone_lateral_excess(v16, 50.0)

    v16._previous_cone_exit_depth = 0.20
    v12_reward, _ = _reward_at_state(v12, state)
    worsening_reward, worsening = _reward_at_state(v16, state)
    np.testing.assert_allclose(worsening["cone_depth_improvement"], -0.05)
    np.testing.assert_allclose(worsening["reward_cone_progress"], -0.05)
    np.testing.assert_allclose(worsening_reward, v12_reward - 0.05)

    recovery_state = _state_at_cone_lateral_excess(v16, 40.0)
    v12_reward, _ = _reward_at_state(v12, recovery_state)
    recovery_reward, recovery = _reward_at_state(v16, recovery_state)
    np.testing.assert_allclose(recovery["cone_depth_improvement"], 0.05)
    np.testing.assert_allclose(recovery["reward_cone_progress"], 0.05)
    np.testing.assert_allclose(recovery_reward, v12_reward + 0.05)


def test_v16_cone_progress_reward_is_clipped_to_safety_scale():
    env = ClosingSafetyExplorationV16GymEnv(mode="testing", seed=263)
    env.reset(seed=263)
    outside_state = _state_at_cone_lateral_excess(env, 100.0)

    env._previous_cone_exit_depth = 0.0
    _, worsening = _reward_at_state(env, outside_state)
    assert worsening["cone_progress_raw"] == -0.5
    assert worsening["reward_cone_progress"] == -0.1

    inside_state = env.core_env.p3_target.copy()
    env._previous_cone_exit_depth = 0.5
    _, recovery = _reward_at_state(env, inside_state)
    assert recovery["cone_progress_raw"] == 0.5
    assert recovery["reward_cone_progress"] == 0.1
    assert env.reward_scheme == "p3_reach_v16"


def _prepare_v12_progress_offset(env, state, offset):
    goal = env._v12_goal_state(state)
    env._previous_position_cost = goal["position_cost"] + offset
    env._previous_attitude_cost = goal["attitude_cost"] + offset
    env._previous_velocity_cost = goal["velocity_cost"] + offset
    env._previous_capture_position_cost = goal["position_cost"] + offset
    env._previous_capture_attitude_cost = goal["attitude_cost"] + offset
    env._previous_capture_velocity_cost = goal["velocity_cost"] + offset


def _reward_with_progress_offset(env, state, offset):
    _prepare_v12_progress_offset(env, state, offset)
    return env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"success": False, "safety_violation_observed": False},
        terminated=False,
        truncated=False,
    )


def test_v17_preserves_v12_positive_progress_inside_cone():
    v12 = ClosingSafetyExplorationV12GymEnv(mode="testing", seed=269)
    v17 = ClosingSafetyExplorationV17GymEnv(mode="testing", seed=269)
    v12.reset(seed=269)
    _, reset_info = v17.reset(seed=269)
    state = v17.core_env.p3_target.copy()
    state[0] += 100.0
    v17._previous_cone_exit_depth = 0.0

    v12_reward, _ = _reward_with_progress_offset(v12, state, 0.01)
    v17_reward, components = _reward_with_progress_offset(v17, state, 0.01)

    np.testing.assert_allclose(v17_reward, v12_reward)
    assert reset_info["reward_scheme"] == "p3_reach_v17"
    assert components["cone_constraint_inside"] is True
    assert components["cone_positive_progress_eligible"] is True
    assert components["cone_suppressed_positive_progress_total"] == 0.0


def test_v17_suppresses_positive_progress_outside_without_recovery():
    v12 = ClosingSafetyExplorationV12GymEnv(mode="testing", seed=271)
    v17 = ClosingSafetyExplorationV17GymEnv(mode="testing", seed=271)
    v12.reset(seed=271)
    v17.reset(seed=271)
    state = _state_at_cone_lateral_excess(v17, 50.0)
    v17._previous_cone_exit_depth = 0.25

    v12_reward, v12_components = _reward_with_progress_offset(
        v12,
        state,
        0.01,
    )
    v17_reward, components = _reward_with_progress_offset(v17, state, 0.01)

    assert components["cone_constraint_inside"] is False
    assert components["cone_recovering"] is False
    assert components["cone_positive_progress_eligible"] is False
    assert components["cone_suppressed_positive_progress_total"] > 0.0
    for name in v17.gated_positive_progress_terms:
        assert v12_components[name] > 0.0
        assert components[f"{name}_before_cone_gate"] > 0.0
        assert components[name] == 0.0
    np.testing.assert_allclose(
        v17_reward,
        v12_reward - components["cone_suppressed_positive_progress_total"],
    )
    assert components["cone_occupancy_penalty_active"] is False
    assert components["cone_additive_progress_reward_active"] is False


def test_v17_preserves_positive_progress_while_recovering():
    v12 = ClosingSafetyExplorationV12GymEnv(mode="testing", seed=277)
    v17 = ClosingSafetyExplorationV17GymEnv(mode="testing", seed=277)
    v12.reset(seed=277)
    v17.reset(seed=277)
    state = _state_at_cone_lateral_excess(v17, 50.0)
    v17._previous_cone_exit_depth = 0.30

    v12_reward, _ = _reward_with_progress_offset(v12, state, 0.01)
    v17_reward, components = _reward_with_progress_offset(v17, state, 0.01)

    np.testing.assert_allclose(v17_reward, v12_reward)
    assert components["cone_constraint_inside"] is False
    assert components["cone_recovering"] is True
    assert components["cone_positive_progress_eligible"] is True
    assert components["cone_suppressed_positive_progress_total"] == 0.0


def test_v17_preserves_negative_progress_outside_cone():
    v12 = ClosingSafetyExplorationV12GymEnv(mode="testing", seed=281)
    v17 = ClosingSafetyExplorationV17GymEnv(mode="testing", seed=281)
    v12.reset(seed=281)
    v17.reset(seed=281)
    state = _state_at_cone_lateral_excess(v17, 50.0)
    v17._previous_cone_exit_depth = 0.25

    v12_reward, v12_components = _reward_with_progress_offset(
        v12,
        state,
        -0.01,
    )
    v17_reward, components = _reward_with_progress_offset(v17, state, -0.01)

    np.testing.assert_allclose(v17_reward, v12_reward)
    assert components["cone_positive_progress_eligible"] is False
    assert components["cone_suppressed_positive_progress_total"] == 0.0
    for name in v17.gated_positive_progress_terms:
        assert v12_components[name] < 0.0
        np.testing.assert_allclose(components[name], v12_components[name])


def test_v17_cone_depth_includes_axial_exits():
    env = ClosingSafetyExplorationV17GymEnv(mode="testing", seed=283)
    env.reset(seed=283)
    state = env.core_env.state.copy()
    q = env.core_env.get_derived_quantities(state)
    state[0] += env.core_env.cone_length + 100.0 - q["s"]

    constraint = env._v17_cone_constraint(state)

    assert constraint["inside"] is False
    np.testing.assert_allclose(constraint["axial_excess"], 100.0)
    np.testing.assert_allclose(constraint["axial_depth"], 0.05)
    np.testing.assert_allclose(constraint["exit_depth"], 0.05)


def _reward_v18_with_axial_progress_offset(env, state, offset):
    goal = env._v12_goal_state(state)
    env._previous_position_cost = goal["position_cost"]
    env._previous_attitude_cost = goal["attitude_cost"]
    env._previous_velocity_cost = goal["velocity_cost"]
    env._previous_capture_position_cost = goal["position_cost"]
    env._previous_capture_attitude_cost = goal["attitude_cost"]
    env._previous_capture_velocity_cost = goal["velocity_cost"]
    axial = env._v18_axial_capture_state(state)
    env._previous_axial_capture_cost = axial["axial_capture_cost"] + offset
    return env._compute_sac_reward(
        state=state,
        normalized_action=np.zeros(3),
        physical_action=np.zeros(3),
        info={"success": False, "safety_violation_observed": False},
        terminated=False,
        truncated=False,
    )


def test_v18_redistributes_v12_group_weights_toward_capture():
    env = ClosingSafetyExplorationV18GymEnv(mode="testing", seed=293)
    _, reset_info = env.reset(seed=293)

    assert reset_info["reward_scheme"] == "p3_reach_v18"
    assert env.position_group_weight == 0.60
    assert env.velocity_group_weight == 0.30
    assert env.attitude_group_weight == 0.10
    np.testing.assert_allclose(
        env.position_group_weight
        + env.velocity_group_weight
        + env.attitude_group_weight,
        1.0,
    )


def test_v18_axial_capture_progress_is_symmetric_and_near_target_gated():
    env = ClosingSafetyExplorationV18GymEnv(mode="testing", seed=307)
    env.reset(seed=307)
    state = env.core_env.p3_target.copy()
    state[0] += 100.0

    _, improving = _reward_v18_with_axial_progress_offset(env, state, 0.02)
    _, worsening = _reward_v18_with_axial_progress_offset(env, state, -0.02)

    np.testing.assert_allclose(
        improving["axial_capture_progress_clipped"],
        0.02,
    )
    np.testing.assert_allclose(
        worsening["axial_capture_progress_clipped"],
        -0.02,
    )
    np.testing.assert_allclose(
        improving["reward_axial_capture_progress"],
        -worsening["reward_axial_capture_progress"],
    )
    assert 0.0 < improving["reward_axial_capture_progress"] < 0.02
    assert improving["axial_capture_occupancy_penalty_active"] is False
    assert improving["axial_capture_overshoot_penalty_active"] is False


def test_v18_axial_capture_progress_is_clipped_to_half_safety_scale():
    env = ClosingSafetyExplorationV18GymEnv(mode="testing", seed=311)
    env.reset(seed=311)
    state = env.core_env.p3_target.copy()

    _, improving = _reward_v18_with_axial_progress_offset(env, state, 1.0)
    _, worsening = _reward_v18_with_axial_progress_offset(env, state, -1.0)

    assert improving["axial_capture_progress_clipped"] == 0.05
    assert worsening["axial_capture_progress_clipped"] == -0.05
    np.testing.assert_allclose(
        improving["reward_axial_capture_progress"],
        -worsening["reward_axial_capture_progress"],
    )
    assert abs(improving["reward_axial_capture_progress"]) <= 0.05
