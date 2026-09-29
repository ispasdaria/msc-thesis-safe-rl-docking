from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TRIAL_DIR = Path(__file__).resolve().parents[1]
SRC_ROOT = TRIAL_DIR.parents[4] / "src"
for path in (TRIAL_DIR, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from systematic_capture_env import SystematicCaptureEnv  # noqa: E402
from systematic_capture_action import LowPassCaptureActionWrapper  # noqa: E402
from systematic_capture_observation import (  # noqa: E402
    FixedCaptureObservationScale,
    MarkovCaptureObservationScale,
)
from systematic_capture_reset_env import SystematicCaptureResetWrapper  # noqa: E402
from systematic_interface import DEFAULT_CONTRACT  # noqa: E402
from systematic_long_range_env import (  # noqa: E402
    SystematicLongRangeEnv,
    SystematicLongRangeResetWrapper,
)
from train_systematic_capture_sac import build_config, make_capture_env  # noqa: E402


def test_validated_p3_sampling_matches_final_controller_testing_box():
    rng = np.random.default_rng(123)
    samples = np.array([DEFAULT_CONTRACT.sample_p3(rng) for _ in range(1_000)])
    assert np.all(samples >= DEFAULT_CONTRACT.p3_lower)
    assert np.all(samples <= DEFAULT_CONTRACT.p3_upper)
    assert all(DEFAULT_CONTRACT.in_validated_p3(state) for state in samples)


def test_full_six_state_handover_target_and_sampling():
    p3 = np.array([300.0, 1.0, -0.03, 0.005, np.deg2rad(0.5), 0.0])
    target = DEFAULT_CONTRACT.intermediate_target(p3)
    np.testing.assert_allclose(
        target,
        [450.0, 1.0, -0.11, 0.005, np.deg2rad(0.5), 0.0],
    )
    state, sampled_target = DEFAULT_CONTRACT.sample_handover_state(
        np.random.default_rng(2), p3
    )
    np.testing.assert_allclose(sampled_target, target)
    assert DEFAULT_CONTRACT.in_handover_set(state, p3)


def test_capture_success_is_box_membership_not_sample_target_tolerance():
    env = SystematicCaptureEnv(max_time=2_000.0, seed=0)
    target = np.array([280.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    valid_but_far_from_target = np.array([320.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    env.reset(
        options={
            "initial_state": valid_but_far_from_target,
            "p3_target": target,
            "mode": "testing",
        }
    )
    done, info = env.core_env.check_done(valid_but_far_from_target)
    assert done and info["success"]
    assert info["termination_reason"] == "success_validated_X_P3"

    outside_box_but_close_to_target = np.array(
        [339.0, 0.0, -0.03, 0.0, 0.0, 0.0]
    )
    target_at_edge = np.array([320.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    env.reset(
        options={
            "initial_state": outside_box_but_close_to_target,
            "p3_target": target_at_edge,
            "mode": "testing",
        }
    )
    done, info = env.core_env.check_done(outside_box_but_close_to_target)
    assert not done and not info["success"]
    env.close()


def test_capture_resets_are_safe_full_state_handoffs_at_tested_distances():
    env = SystematicCaptureResetWrapper(
        SystematicCaptureEnv(max_time=2_000.0, seed=0)
    )
    for index, distance in enumerate((30.0, 60.0, 100.0, 150.0)):
        env.set_capture_distance(distance)
        observation, info = env.reset(seed=100 + index)
        assert observation.shape == (12,)
        assert info["initial_in_handover_set"]
        assert not info["unsafe"]
        assert not DEFAULT_CONTRACT.in_validated_p3(env.core_env.state)
    env.close()


def test_alignment_violation_is_terminal_in_systematic_capture():
    env = SystematicCaptureEnv(max_time=2_000.0, seed=0)
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    state = np.array([450.0, 0.0, -0.11, 0.0, np.deg2rad(4.1), 0.0])
    env.reset(options={"initial_state": state, "p3_target": target, "mode": "testing"})
    done, info = env.core_env.check_done(state)
    assert done and info["failure"]
    assert "attitude" in info["alignment_unsafe_reasons"]
    env.close()


def test_long_range_observation_goal_and_terminal_tolerance_are_XH():
    env = SystematicLongRangeResetWrapper(
        SystematicLongRangeEnv(max_time=12_000.0, seed=0)
    )
    observation, info = env.reset(seed=33)
    assert observation.shape == (12,)
    np.testing.assert_allclose(
        env.core_env.p3_target_tolerance, DEFAULT_CONTRACT.handover_tolerance
    )
    np.testing.assert_allclose(env.core_env.p3_target, info["handover_target"])
    assert not info["unsafe"]
    env.close()


def test_training_environment_starts_directly_at_full_150_m_handover():
    class Args:
        max_time = 2_000.0
        training_seed = 0

    monitored_env = make_capture_env(Args())
    _, info = monitored_env.reset(seed=7)
    assert info["capture_distance_m"] == 150.0
    monitored_env.close()


def test_v5_fixed_observation_scaling_preserves_p3_resolution():
    base = SystematicCaptureResetWrapper(
        SystematicCaptureEnv(max_time=2_000.0, seed=0)
    )
    env = FixedCaptureObservationScale(base)
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    state = np.array([450.0, 3.0, -0.11, 0.01, np.deg2rad(1.0), 0.0])
    observation, _ = env.reset(
        options={
            "initial_state": state,
            "p3_target": target,
            "capture_distance_m": 150.0,
            "mode": "testing",
        }
    )
    assert observation.shape == (12,)
    assert observation.dtype == np.float32
    # Error ry=3 m and vy=0.01 m/s remain visible rather than being divided
    # by failure-dominated adaptive standard deviations.
    assert np.isclose(observation[7], 0.3)
    assert np.isclose(observation[9], 1.0 / 3.0)
    env.close()


def test_tracking_is_persistent_and_positive_progress_is_alignment_gated():
    env = SystematicCaptureEnv(max_time=2_000.0, seed=0)
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    aligned = np.array([450.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    off_track = np.array(
        [450.0, 30.0, -0.09, 0.06, np.deg2rad(3.5), 0.0]
    )
    farther_off_track = np.array(
        [450.0, 60.0, -0.15, 0.12, np.deg2rad(7.0), 0.0]
    )
    env.reset(
        options={"initial_state": aligned, "p3_target": target, "mode": "testing"}
    )
    aligned_terms = env._tracking_terms(aligned)
    off_track_terms = env._tracking_terms(off_track)
    assert aligned_terms["cost"] == 0.0
    assert aligned_terms["progress_gate"] == 1.0
    assert off_track_terms["cost"] > aligned_terms["cost"]
    assert env.progress_gate_floor < off_track_terms["progress_gate"] < 0.3
    assert (
        env._tracking_terms(farther_off_track)["cost"]
        > off_track_terms["cost"]
    )
    # vx is tracked directly to sampled P3 vx; there is no reference schedule.
    assert off_track_terms["errors"][0] == off_track[2] - target[2]
    env.close()


def test_vx_target_is_fixed_but_its_tracking_weight_increases_near_p3():
    env = SystematicCaptureEnv(max_time=2_000.0, seed=0)
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    far = np.array([450.0, 0.0, -0.09, 0.0, 0.0, 0.0])
    near = np.array([330.0, 0.0, -0.09, 0.0, 0.0, 0.0])
    env.reset(options={"initial_state": far, "p3_target": target, "mode": "testing"})
    far_terms = env._tracking_terms(far)
    near_terms = env._tracking_terms(near)
    assert far_terms["errors"][0] == near_terms["errors"][0]
    assert near_terms["vx_weight"] > far_terms["vx_weight"]
    assert near_terms["component_costs"][0] > far_terms["component_costs"][0]
    env.close()


def test_geometry_warning_depth_is_continuous_not_boolean():
    env = SystematicCaptureEnv(max_time=2_000.0, seed=0)
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    env.reset(
        options={"initial_state": target, "p3_target": target, "mode": "testing"}
    )
    geometry = {
        "warn": True,
        "in_S_KOS": False,
        "passed_P3_region": False,
    }
    shallow = {
        "s": 400.0,
        "rho": 44.0,
        "com_distance": 400.0,
    }
    deep = dict(shallow, rho=80.0)
    shallow_depth = env._geometry_warning_depth(shallow, geometry)
    deep_depth = env._geometry_warning_depth(deep, geometry)
    assert 0.0 < shallow_depth < deep_depth < 1.0
    env.close()


def test_v7_does_not_penalize_valid_lower_p3_box_as_passed_target():
    env = SystematicCaptureEnv(max_time=2_000.0, seed=0)
    target = np.array([315.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    state = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    env.reset(
        options={"initial_state": state, "p3_target": target, "mode": "testing"}
    )
    derived = env.core_env.get_derived_quantities(state)
    geometry = env.core_env.monitor_safety(state)["geometry"]
    assert geometry["passed_P3_region"]
    assert geometry["warn"]
    assert env._geometry_warning_depth(derived, geometry) == 0.0
    env.close()


def test_v7_terminal_box_progress_and_unready_inward_cost_are_aligned():
    env = SystematicCaptureEnv(max_time=2_000.0, seed=0)
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    bad = np.array([330.0, 10.0, -0.09, 0.04, np.deg2rad(2.0), 0.0])
    ready = np.array([330.0, 2.0, -0.04, 0.005, np.deg2rad(0.5), 0.0])
    env.reset(options={"initial_state": bad, "p3_target": target, "mode": "testing"})
    bad_box = env._terminal_box_terms(bad)
    ready_box = env._terminal_box_terms(ready)
    bad_readiness = env._terminal_readiness_terms(bad, bad_box["ratios"])
    ready_readiness = env._terminal_readiness_terms(ready, ready_box["ratios"])
    assert ready_box["cost"] < bad_box["cost"]
    assert bad_readiness["cost"] > 0.0
    assert ready_readiness["cost"] == 0.0
    env.close()


def test_v7_stronger_vx_and_box_progress_are_terminal_zone_only():
    env = SystematicCaptureEnv(max_time=2_000.0, seed=0)
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    far = np.array([370.0, 5.0, -0.09, 0.02, np.deg2rad(2.0), 0.0])
    near = far.copy()
    near[0] = 330.0
    env.reset(options={"initial_state": far, "p3_target": target, "mode": "testing"})
    far_tracking = env._tracking_terms(far)
    near_tracking = env._tracking_terms(near)
    far_readiness = env._terminal_readiness_terms(
        far, env._terminal_box_terms(far)["ratios"]
    )
    near_readiness = env._terminal_readiness_terms(
        near, env._terminal_box_terms(near)["ratios"]
    )
    assert far_tracking["vx_weight"] == env.vx_tracking_weight_far
    assert near_tracking["vx_weight"] > far_tracking["vx_weight"]
    np.testing.assert_allclose(
        far_tracking["component_weights"][1:],
        near_tracking["component_weights"][1:],
    )
    assert far_readiness["zone_fraction"] == 0.0
    assert near_readiness["zone_fraction"] > 0.0
    env.close()


def test_crossing_inside_near_p3_face_is_immediate_failed_handover():
    env = SystematicCaptureResetWrapper(
        SystematicCaptureEnv(max_time=2_000.0, seed=0)
    )
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    state = np.array([279.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    env.reset(
        options={
            "initial_state": state,
            "p3_target": target,
            "capture_distance_m": 150.0,
            "mode": "testing",
        }
    )
    _, reward, terminated, truncated, info = env.step(np.zeros(3))
    assert terminated and not truncated
    assert reward < -400.0
    assert info["termination_reason"] == "failed_capture_missed_p3"
    assert info["reward_components"]["reward_missed_handover"] == -500.0
    np.testing.assert_allclose(info["terminal_physical_state"], env.core_env.state)
    np.testing.assert_allclose(info["terminal_p3_target"], target)
    env.close()


def test_v8_lateral_divergence_terminates_early_with_scaled_failure_penalty():
    env = SystematicCaptureResetWrapper(
        SystematicCaptureEnv(max_time=2_000.0, seed=0),
        enable_lateral_divergence_termination=True,
    )
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    state = np.array([400.0, 51.0, -0.03, 0.0, 0.0, 0.0])
    env.reset(
        options={
            "initial_state": state,
            "p3_target": target,
            "capture_distance_m": 150.0,
            "mode": "testing",
        }
    )
    _, reward, terminated, truncated, info = env.step(np.zeros(3))
    assert terminated and not truncated
    assert reward < -400.0
    assert info["termination_reason"] == "failed_capture_lateral_divergence"
    assert "lateral_position" in info["lateral_divergence_reasons"]
    assert info["reward_components"]["reward_lateral_divergence"] == -500.0
    env.close()


def test_v10_couples_lateral_energy_and_penalizes_action_changes():
    env = SystematicCaptureEnv(max_time=2_000.0, seed=0)
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    state = np.array([450.0, 3.0, -0.09, 0.01, 0.0, 0.0])
    env.reset(
        options={"initial_state": state, "p3_target": target, "mode": "testing"}
    )
    tracking = env._tracking_terms(state)
    lateral = env._lateral_energy_terms(tracking["errors"])
    assert lateral["energy"] > 0.0
    assert 0.0 < lateral["bounded_energy"] < 1.0

    _, _, terminated, truncated, first_info = env.step(np.ones(3))
    assert not terminated and not truncated
    assert first_info["reward_components"]["reward_action_delta"] < 0.0
    _, _, _, _, second_info = env.step(-np.ones(3))
    assert second_info["reward_components"]["reward_action_delta"] < 0.0
    env.close()


def test_v10_attitude_margin_precedes_hard_warning():
    env = SystematicCaptureEnv(max_time=2_000.0, seed=0)
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    state = np.array([450.0, 0.0, -0.09, 0.0, np.deg2rad(2.5), 0.0])
    env.reset(
        options={"initial_state": state, "p3_target": target, "mode": "testing"}
    )
    derived = env.core_env.get_derived_quantities(state)
    info = env.core_env.monitor_safety(state)
    warnings = env._warning_costs(derived, info)
    assert warnings["alignment_depth"] == 0.0
    assert 0.0 < warnings["attitude_margin_depth"] < 1.0
    assert warnings["attitude_margin_cost"] > 0.0
    env.close()


def test_v10_filtered_action_is_the_previous_action_in_next_observation():
    base = SystematicCaptureResetWrapper(
        SystematicCaptureEnv(max_time=2_000.0, seed=0)
    )
    observed = MarkovCaptureObservationScale(base)
    env = LowPassCaptureActionWrapper(observed, alpha=0.25)
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    state = np.array([450.0, 0.0, -0.09, 0.0, 0.0, 0.0])
    observation, _ = env.reset(
        options={
            "initial_state": state,
            "p3_target": target,
            "capture_distance_m": 150.0,
            "mode": "testing",
        }
    )
    assert observation.shape == (15,)
    np.testing.assert_allclose(observation[-3:], 0.0)
    action = np.array([1.0, -1.0, 0.5], dtype=np.float32)
    observation, _, _, _, info = env.step(action)
    expected_executed = 0.25 * action
    np.testing.assert_allclose(observation[-3:], expected_executed)
    np.testing.assert_allclose(info["commanded_normalized_action"], action)
    np.testing.assert_allclose(
        info["executed_normalized_action"], expected_executed
    )
    env.close()


def test_v10_outward_velocity_receives_dense_penalty():
    env = SystematicCaptureEnv(max_time=2_000.0, seed=0)
    target = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    state = np.array([400.0, 0.0, 0.02, 0.0, 0.0, 0.0])
    env.reset(
        options={"initial_state": state, "p3_target": target, "mode": "testing"}
    )
    _, _, _, _, info = env.step(np.zeros(3))
    components = info["reward_components"]
    assert components["outward_velocity_mps"] > 0.0
    assert components["reward_outward_velocity"] < 0.0
    env.close()


def test_v10_training_config_uses_filtered_markov_reward_and_fixed_scaling():
    class Args:
        training_seed = 0
        max_time = 2_000.0
        total_timesteps = 600_000
        checkpoint_frequency = 50_000
        device = "auto"

    config = build_config(Args(), "test")
    assert config["total_timesteps"] == 600_000
    assert config["sac"]["buffer_size"] == config["total_timesteps"]
    assert config["reward"]["distance_dependent_vx_reference"] is False
    assert config["reward"]["distance_dependent_vx_weight"] is True
    assert config["reward"]["progress_gate_floor"] > 0.0
    assert "pseudo_huber" in config["reward"]["tracking_cost"]
    assert config["observation_scaling"]["type"] == (
        "fixed_physical_scaling_with_previous_action"
    )
    assert len(config["observation_scaling"]["labels"]) == 15
    assert config["vecnormalize"]["norm_obs"] is False
    assert config["failed_approach_termination"]["lateral_divergence"]
    assert config["reward"]["terminal_box_progress"]["scale"] > 0.0
    assert config["reward"]["terminal_box_progress"][
        "active_only_in_terminal_sync_zone"
    ]
    assert config["reward"]["terminal_readiness"]["penalty_weight"] > 0.0
    assert config["reward"]["terminal_readiness"]["penalty_weight"] == 0.20
    assert config["reward"]["scheme"].endswith("_v10")
    assert config["reward"]["coupled_lateral_energy"]["components"] == [
        "ry",
        "vy",
    ]
    assert config["reward"]["coupled_lateral_energy"][
        "progress_negative_clip"
    ] == config["reward"]["coupled_lateral_energy"]["progress_positive_clip"]
    assert not config["reward"]["coupled_lateral_energy"]["state_cost_bounded"]
    assert config["reward"]["action_delta_weights"][1] > 0.0
    assert config["reward"]["action_delta_weights"][2] > 0.0
    assert config["reward"]["attitude_margin"]["start_deg"] == 2.0
    assert config["reward"]["success_bonus"] == 500.0
    assert config["reward"]["safety_failure_penalty"] == 500.0
    assert config["reward"]["failed_approach_penalty"] == 500.0
    assert config["reward"]["terminal_readiness"][
        "lateral_position_weight"
    ] > 0.0
    assert config["reward"]["outward_velocity"]["weight"] > 0.0
    assert config["action_filter"]["alpha"] == 0.25
    assert config["sac"]["gradient_steps"] == 4
    assert config["reward"]["passed_p3_warning_boundary_rx_m"] == 280.0
    assert config["failed_approach_termination"][
        "inward_below_validated_p3_box"
    ]
