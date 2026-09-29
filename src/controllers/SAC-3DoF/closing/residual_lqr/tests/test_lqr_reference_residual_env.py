from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SRC_PATH = Path(__file__).resolve().parents[5]
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
CLOSING_PATH = Path(__file__).resolve().parents[1]
if str(CLOSING_PATH) not in sys.path:
    sys.path.insert(0, str(CLOSING_PATH))

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.docking3dof.goal_conditioned_closing_env import (
    GoalConditionedClosingV3Env,
)
from evaluate_sac_closing_reference_residual_no_her import (
    environment_kwargs as evaluation_environment_kwargs,
)
from lqr_reference_residual_env import (
    ReferenceResidualLQRGoalConditionedClosingEnv,
)
from her_source_warmstart import prefill_her_replay_buffer
from train_sac_closing_her_reference_residual_lqr import (
    collect_reference_residual_trajectories,
)
from train_sac_closing_her_v4_residual_v15 import (
    initialize_zero_residual_actor,
)
from train_sac_closing_reference_residual_lqr_no_her import (
    prefill_standard_replay_buffer,
)
from train_sac_closing_reference_residual_lqr_pilot_e_dt01 import (
    DEFAULT_ARGUMENTS as PILOT_E_DT01_ARGUMENTS,
)
from train_sac_closing_reference_residual_lqr_pilot_g_dt01_reward_v2 import (
    DEFAULT_ARGUMENTS as PILOT_G_DT01_ARGUMENTS,
)
from train_sac_closing_reference_residual_lqr_stratified import (
    build_parser as build_stratified_parser,
    prepare_output_directories,
    validate_args as validate_stratified_args,
)
from zero_residual_constrained_sac import ZeroResidualConstrainedSAC


def make_env(seed=700, max_time=10.0):
    return ReferenceResidualLQRGoalConditionedClosingEnv(
        mode="testing",
        seed=seed,
        max_time=max_time,
        reference_residual_bounds=(0.02, 0.01, np.deg2rad(1.0)),
        progress_gamma=0.9999,
    )


def test_reference_residual_environment_contract():
    env = make_env(max_time=3.0)
    check_env(env, warn=True)
    observation, info = env.reset(seed=700)

    assert set(observation) == {
        "observation",
        "achieved_goal",
        "desired_goal",
    }
    assert env.action_space.shape == (3,)
    assert info["reward_scheme"] == env.reward_scheme
    env.close()


def test_pilot_e_dt01_defaults_are_physically_time_matched():
    parser = build_stratified_parser()
    args = parser.parse_args(PILOT_E_DT01_ARGUMENTS)
    validate_stratified_args(parser, args)

    assert args.dt == 0.1
    assert args.total_timesteps == 500_000
    assert args.source_transition_count == 600_000
    assert args.prefill_residual_hold_steps == 10
    assert args.critic_warmup_steps == 100_000
    assert args.zero_residual_actor_anneal_end_step == 300_000
    assert args.train_freq == 40
    assert np.isclose(args.gamma, 0.9999**0.1)
    assert args.time_scale_persistent_reward is True


def test_pilot_g_dt01_defaults_select_lateral_time_consistent_reward():
    parser = build_stratified_parser()
    args = parser.parse_args(PILOT_G_DT01_ARGUMENTS)
    validate_stratified_args(parser, args)

    assert args.pilot_label == "G"
    assert args.dt == 0.1
    assert args.reward_design == "dt01_lateral_v2"
    assert args.goal_component_weights == [1.0, 4.0, 1.0, 2.0, 1.0, 1.0]
    assert args.worst_component_weight == 0.75
    assert args.lateral_constraint_progress_weight == 4.0
    assert args.timeout_violation_weight == 2.0
    assert args.time_scale_persistent_reward is True
    assert args.time_consistent_smoothness_reward is True
    assert args.buffer_size == 650_000


def test_dt01_v2_rewards_reducing_lateral_constraint_violation():
    env = ReferenceResidualLQRGoalConditionedClosingEnv(
        mode="testing",
        seed=712,
        dt=0.1,
        max_time=1.0,
        progress_gamma=0.9999899995499715,
        progress_weight=0.0,
        lateral_constraint_progress_weight=4.0,
        reference_magnitude_weight=0.0,
        reference_smoothness_weight=0.0,
        success_bonus=0.0,
        safety_failure_penalty=0.0,
        timeout_penalty=0.0,
        goal_component_weights=(1.0, 4.0, 1.0, 2.0, 1.0, 1.0),
        worst_component_weight=0.75,
        reward_design="dt01_lateral_v2",
    )
    observation, _ = env.reset(seed=712)
    target = env.core_env.p3_target.copy()
    tolerance = env.core_env.p3_target_tolerance
    previous_state = target.copy()
    next_state = target.copy()
    previous_state[1] += 2.0 * tolerance[1]
    next_state[1] += 1.5 * tolerance[1]
    info = {
        "previous_achieved_goal": env._scale_state(previous_state),
        "reference_residual_normalized_action": np.zeros(3),
        "previous_reference_residual_normalized_action": np.zeros(3),
    }

    reward = env.compute_reward(
        env._scale_state(next_state),
        observation["desired_goal"],
        info,
    )

    assert reward > 0.0
    assert env.reward_scheme.endswith("dt01_lateral_v2")
    env.close()


def test_dt01_v2_smoothness_uses_action_rate_scaling():
    env = ReferenceResidualLQRGoalConditionedClosingEnv(
        mode="testing",
        seed=713,
        dt=0.1,
        max_time=1.0,
        time_scale_persistent_reward=True,
        time_consistent_smoothness_reward=True,
        reward_design="dt01_lateral_v2",
    )
    env.reset(seed=713)

    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))

    assert np.isclose(info["persistent_reward_time_scale"], 0.1)
    assert np.isclose(info["smoothness_reward_time_scale"], 10.0)
    env.close()


def test_dt01_v2_saved_reward_config_reconstructs_evaluation_environment():
    config = {
        "dt_seconds": 0.1,
        "reference_residual_bounds": {
            "vx_mps": 0.02,
            "vy_mps": 0.01,
            "attitude_deg": 1.0,
        },
        "sac": {"gamma": 0.9999899995499715},
        "actual_p3_success_terminal": True,
        "terminal_residual_fade": {
            "enabled": True,
            "full_authority_at_or_above_ratio": 1.2,
            "zero_authority_at_or_below_ratio": 1.0,
        },
        "reward": {
            "design": "dt01_lateral_v2",
            "progress_weight": 2.0,
            "goal_component_weights": [1.0, 4.0, 1.0, 2.0, 1.0, 1.0],
            "worst_component_weight": 0.75,
            "lateral_constraint_progress_weight": 4.0,
            "magnitude_weight": 0.002,
            "smoothness_weight": 0.001,
            "success_bonus": 5.0,
            "failure_penalty": 25.0,
            "timeout_penalty": 2.0,
            "timeout_violation_weight": 2.0,
            "persistent_terms_time_scaled": True,
            "smoothness_is_action_rate_cost": True,
        },
    }

    kwargs = evaluation_environment_kwargs(config)
    env = ReferenceResidualLQRGoalConditionedClosingEnv(
        mode="testing",
        seed=714,
        max_time=1.0,
        **kwargs,
    )

    assert env.reward_design == "dt01_lateral_v2"
    assert env.lateral_constraint_progress_weight == 4.0
    assert env.time_consistent_smoothness_reward is True
    assert env.terminal_residual_fade_full_ratio == 1.2
    assert env.terminal_residual_fade_zero_ratio == 1.0
    np.testing.assert_allclose(
        env.goal_component_weights,
        [1.0, 4.0, 1.0, 2.0, 1.0, 1.0],
    )
    env.close()


def test_terminal_residual_fade_scales_applied_reference_action():
    env = ReferenceResidualLQRGoalConditionedClosingEnv(
        mode="testing",
        seed=715,
        dt=0.1,
        max_time=1.0,
        terminal_residual_fade_full_ratio=1.2,
        terminal_residual_fade_zero_ratio=1.0,
    )
    env.reset(seed=715)
    target = np.asarray(env.core_env.p3_target, dtype=np.float64).copy()
    state = target.copy()
    state[1] += 1.1 * env.core_env.p3_target_tolerance[1]
    env.core_env.state = state

    authority, maximum_ratio = env.terminal_residual_authority()
    action = np.asarray([0.8, -0.4, 0.6], dtype=np.float32)
    _, _, _, _, info = env.step(action)

    assert np.isclose(maximum_ratio, 1.1)
    assert np.isclose(authority, 0.5)
    assert np.isclose(info["terminal_residual_authority"], 0.5)
    np.testing.assert_allclose(
        info["reference_residual_commanded_action"],
        action,
    )
    np.testing.assert_allclose(
        info["reference_residual_normalized_action"],
        0.5 * action,
    )
    np.testing.assert_allclose(
        info["reference_residual_physical_offsets"],
        env.reference_residual_bounds * 0.5 * action,
    )
    env.close()


def test_stratified_overwrite_replaces_only_exact_experiment_dirs(tmp_path):
    roots = tuple(tmp_path / name for name in ("models", "results", "logs"))
    paths = tuple(root / "experiment" for root in roots)
    for path in paths:
        path.mkdir(parents=True)
        (path / "stale.txt").write_text("stale", encoding="utf-8")

    prepare_output_directories(paths, roots, overwrite=True)

    assert all(path.is_dir() for path in paths)
    assert all(not (path / "stale.txt").exists() for path in paths)


def test_stratified_overwrite_rejects_output_root_itself(tmp_path):
    root = tmp_path / "models"
    root.mkdir()

    with np.testing.assert_raises(ValueError):
        prepare_output_directories((root,), (root,), overwrite=True)


def test_zero_reference_residual_exactly_reproduces_closing_lqr_transition():
    residual_env = make_env(seed=701)
    reference_env = GoalConditionedClosingV3Env(
        mode="testing",
        seed=701,
        max_time=10.0,
    )
    residual_env.reset(seed=701)
    reference_env.reset(seed=701)
    physical_lqr_action = residual_env.lqr_controller.compute_action(
        residual_env.core_env,
        residual_env.core_env.state,
    )
    limits = residual_env.env.get_physical_action_limits()
    expected_normalized = residual_env.env.physical_action_to_normalized(
        np.clip(physical_lqr_action, -limits, limits)
    )

    _, _, _, _, info = residual_env.step(np.zeros(3, dtype=np.float32))
    reference_env.step(expected_normalized)

    np.testing.assert_allclose(
        residual_env.core_env.state,
        reference_env.core_env.state,
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        info["executed_normalized_action"],
        expected_normalized,
        atol=1e-7,
    )
    assert info["cbf_filter_applied"] is False
    residual_env.close()
    reference_env.close()


def test_dt01_zero_residual_uses_retuned_lqr_and_scaled_persistent_costs():
    residual_env = ReferenceResidualLQRGoalConditionedClosingEnv(
        mode="testing",
        seed=709,
        dt=0.1,
        max_time=1.0,
        reference_residual_bounds=(0.02, 0.01, np.deg2rad(1.0)),
        progress_gamma=0.9999899995499715,
        time_scale_persistent_reward=True,
    )
    reference_env = GoalConditionedClosingV3Env(
        mode="testing",
        seed=709,
        dt=0.1,
        max_time=1.0,
    )
    residual_env.reset(seed=709)
    reference_env.reset(seed=709)
    expected = residual_env.lqr_action_for_reference_residual(np.zeros(3))[2]

    _, _, _, _, info = residual_env.step(np.zeros(3, dtype=np.float32))
    reference_env.step(expected)

    np.testing.assert_allclose(
        residual_env.core_env.state,
        reference_env.core_env.state,
        rtol=0.0,
        atol=1e-10,
    )
    assert np.isclose(info["sac_decision_interval_seconds"], 0.1)
    assert np.isclose(info["lqr_update_interval_seconds"], 0.1)
    assert np.isclose(info["persistent_reward_time_scale"], 0.1)
    residual_env.close()
    reference_env.close()


def test_dt01_prefill_horizon_is_interpreted_in_physical_seconds():
    transitions, summaries = collect_reference_residual_trajectories(
        minimum_transitions=3,
        seed_start=710,
        max_episodes=1,
        max_steps=0.3,
        env_kwargs={
            "dt": 0.1,
            "reference_residual_bounds": (
                0.02,
                0.01,
                np.deg2rad(1.0),
            ),
            "progress_gamma": 0.9999899995499715,
        },
        residual_std=0.0,
        residual_clip=0.5,
        zero_episode_fraction=1.0,
        random_seed=710,
        retain_failure_episodes=True,
    )

    assert len(transitions) == 3
    assert summaries[0]["steps"] == 3


def test_sac_action_offsets_only_vx_vy_and_attitude_references():
    env = make_env(seed=702)
    env.reset(seed=702)
    action = np.array([0.5, -0.25, 1.0], dtype=np.float32)

    _, reward, _, _, info = env.step(action)

    expected_offsets = np.array(
        [0.01, -0.0025, np.deg2rad(1.0)],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        info["reference_residual_physical_offsets"],
        expected_offsets,
    )
    reference_change = (
        info["modified_lqr_reference"] - info["nominal_lqr_reference"]
    )
    np.testing.assert_allclose(reference_change[[0, 1, 5]], 0.0, atol=1e-15)
    np.testing.assert_allclose(reference_change[2:4], expected_offsets[:2])
    assert np.isclose(reference_change[4], expected_offsets[2])
    assert np.isclose(reward, info["her_reward"])
    env.close()


def test_progress_reward_recomputes_for_batched_her_goals():
    env = make_env(seed=703)
    observation, _ = env.reset(seed=703)
    previous = observation["achieved_goal"].copy()
    next_achieved = previous.copy()
    next_achieved[0] -= 1e-3
    desired = observation["desired_goal"].copy()
    info = {
        "previous_achieved_goal": previous,
        "reference_residual_normalized_action": np.zeros(3),
        "previous_reference_residual_normalized_action": np.zeros(3),
    }

    scalar = env.compute_reward(next_achieved, desired, info)
    batched = env.compute_reward(
        np.stack([next_achieved, next_achieved]),
        np.stack([desired, next_achieved]),
        [info, info],
    )

    assert np.isfinite(scalar)
    assert batched.shape == (2,)
    assert np.all(np.isfinite(batched))
    assert batched[1] > batched[0]
    env.close()


def test_reference_residual_prefill_and_sac_update_are_finite():
    env_kwargs = {
        "reference_residual_bounds": (0.02, 0.01, np.deg2rad(1.0)),
        "progress_gamma": 0.9999,
    }
    transitions, summaries = collect_reference_residual_trajectories(
        minimum_transitions=4,
        seed_start=704,
        max_episodes=3,
        max_steps=2,
        env_kwargs=env_kwargs,
        residual_std=0.2,
        residual_clip=0.5,
        zero_episode_fraction=0.0,
        random_seed=704,
        retain_failure_episodes=True,
    )
    assert len(transitions) >= 4
    assert all(summary["accepted"] == 1 for summary in summaries)
    assert np.max(
        np.abs(np.stack([item["action"] for item in transitions]))
    ) <= 0.5

    vec_env = DummyVecEnv(
        [
            lambda: Monitor(
                ReferenceResidualLQRGoalConditionedClosingEnv(
                    mode="testing",
                    seed=705,
                    max_time=2.0,
                    **env_kwargs,
                )
            )
        ]
    )
    model = ZeroResidualConstrainedSAC(
        "MultiInputPolicy",
        vec_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": 1,
            "goal_selection_strategy": "future",
            "copy_info_dict": True,
        },
        buffer_size=100,
        learning_starts=0,
        batch_size=2,
        train_freq=1,
        gradient_steps=1,
        critic_warmup_steps=2,
        zero_residual_actor_weight=1.0,
        actor_learning_rate_scale=0.25,
        policy_kwargs={"net_arch": [16, 16]},
        verbose=0,
        seed=705,
    )
    initialize_zero_residual_actor(model, log_std=-3.0)
    prefill_her_replay_buffer(model, transitions)
    model.learn(total_timesteps=6)
    action, _ = model.predict(vec_env.reset(), deterministic=True)

    assert np.all(np.isfinite(action))
    assert model.critic_warmup_gradient_updates > 0
    assert model.constrained_actor_gradient_updates > 0
    vec_env.close()


def test_fixed_goal_standard_replay_updates_without_her():
    env_kwargs = {
        "reference_residual_bounds": (0.02, 0.01, np.deg2rad(1.0)),
        "progress_gamma": 0.9999,
        "progress_weight": 1.0,
        "success_bonus": 1.0,
        "safety_failure_penalty": 10.0,
        "timeout_penalty": 1.0,
        "terminate_on_success": True,
    }
    transitions, _ = collect_reference_residual_trajectories(
        minimum_transitions=4,
        seed_start=706,
        max_episodes=3,
        max_steps=2,
        env_kwargs=env_kwargs,
        residual_std=0.2,
        residual_clip=0.5,
        zero_episode_fraction=0.0,
        random_seed=706,
        retain_failure_episodes=True,
    )
    vec_env = DummyVecEnv(
        [
            lambda: Monitor(
                ReferenceResidualLQRGoalConditionedClosingEnv(
                    mode="testing",
                    seed=707,
                    max_time=2.0,
                    **env_kwargs,
                )
            )
        ]
    )
    model = ZeroResidualConstrainedSAC(
        "MultiInputPolicy",
        vec_env,
        buffer_size=100,
        learning_starts=0,
        batch_size=2,
        train_freq=1,
        gradient_steps=1,
        critic_warmup_steps=2,
        zero_residual_actor_weight=1.0,
        actor_learning_rate_scale=0.25,
        policy_kwargs={"net_arch": [16, 16]},
        verbose=0,
        seed=707,
    )
    initialize_zero_residual_actor(model, log_std=-3.0)
    metrics = prefill_standard_replay_buffer(model, transitions)
    model.learn(total_timesteps=6)
    action, _ = model.predict(vec_env.reset(), deterministic=True)

    assert metrics["replay_buffer_class"] == "DictReplayBuffer"
    assert np.all(np.isfinite(action))
    assert model.critic_warmup_gradient_updates > 0
    assert model.constrained_actor_gradient_updates > 0
    vec_env.close()


def test_zero_residual_actor_weight_anneals_and_survives_save_load(tmp_path):
    vec_env = DummyVecEnv(
        [
            lambda: Monitor(
                ReferenceResidualLQRGoalConditionedClosingEnv(
                    mode="testing",
                    seed=708,
                    max_time=2.0,
                    reference_residual_bounds=(
                        0.02,
                        0.01,
                        np.deg2rad(1.0),
                    ),
                    progress_gamma=0.9999,
                )
            )
        ]
    )
    model = ZeroResidualConstrainedSAC(
        "MultiInputPolicy",
        vec_env,
        buffer_size=100,
        batch_size=2,
        critic_warmup_steps=10_000,
        zero_residual_actor_weight=1.0,
        zero_residual_actor_final_weight=0.1,
        zero_residual_actor_anneal_end_step=30_000,
        actor_learning_rate_scale=0.25,
        policy_kwargs={"net_arch": [16, 16]},
        verbose=0,
        seed=708,
    )

    assert model.current_zero_residual_actor_weight(0) == 1.0
    assert model.current_zero_residual_actor_weight(10_000) == 1.0
    assert model.current_zero_residual_actor_weight(20_000) == 0.55
    assert model.current_zero_residual_actor_weight(30_000) == 0.1
    assert model.current_zero_residual_actor_weight(50_000) == 0.1

    model_path = tmp_path / "annealed_actor_prior_model"
    model.save(model_path)
    loaded = ZeroResidualConstrainedSAC.load(model_path, env=vec_env)
    assert loaded.current_zero_residual_actor_weight(20_000) == 0.55
    assert loaded.current_zero_residual_actor_weight(50_000) == 0.1
    vec_env.close()
