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

from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.docking3dof.goal_conditioned_closing_env import (
    GoalConditionedClosingEnv,
    GoalConditionedClosingV2Env,
    GoalConditionedClosingV3Env,
)
from her_source_warmstart import (
    behavior_distill_source_actor,
    collect_safe_source_trajectories,
    prefill_her_replay_buffer,
)
from her_residual_env import ResidualV15GoalConditionedClosingEnv
from her_closing_utils import (
    _classify_rollout_outcome,
    evaluate_model,
    physical_goal_metrics,
    summarize_rollouts,
)
from train_sac_closing_her_v4_residual_v15 import (
    collect_residual_source_trajectories,
    initialize_zero_residual_actor,
)


def test_goal_conditioned_observation_and_environment_contract():
    env = GoalConditionedClosingEnv(mode="testing", seed=200, max_time=10.0)
    check_env(env, warn=True)
    observation, info = env.reset(seed=200)

    assert set(observation) == {
        "observation",
        "achieved_goal",
        "desired_goal",
    }
    assert all(value.shape == (6,) for value in observation.values())
    np.testing.assert_allclose(
        observation["achieved_goal"] * env.state_scale,
        env.core_env.state,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        observation["desired_goal"] * env.state_scale,
        env.core_env.p3_target,
        rtol=1e-6,
        atol=1e-6,
    )
    assert info["reward_scheme"] == "goal_conditioned_p3_v1"
    env.close()


def test_compute_reward_is_vectorized_bounded_and_wraps_angle():
    env = GoalConditionedClosingEnv(mode="testing", seed=201, max_time=10.0)
    env.reset(seed=201)
    desired = np.zeros((3, 6), dtype=np.float32)
    achieved = desired.copy()

    # One exact success, one axial error at twice tolerance, and an angular
    # difference of two degrees across the -pi/pi boundary.
    achieved[1, 0] = float(
        2.0 * env.core_env.p3_target_tolerance[0] / env.state_scale[0]
    )
    desired[2, 4] = float(np.deg2rad(179.0) / env.state_scale[4])
    achieved[2, 4] = float(np.deg2rad(-179.0) / env.state_scale[4])

    rewards = env.compute_reward(
        achieved,
        desired,
        [{}, {}, {"failure": True}],
    )
    ratios = env.goal_error_ratios(achieved, desired)

    assert rewards.shape == (3,)
    assert rewards.dtype == np.float32
    assert rewards[0] == 1.0
    assert -1.0 < rewards[1] < 0.0
    assert np.isclose(ratios[2, 4], 4.0, atol=1e-4)
    assert rewards[2] < -1.0
    env.close()


def test_v2_restores_far_field_gradient_and_strong_failure_cost():
    env = GoalConditionedClosingV2Env(
        mode="testing",
        seed=207,
        max_time=10.0,
    )
    env.reset(seed=207)
    desired = np.zeros((2, 6), dtype=np.float32)
    achieved = desired.copy()
    achieved[0, 0] = float(
        20.0 * env.core_env.p3_target_tolerance[0] / env.state_scale[0]
    )
    achieved[1, 0] = float(
        80.0 * env.core_env.p3_target_tolerance[0] / env.state_scale[0]
    )
    rewards = env.compute_reward(achieved, desired, [{}, {}])
    assert rewards[0] > rewards[1]

    failure_reward = env.compute_reward(
        desired[0],
        desired[0],
        {"failure": True},
    )
    assert np.isclose(failure_reward, -999.0)
    env.close()


def test_v2_velocity_warning_depth_is_bounded_and_goal_independent():
    env = GoalConditionedClosingV2Env(
        mode="testing",
        seed=208,
        max_time=10.0,
    )
    env.reset(seed=208)
    goal = np.zeros(6, dtype=np.float32)
    safe_reward = env.compute_reward(
        goal,
        goal,
        {"port_speed": env.core_env.vp_safe_close, "s_dot": 0.0},
    )
    half_warning_reward = env.compute_reward(
        goal,
        goal,
        {
            "port_speed": 0.5
            * (env.core_env.vp_safe_close + env.core_env.vp_unsafe_close),
            "s_dot": 0.0,
        },
    )
    unsafe_reward = env.compute_reward(
        goal,
        goal,
        {"port_speed": 10.0, "s_dot": 10.0},
    )
    assert np.isclose(safe_reward, 1.0)
    assert np.isclose(half_warning_reward, -4.0)
    assert np.isclose(unsafe_reward, -9.0)
    env.close()


def test_v3_reward_has_long_horizon_scale_and_no_goal_ratio_clipping():
    env = GoalConditionedClosingV3Env(
        mode="testing",
        seed=209,
        max_time=10.0,
    )
    env.reset(seed=209)
    desired = np.zeros((2, 6), dtype=np.float32)
    achieved = desired.copy()
    achieved[0, 0] = float(
        80.0 * env.core_env.p3_target_tolerance[0] / env.state_scale[0]
    )
    achieved[1, 0] = float(
        200.0 * env.core_env.p3_target_tolerance[0] / env.state_scale[0]
    )
    rewards = env.compute_reward(achieved, desired, [{}, {}])
    assert rewards[0] > rewards[1]
    assert rewards[0] > -0.02
    assert rewards[1] > -0.02

    exact_success = env.compute_reward(desired[0], desired[0], {})
    exact_failure = env.compute_reward(
        desired[0], desired[0], {"failure": True}
    )
    assert np.isclose(exact_success, 100.0)
    assert np.isclose(exact_failure, -900.0)
    env.close()


def test_v3_geometry_and_alignment_warning_depths_use_stored_state():
    env = GoalConditionedClosingV3Env(
        mode="testing",
        seed=210,
        max_time=10.0,
    )
    env.reset(seed=210)
    state = np.asarray(env.core_env.state, dtype=np.float64).copy()
    state[0] = env.core_env.ae_x
    state[1] = 0.0
    geometry_depth = env._geometry_warning_depth({"state": state}, ())
    alignment_depth = env._alignment_warning_depth(
        {
            "psi_e": 0.5 * (env.core_env.psi_safe + env.core_env.psi_unsafe),
            "omega": 0.0,
        },
        (),
    )
    assert np.isclose(geometry_depth, 1.0)
    assert np.isclose(alignment_depth, 0.5)
    env.close()


def test_step_reward_matches_recomputed_reward():
    env = GoalConditionedClosingEnv(mode="testing", seed=202, max_time=10.0)
    env.reset(seed=202)
    observation, reward, _, _, info = env.step(
        np.zeros(3, dtype=np.float32)
    )
    recomputed = env.compute_reward(
        observation["achieved_goal"],
        observation["desired_goal"],
        info,
    )
    assert np.isclose(reward, recomputed)
    assert np.isclose(info["her_reward"], reward)
    env.close()


def test_goal_success_is_nonterminal_for_her_but_safety_remains_terminal():
    env = GoalConditionedClosingEnv(mode="testing", seed=204, max_time=10.0)
    _, _ = env.reset(seed=204)
    target = np.asarray(env.core_env.p3_target, dtype=np.float64).copy()
    observation, _ = env.reset(
        options={
            "initial_state": target,
            "p3_target": target,
            "mode": "testing",
        }
    )
    assert np.all(
        env.goal_error_ratios(
            observation["achieved_goal"], observation["desired_goal"]
        )
        <= 1.0
    )
    _, _, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert info["is_success"] == 1.0
    assert info["original_p3_success"] is True
    assert terminated is False
    env.close()


def test_sac_can_sample_and_update_from_her_buffer():
    vec_env = DummyVecEnv(
        [
            lambda: Monitor(
                GoalConditionedClosingEnv(
                    mode="testing",
                    seed=203,
                    max_time=3.0,
                )
            )
        ]
    )
    model = SAC(
        "MultiInputPolicy",
        vec_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": 2,
            "goal_selection_strategy": "future",
            "copy_info_dict": True,
        },
        buffer_size=100,
        learning_starts=4,
        batch_size=4,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [16, 16]},
        verbose=0,
        seed=203,
    )
    model.learn(total_timesteps=8)
    sample = model.replay_buffer.sample(4)
    assert set(sample.observations) == {
        "observation",
        "achieved_goal",
        "desired_goal",
    }
    assert sample.rewards.shape == (4, 1)
    vec_env.close()


def test_saved_her_sac_can_be_loaded_for_inference(tmp_path):
    def make_vec_env(seed):
        return DummyVecEnv(
            [
                lambda: Monitor(
                    GoalConditionedClosingEnv(
                        mode="testing",
                        seed=seed,
                        max_time=3.0,
                    )
                )
            ]
        )

    training_env = make_vec_env(205)
    model = SAC(
        "MultiInputPolicy",
        training_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": 2,
            "goal_selection_strategy": "future",
            "copy_info_dict": True,
        },
        buffer_size=100,
        learning_starts=4,
        batch_size=4,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [16, 16]},
        verbose=0,
        seed=205,
    )
    model.learn(total_timesteps=8)
    model_path = tmp_path / "her_model"
    model.save(model_path)
    training_env.close()

    inference_env = make_vec_env(206)
    loaded = SAC.load(model_path, env=inference_env, device="cpu")
    observation = inference_env.reset()
    action, _ = loaded.predict(observation, deterministic=True)
    assert action.shape == (1, 3)
    inference_env.close()


def test_source_trajectory_prefill_distillation_and_her_update(tmp_path):
    class ZeroSource:
        def predict(self, observation, deterministic=True):
            return np.zeros((len(observation), 3), dtype=np.float32), None

    class IdentityNormalizer:
        @staticmethod
        def normalize_obs(observation):
            return observation

    transitions, summaries = collect_safe_source_trajectories(
        source_model=ZeroSource(),
        source_vecnormalize=IdentityNormalizer(),
        minimum_transitions=6,
        seed_start=300,
        max_episodes=3,
        max_steps=3,
    )
    assert len(transitions) >= 6
    assert sum(row["accepted"] for row in summaries) >= 2

    vec_env = DummyVecEnv(
        [
            lambda: Monitor(
                GoalConditionedClosingV3Env(
                    mode="testing",
                    seed=300,
                    max_time=3.0,
                )
            )
        ]
    )
    model = SAC(
        "MultiInputPolicy",
        vec_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": 2,
            "goal_selection_strategy": "future",
            "copy_info_dict": True,
        },
        buffer_size=100,
        learning_starts=0,
        batch_size=4,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [16, 16]},
        verbose=0,
        seed=300,
    )
    prefill = prefill_her_replay_buffer(model, transitions)
    assert prefill["prefill_transition_count"] >= 6
    metrics = behavior_distill_source_actor(
        model=model,
        transitions=transitions,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        validation_fraction=0.5,
        split_seed=300,
        log_std_target=-3.0,
        log_std_weight=0.01,
        result_path=tmp_path / "bc.csv",
    )
    assert np.isfinite(metrics["best_validation_action_mse"])
    model.learn(total_timesteps=4)
    vec_env.close()


def test_zero_residual_executes_exact_frozen_source_action():
    residual_env = ResidualV15GoalConditionedClosingEnv(
        mode="testing",
        seed=400,
        max_time=10.0,
        residual_scale=0.05,
        source_device="cpu",
    )
    reference_env = GoalConditionedClosingV3Env(
        mode="testing",
        seed=400,
        max_time=10.0,
    )
    residual_env.reset(seed=400)
    reference_env.reset(seed=400)
    source_action = residual_env.source_action()
    _, _, _, _, residual_info = residual_env.step(
        np.zeros(3, dtype=np.float32)
    )
    reference_env.step(source_action)

    np.testing.assert_allclose(
        residual_env.core_env.state,
        reference_env.core_env.state,
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        residual_info["executed_normalized_action"],
        source_action,
    )
    residual_env.close()
    reference_env.close()


def test_zero_residual_actor_initialization_is_exact():
    vec_env = DummyVecEnv(
        [
            lambda: Monitor(
                GoalConditionedClosingV3Env(
                    mode="testing",
                    seed=401,
                    max_time=3.0,
                )
            )
        ]
    )
    model = SAC(
        "MultiInputPolicy",
        vec_env,
        buffer_size=100,
        learning_starts=10,
        policy_kwargs={"net_arch": [16, 16]},
        verbose=0,
        seed=401,
    )
    initialize_zero_residual_actor(model, log_std=-4.0)
    observation = vec_env.reset()
    action, _ = model.predict(observation, deterministic=True)
    np.testing.assert_allclose(action, np.zeros((1, 3)), atol=1e-7)
    vec_env.close()


def test_physical_goal_metrics_identify_limiting_component():
    env = GoalConditionedClosingV3Env(
        mode="testing",
        seed=402,
        max_time=3.0,
    )
    env.reset(seed=402)
    env.core_env.state = np.asarray(env.core_env.p3_target).copy()
    env.core_env.state[1] += 3.0 * env.core_env.p3_target_tolerance[1]
    env.core_env.state[4] += 2.0 * env.core_env.p3_target_tolerance[4]

    metrics = physical_goal_metrics(env)

    assert metrics["limiting_component"] == "lateral"
    assert np.isclose(metrics["lateral_ratio"], 3.0)
    assert np.isclose(metrics["attitude_ratio"], 2.0)
    env.close()


def test_exploratory_residual_collection_retains_action_diversity():
    transitions, summaries = collect_residual_source_trajectories(
        minimum_transitions=4,
        seed_start=403,
        max_episodes=4,
        max_steps=2,
        residual_scale=0.10,
        residual_std=0.35,
        residual_clip=0.75,
        residual_hold_steps=1,
        zero_episode_fraction=0.0,
        random_seed=403,
        retain_failure_episodes=True,
    )
    actions = np.stack([row["action"] for row in transitions])

    assert len(transitions) >= 4
    assert np.any(np.abs(actions) > 0.0)
    assert all(row["accepted"] == 1 for row in summaries)
    assert all(row["zero_residual_episode"] == 0 for row in summaries)


def test_evaluator_records_component_snapshots_and_summary():
    class ZeroPolicy:
        @staticmethod
        def predict(observation, deterministic=True):
            return np.zeros(3, dtype=np.float32), None

    rows = evaluate_model(
        ZeroPolicy(),
        seeds=[404],
        max_steps=2,
        reward_version="v3",
    )
    summary = summarize_rollouts(rows)

    assert rows[0]["best_joint_limiting_component"] in {
        "axial",
        "lateral",
        "vx",
        "vy",
        "attitude",
        "rate",
    }
    assert "closest_attitude_ratio" in rows[0]
    assert "final_lateral_ratio" in rows[0]
    assert "mean_best_joint_attitude_ratio" in summary
    assert np.isclose(
        sum(
            summary[f"best_joint_limiting_{component}_fraction"]
            for component in (
                "axial",
                "lateral",
                "vx",
                "vy",
                "attitude",
                "rate",
            )
        ),
        1.0,
    )


def test_evaluator_accepts_core_success_when_wrapper_success_is_false():
    success, failure, time_limit = _classify_rollout_outcome(
        {"success": True, "is_success": False},
        steps=37,
        max_steps=100,
    )

    assert success is True
    assert failure is False
    assert time_limit is False
