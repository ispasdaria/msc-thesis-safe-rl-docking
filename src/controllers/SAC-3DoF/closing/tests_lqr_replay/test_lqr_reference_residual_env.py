from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SRC_PATH = Path(__file__).resolve().parents[4]
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
