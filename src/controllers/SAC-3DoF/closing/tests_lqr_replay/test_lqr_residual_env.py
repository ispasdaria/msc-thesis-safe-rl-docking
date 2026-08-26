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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.docking3dof.goal_conditioned_closing_env import (
    GoalConditionedClosingV3Env,
)
from lqr_residual_env import ResidualLQRGoalConditionedClosingEnv
from train_sac_closing_her_residual_lqr import (
    collect_lqr_residual_trajectories,
)
from train_sac_closing_her_v4_residual_v15 import (
    initialize_zero_residual_actor,
)
from zero_residual_constrained_sac import ZeroResidualConstrainedSAC


def test_lqr_residual_environment_contract():
    env = ResidualLQRGoalConditionedClosingEnv(
        mode="testing",
        seed=500,
        max_time=3.0,
        residual_scale=0.05,
    )
    check_env(env, warn=True)
    observation, info = env.reset(seed=500)

    assert set(observation) == {
        "observation",
        "achieved_goal",
        "desired_goal",
    }
    assert env.action_space.shape == (3,)
    assert info["reward_scheme"] == env.reward_scheme
    env.close()


def test_zero_residual_exactly_reproduces_clipped_lqr_transition():
    residual_env = ResidualLQRGoalConditionedClosingEnv(
        mode="testing",
        seed=501,
        max_time=10.0,
        residual_scale=0.05,
    )
    reference_env = GoalConditionedClosingV3Env(
        mode="testing",
        seed=501,
        max_time=10.0,
    )
    residual_env.reset(seed=501)
    reference_env.reset(seed=501)
    _, clipped_physical, source_action = residual_env.lqr_action()

    _, residual_reward, _, _, residual_info = residual_env.step(
        np.zeros(3, dtype=np.float32)
    )
    _, reference_reward, _, _, _ = reference_env.step(source_action)

    np.testing.assert_allclose(
        residual_env.core_env.state,
        reference_env.core_env.state,
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        residual_info["executed_normalized_action"],
        source_action,
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        residual_info["lqr_clipped_physical_action"],
        clipped_physical,
    )
    assert np.isclose(residual_reward, reference_reward)
    residual_env.close()
    reference_env.close()


def test_residual_is_scaled_and_clipped_in_normalized_action_space():
    env = ResidualLQRGoalConditionedClosingEnv(
        mode="testing",
        seed=502,
        max_time=10.0,
        residual_scale=0.10,
    )
    env.reset(seed=502)
    source_action = env.source_action()
    residual_action = np.array([0.5, -0.25, 1.0], dtype=np.float32)
    expected = np.clip(source_action + 0.10 * residual_action, -1.0, 1.0)

    _, _, _, _, info = env.step(residual_action)

    np.testing.assert_allclose(info["source_normalized_action"], source_action)
    np.testing.assert_allclose(
        info["scaled_residual_normalized_action"],
        0.10 * residual_action,
    )
    np.testing.assert_allclose(info["executed_normalized_action"], expected)
    assert isinstance(info["lqr_saturated"], bool)
    env.close()


def test_residual_gate_restores_exact_lqr_inside_braking_region():
    env = ResidualLQRGoalConditionedClosingEnv(
        mode="testing",
        seed=505,
        max_time=10.0,
        residual_scale=0.10,
        residual_zero_distance=100.0,
        residual_full_distance=200.0,
    )
    env.reset(seed=505)
    env.core_env.state[0] = env.core_env.p3_target[0] + 50.0
    source_action = env.source_action()

    _, _, _, _, info = env.step(np.ones(3, dtype=np.float32))

    assert info["residual_gate"] == 0.0
    np.testing.assert_allclose(
        info["executed_normalized_action"],
        source_action,
    )
    np.testing.assert_allclose(
        info["scaled_residual_normalized_action"],
        np.zeros(3),
    )
    env.close()


def test_held_residual_recomputes_lqr_on_every_simulator_step():
    class CountingLQR:
        def __init__(self):
            self.calls = 0

        def compute_action(self, core_env, state):
            self.calls += 1
            return np.zeros(3, dtype=np.float64)

    controller = CountingLQR()
    env = ResidualLQRGoalConditionedClosingEnv(
        mode="testing",
        seed=506,
        max_time=100.0,
        residual_scale=0.05,
        residual_hold_steps=10,
        lqr_controller=controller,
    )
    env.reset(seed=506)
    initial_time = float(env.core_env.time)

    _, reward, terminated, truncated, info = env.step(
        np.zeros(3, dtype=np.float32)
    )

    assert not terminated
    assert not truncated
    assert controller.calls == 10
    assert info["residual_hold_steps_requested"] == 10
    assert info["residual_hold_steps_executed"] == 10
    assert info["lqr_recomputed_each_simulator_step"] is True
    assert info["residual_hold_reward_aggregation"] == (
        "final_substep_for_her_compatibility"
    )
    assert np.isclose(env.core_env.time - initial_time, 10.0)
    assert np.isclose(reward, info["her_reward"])
    env.close()


def test_lqr_residual_prefill_stores_residual_not_executed_action():
    transitions, summaries = collect_lqr_residual_trajectories(
        minimum_transitions=4,
        seed_start=503,
        max_episodes=3,
        max_steps=2,
        residual_scale=0.05,
        residual_std=0.2,
        residual_clip=0.5,
        residual_hold_steps=1,
        zero_episode_fraction=0.0,
        random_seed=503,
        retain_failure_episodes=True,
    )
    replay_actions = np.stack([transition["action"] for transition in transitions])

    assert len(transitions) >= 4
    assert np.any(np.abs(replay_actions) > 0.0)
    assert np.max(np.abs(replay_actions)) <= 0.5
    assert all(summary["accepted"] == 1 for summary in summaries)
    assert all(summary["zero_residual_episode"] == 0 for summary in summaries)


def test_zero_residual_constrained_sac_warmup_and_update_are_finite():
    vec_env = DummyVecEnv(
        [
            lambda: Monitor(
                ResidualLQRGoalConditionedClosingEnv(
                    mode="testing",
                    seed=504,
                    max_time=2.0,
                    residual_scale=0.05,
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
        zero_residual_actor_weight=20.0,
        actor_learning_rate_scale=0.1,
        policy_kwargs={"net_arch": [16, 16]},
        verbose=0,
        seed=504,
    )
    initialize_zero_residual_actor(model, log_std=-4.0)
    observation = vec_env.reset()
    initial_action, _ = model.predict(observation, deterministic=True)
    np.testing.assert_allclose(initial_action, 0.0, atol=1e-7)

    model.learn(total_timesteps=6)

    action, _ = model.predict(vec_env.reset(), deterministic=True)
    assert np.all(np.isfinite(action))
    assert model.critic_warmup_gradient_updates > 0
    assert model.constrained_actor_gradient_updates > 0
    vec_env.close()
