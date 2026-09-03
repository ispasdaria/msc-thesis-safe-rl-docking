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
from lqr_residual_lateral_env import (
    LateralProgressResidualLQRGoalConditionedClosingEnv,
)
from lqr_residual_terminal_fade_env import (
    TerminalFadeLateralResidualLQRGoalConditionedClosingEnv,
)
from her_closing_utils import make_her_env
from train_sac_closing_her_residual_lqr import (
    PeriodicTrainingLoggerCallback,
    build_parser,
    collect_lqr_residual_trajectories,
    reward_environment_kwargs,
    _validate_args,
)
from train_sac_closing_her_residual_lqr_pilot_lateral_balanced_200k import (
    DEFAULT_ARGUMENTS as LATERAL_BALANCED_200K_ARGUMENTS,
)
from train_sac_closing_her_residual_lqr_pilot_terminal_fade_200k import (
    DEFAULT_ARGUMENTS as TERMINAL_FADE_200K_ARGUMENTS,
)
from train_sac_closing_residual_lqr_no_her_stable_200k import (
    DEFAULT_ARGUMENTS as NO_HER_STABLE_200K_ARGUMENTS,
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
        critic_loss_type="huber",
        max_critic_grad_norm=10.0,
        max_actor_grad_norm=10.0,
        ent_coef=0.001,
        policy_kwargs={"net_arch": [16, 16]},
        verbose=0,
        seed=504,
    )
    initialize_zero_residual_actor(model, log_std=-4.0)
    observation = vec_env.reset()
    initial_action, _ = model.predict(observation, deterministic=True)
    np.testing.assert_allclose(initial_action, 0.0, atol=1e-7)

    model.learn(
        total_timesteps=6,
        callback=PeriodicTrainingLoggerCallback(interval_steps=2),
    )

    action, _ = model.predict(vec_env.reset(), deterministic=True)
    assert np.all(np.isfinite(action))
    assert model.critic_warmup_gradient_updates > 0
    assert model.constrained_actor_gradient_updates > 0
    assert type(model.replay_buffer).__name__ == "DictReplayBuffer"
    assert model.critic_loss_type == "huber"
    assert model.ent_coef_optimizer is None
    vec_env.close()


def test_lateral_reward_factory_defaults_to_canonical_dt01():
    env = make_her_env(
        seed=507,
        mode="testing",
        max_time=1.0,
        reward_version="lqr_residual_lateral",
    )
    _, info = env.reset(seed=507)

    assert np.isclose(env.core_env.dt, 0.1)
    assert info["reward_scheme"] == env.reward_scheme
    assert env.reward_scheme.endswith("dt01_lateral_v1")
    env.close()


def test_lateral_reward_does_not_change_zero_residual_dynamics():
    common = {
        "mode": "testing",
        "seed": 508,
        "dt": 0.1,
        "max_time": 1.0,
        "residual_scale": 0.05,
        "residual_zero_distance": 100.0,
        "residual_full_distance": 200.0,
    }
    absolute_env = ResidualLQRGoalConditionedClosingEnv(**common)
    lateral_env = LateralProgressResidualLQRGoalConditionedClosingEnv(**common)
    absolute_env.reset(seed=508)
    lateral_env.reset(seed=508)

    _, _, _, _, absolute_info = absolute_env.step(np.zeros(3, dtype=np.float32))
    _, _, _, _, lateral_info = lateral_env.step(np.zeros(3, dtype=np.float32))

    np.testing.assert_allclose(
        lateral_env.core_env.state,
        absolute_env.core_env.state,
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        lateral_info["executed_normalized_action"],
        absolute_info["executed_normalized_action"],
        rtol=0.0,
        atol=1e-7,
    )
    absolute_env.close()
    lateral_env.close()


def test_lateral_reward_rewards_reducing_lateral_constraint_violation():
    env = LateralProgressResidualLQRGoalConditionedClosingEnv(
        mode="testing",
        seed=509,
        dt=0.1,
        max_time=1.0,
        progress_weight=0.0,
        lateral_constraint_progress_weight=4.0,
        success_bonus=0.0,
        safety_failure_penalty=0.0,
        timeout_penalty=0.0,
    )
    observation, _ = env.reset(seed=509)
    target = env.core_env.p3_target.copy()
    tolerance = env.core_env.p3_target_tolerance
    previous_state = target.copy()
    next_state = target.copy()
    previous_state[1] += 2.0 * tolerance[1]
    next_state[1] += 1.5 * tolerance[1]
    info = {"previous_achieved_goal": env._scale_state(previous_state)}

    reward = env.compute_reward(
        env._scale_state(next_state),
        observation["desired_goal"],
        info,
    )

    assert reward > 0.0
    env.close()


def test_lateral_reward_step_records_her_previous_goal():
    env = LateralProgressResidualLQRGoalConditionedClosingEnv(
        mode="testing",
        seed=510,
        dt=0.1,
        max_time=1.0,
    )
    observation, _ = env.reset(seed=510)
    previous = observation["achieved_goal"].copy()
    next_observation, reward, _, _, info = env.step(
        np.zeros(3, dtype=np.float32)
    )
    recomputed = env.compute_reward(
        next_observation["achieved_goal"],
        next_observation["desired_goal"],
        info,
    )

    np.testing.assert_allclose(info["previous_achieved_goal"], previous)
    assert np.isclose(reward, recomputed)
    assert np.isclose(info["her_reward"], reward)
    env.close()


def test_lateral_reward_recomputes_for_batched_her_goals():
    env = LateralProgressResidualLQRGoalConditionedClosingEnv(
        mode="testing",
        seed=512,
        dt=0.1,
        max_time=1.0,
    )
    observation, _ = env.reset(seed=512)
    previous = observation["achieved_goal"].copy()
    achieved = previous.copy()
    achieved[1] -= 1e-3
    desired = observation["desired_goal"].copy()
    info = {"previous_achieved_goal": previous}

    rewards = env.compute_reward(
        np.stack([achieved, achieved]),
        np.stack([desired, achieved]),
        [info, info],
    )

    assert rewards.shape == (2,)
    assert np.all(np.isfinite(rewards))
    assert rewards[1] > rewards[0]
    env.close()


def test_balanced_prefill_deterministically_collects_both_roles():
    transitions, summaries = collect_lqr_residual_trajectories(
        minimum_transitions=4,
        seed_start=511,
        max_episodes=2,
        max_steps=2,
        residual_scale=0.05,
        residual_std=0.2,
        residual_clip=0.5,
        residual_hold_steps=1,
        zero_episode_fraction=1.0,
        minimum_zero_episodes=1,
        minimum_exploratory_episodes=1,
        random_seed=511,
        retain_failure_episodes=True,
        reward_version="lqr_residual_lateral",
    )

    assert len(transitions) == 4
    assert [row["prefill_role"] for row in summaries] == [
        "required_zero",
        "required_exploratory",
    ]
    first_actions = np.stack(
        [item["action"] for item in transitions if item["episode_id"] == 0]
    )
    second_actions = np.stack(
        [item["action"] for item in transitions if item["episode_id"] == 1]
    )
    np.testing.assert_allclose(first_actions, 0.0)
    assert np.any(np.abs(second_actions) > 0.0)
    assert all("previous_achieved_goal" in item["info"] for item in transitions)


def test_balanced_lateral_200k_preset_is_explicit_and_valid():
    parser = build_parser()
    args = parser.parse_args(LATERAL_BALANCED_200K_ARGUMENTS)
    _validate_args(parser, args)
    reward_kwargs = reward_environment_kwargs(args)

    assert args.reward_version == "lqr_residual_lateral"
    assert args.total_timesteps == 200_000
    assert args.checkpoint_steps == [120_000, 200_000]
    assert args.prefill_min_zero_episodes == 1
    assert args.prefill_min_exploratory_episodes == 1
    assert args.source_transition_count == 200_000
    assert args.buffer_size == 500_000
    assert args.critic_warmup_steps == 30_000
    assert args.zero_residual_actor_weight == 1.0
    assert args.zero_residual_actor_final_weight == 0.1
    assert args.zero_residual_actor_anneal_end_step == 150_000
    assert reward_kwargs["lateral_constraint_progress_weight"] == 4.0


def test_terminal_fade_replaces_the_100m_zero_authority_region():
    env = TerminalFadeLateralResidualLQRGoalConditionedClosingEnv(
        mode="testing",
        seed=513,
        dt=0.1,
        max_time=1.0,
        residual_zero_distance=100.0,
        residual_full_distance=200.0,
        terminal_fade_full_ratio=1.2,
        terminal_fade_zero_ratio=1.0,
    )
    env.reset(seed=513)
    target = env.core_env.p3_target.copy()
    state = target.copy()
    state[0] += 50.0
    env.core_env.state = state

    authority, ratio = env.terminal_residual_authority()

    assert abs(state[0] - target[0]) < 100.0
    assert ratio > 1.2
    assert authority == 1.0
    assert env.residual_gate() == 1.0
    env.close()


def test_terminal_fade_factory_enforces_canonical_dt01():
    env = make_her_env(
        seed=517,
        mode="testing",
        max_time=1.0,
        reward_version="lqr_residual_terminal_fade",
    )
    assert np.isclose(env.core_env.dt, 0.1)
    env.close()

    with np.testing.assert_raises(ValueError):
        make_her_env(
            seed=517,
            mode="testing",
            max_time=1.0,
            reward_version="lqr_residual_terminal_fade",
            dt=1.0,
        )


def test_terminal_fade_is_smooth_and_zero_only_in_success_set():
    env = TerminalFadeLateralResidualLQRGoalConditionedClosingEnv(
        mode="testing",
        seed=514,
        dt=0.1,
        max_time=1.0,
        terminal_fade_full_ratio=1.2,
        terminal_fade_zero_ratio=1.0,
    )
    env.reset(seed=514)
    target = env.core_env.p3_target.copy()
    state = target.copy()
    state[1] += 1.1 * env.core_env.p3_target_tolerance[1]
    env.core_env.state = state
    middle_authority, middle_ratio = env.terminal_residual_authority()
    env.core_env.state = target.copy()
    zero_authority, zero_ratio = env.terminal_residual_authority()

    assert np.isclose(middle_ratio, 1.1)
    assert np.isclose(middle_authority, 0.5)
    assert zero_ratio == 0.0
    assert zero_authority == 0.0
    env.close()


def test_terminal_fade_reward_prices_magnitude_and_action_rate():
    env = TerminalFadeLateralResidualLQRGoalConditionedClosingEnv(
        mode="testing",
        seed=515,
        dt=0.1,
        max_time=1.0,
        progress_weight=0.0,
        lateral_constraint_progress_weight=0.0,
        success_bonus=0.0,
        safety_failure_penalty=0.0,
        timeout_penalty=0.0,
        residual_magnitude_weight=0.002,
        residual_smoothness_weight=0.001,
    )
    observation, _ = env.reset(seed=515)
    achieved = observation["achieved_goal"]
    info = {
        "previous_achieved_goal": achieved.copy(),
        "effective_residual_normalized_action": np.array([1.0, 0.0, 0.0]),
        "previous_effective_residual_normalized_action": np.zeros(3),
    }

    reward = env.compute_reward(achieved, achieved, info)

    assert np.isclose(reward, -(0.1 * 0.002 + 10.0 * 0.001))
    env.close()


def test_terminal_fade_step_info_supports_her_reward_recomputation():
    env = TerminalFadeLateralResidualLQRGoalConditionedClosingEnv(
        mode="testing",
        seed=516,
        dt=0.1,
        max_time=1.0,
    )
    env.reset(seed=516)
    next_observation, reward, _, _, info = env.step(
        np.array([0.2, -0.1, 0.3], dtype=np.float32)
    )
    recomputed = env.compute_reward(
        next_observation["achieved_goal"],
        next_observation["desired_goal"],
        info,
    )

    assert info["residual_gate_design"] == "p3_ratio_terminal_fade"
    assert "effective_residual_normalized_action" in info
    assert "previous_effective_residual_normalized_action" in info
    assert np.isclose(reward, recomputed)
    env.close()


def test_terminal_fade_200k_preset_keeps_stronger_prior_and_logging():
    parser = build_parser()
    args = parser.parse_args(TERMINAL_FADE_200K_ARGUMENTS)
    _validate_args(parser, args)
    reward_kwargs = reward_environment_kwargs(args)

    assert args.reward_version == "lqr_residual_terminal_fade"
    assert args.total_timesteps == 200_000
    assert args.zero_residual_actor_weight == 1.0
    assert args.zero_residual_actor_final_weight == 0.5
    assert args.terminal_fade_full_ratio == 1.2
    assert args.terminal_fade_zero_ratio == 1.0
    assert args.log_interval_steps == 10_000
    assert reward_kwargs["residual_magnitude_weight"] == 0.002
    assert reward_kwargs["residual_smoothness_weight"] == 0.001
    PeriodicTrainingLoggerCallback(args.log_interval_steps)


def test_no_her_stable_200k_preset_uses_fixed_goal_replay_and_guards():
    parser = build_parser()
    args = parser.parse_args(NO_HER_STABLE_200K_ARGUMENTS)
    _validate_args(parser, args)

    assert args.replay_mode == "standard"
    assert args.reward_version == "lqr_residual_terminal_fade"
    assert args.total_timesteps == 200_000
    assert args.checkpoint_steps == [80_000, 120_000, 160_000, 200_000]
    assert args.source_transition_count == 300_000
    assert args.prefill_min_zero_episodes == 1
    assert args.prefill_min_exploratory_episodes == 2
    assert args.retain_failure_prefill is True
    assert args.buffer_size == 600_000
    assert args.critic_warmup_steps == 50_000
    assert args.critic_loss_type == "huber"
    assert args.max_critic_grad_norm == 10.0
    assert args.max_actor_grad_norm == 10.0
    assert args.ent_coef == 0.001
    assert args.zero_residual_actor_weight == 1.0
    assert args.zero_residual_actor_final_weight is None
    assert args.actor_learning_rate_scale == 0.1
