"""Focused tests for behavior-constrained SAC fine-tuning."""

import sys
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from behavior_cloned_constrained_sac import (  # noqa: E402
    BehaviorClonedConstrainedSAC,
)
from protected_mixed_replay_buffer import (  # noqa: E402
    ProtectedMixedReplayBuffer,
)
from train_sac_closing_lqr_bc_constrained import (  # noqa: E402
    evaluation_rank,
)
from train_sac_closing_lqr_replay import make_closing_env  # noqa: E402


def small_demonstration_dataset(count=12):
    observations = np.linspace(-1.0, 1.0, count * 12, dtype=np.float32).reshape(
        count,
        12,
    )
    terminated = np.zeros(count, dtype=np.bool_)
    terminated[[3, 7, 11]] = True
    return {
        "observations": observations,
        "actions": np.tanh(observations[:, :3]).astype(np.float32),
        "rewards": np.linspace(-1.0, 1.0, count, dtype=np.float32),
        "next_observations": observations + 0.01,
        "terminated": terminated,
        "truncated": np.zeros(count, dtype=np.bool_),
        "episode_offsets": np.asarray([0, 4, 8, 12], dtype=np.int64),
        "episode_seeds": np.asarray([100, 101, 102], dtype=np.int64),
    }


def make_model(env):
    model = BehaviorClonedConstrainedSAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100,
        learning_starts=0,
        batch_size=8,
        train_freq=16,
        gradient_steps=1,
        replay_buffer_class=ProtectedMixedReplayBuffer,
        replay_buffer_kwargs={
            "initial_demo_fraction": 0.5,
            "later_demo_fraction": 0.5,
            "demo_fraction_switch_step": 32,
        },
        critic_warmup_steps=32,
        deterministic_finetune_steps=16,
        bc_finetune_weight_start=10.0,
        bc_finetune_weight_end=1.0,
        bc_finetune_decay_steps=100,
        bc_finetune_batch_size=8,
        actor_learning_rate_scale=0.1,
        seed=0,
        device="cpu",
        verbose=0,
    )
    model.replay_buffer.set_demonstrations(small_demonstration_dataset())
    return model


def test_bc_weight_schedule_and_deterministic_handoff():
    env = DummyVecEnv(
        [lambda: make_closing_env("training", 0, "v15", monitor=True)]
    )
    try:
        model = make_model(env)
        model.num_timesteps = 32
        assert model.bc_finetune_weight == pytest.approx(10.0)
        assert model.use_deterministic_collection

        model.num_timesteps = 82
        assert model.bc_finetune_weight == pytest.approx(5.5)
        assert not model.use_deterministic_collection

        model.num_timesteps = 132
        assert model.bc_finetune_weight == pytest.approx(1.0)
    finally:
        env.close()


def test_post_warmup_update_uses_bc_loss_and_scaled_actor_rate():
    env = DummyVecEnv(
        [lambda: make_closing_env("training", 0, "v15", monitor=True)]
    )
    try:
        model = make_model(env)
        model.learn(total_timesteps=32)
        assert model.constrained_actor_gradient_updates == 0

        model.learn(total_timesteps=16, reset_num_timesteps=False)
        assert model.constrained_actor_gradient_updates == 1
        assert model.actor.optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
        bc_loss, mean_loss, log_std_loss = model._demonstration_bc_loss(8)
        assert np.isfinite(float(bc_loss.detach().cpu().item()))
        assert np.isfinite(float(mean_loss.detach().cpu().item()))
        assert np.isfinite(float(log_std_loss.detach().cpu().item()))
    finally:
        env.close()


def test_checkpoint_ranking_prefers_success_then_safety():
    successful = {
        "success_rate": 0.85,
        "failure_rate": 0.0,
        "time_limit_rate": 0.15,
        "mean_return": -20_000.0,
    }
    high_return_failure = {
        "success_rate": 0.0,
        "failure_rate": 0.0,
        "time_limit_rate": 1.0,
        "mean_return": -1_000.0,
    }
    safer = {
        "success_rate": 0.85,
        "failure_rate": 0.0,
        "time_limit_rate": 0.15,
        "mean_return": -20_000.0,
    }
    less_safe = {
        "success_rate": 0.85,
        "failure_rate": 0.05,
        "time_limit_rate": 0.10,
        "mean_return": -500.0,
    }
    assert evaluation_rank(successful) > evaluation_rank(high_return_failure)
    assert evaluation_rank(safer) > evaluation_rank(less_safe)
