"""Focused tests for behavior cloning and frozen-actor critic warm-up."""

import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from behavior_cloned_warmup_sac import (  # noqa: E402
    BehaviorClonedWarmupSAC,
)
from protected_mixed_replay_buffer import (  # noqa: E402
    ProtectedMixedReplayBuffer,
)
from train_sac_closing_lqr_bc_warmstart import (  # noqa: E402
    split_demonstration_episodes,
)
from train_sac_closing_lqr_replay import make_closing_env  # noqa: E402


def small_demonstration_dataset(count=12):
    observations = np.linspace(-1.0, 1.0, count * 12, dtype=np.float32).reshape(
        count,
        12,
    )
    terminated = np.zeros(count, dtype=np.bool_)
    terminated[3] = True
    terminated[7] = True
    terminated[11] = True
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


def parameter_snapshot(module):
    return [parameter.detach().cpu().clone() for parameter in module.parameters()]


def parameters_equal(before, module):
    return all(
        torch.equal(old, new.detach().cpu())
        for old, new in zip(before, module.parameters())
    )


def test_episode_split_has_no_transition_or_episode_leakage():
    dataset = small_demonstration_dataset()
    training, validation, metadata = split_demonstration_episodes(
        dataset,
        validation_fraction=1 / 3,
        split_seed=7,
    )
    assert len(set(training) & set(validation)) == 0
    assert sorted(np.concatenate((training, validation)).tolist()) == list(
        range(12)
    )
    assert not (
        set(metadata["training_episode_ids"])
        & set(metadata["validation_episode_ids"])
    )
    assert (
        metadata["training_transition_count"]
        + metadata["validation_transition_count"]
        == 12
    )


def test_actor_is_frozen_during_warmup_then_updates_afterward():
    env = DummyVecEnv(
        [lambda: make_closing_env("training", 0, "v15", monitor=True)]
    )
    try:
        model = BehaviorClonedWarmupSAC(
            "MlpPolicy",
            env,
            buffer_size=100,
            learning_starts=0,
            batch_size=8,
            train_freq=16,
            gradient_steps=1,
            replay_buffer_class=ProtectedMixedReplayBuffer,
            replay_buffer_kwargs={
                "initial_demo_fraction": 0.5,
                "later_demo_fraction": 0.25,
                "demo_fraction_switch_step": 32,
            },
            critic_warmup_steps=32,
            seed=0,
            device="cpu",
            verbose=0,
        )
        model.replay_buffer.set_demonstrations(small_demonstration_dataset())
        actor_before = parameter_snapshot(model.actor)
        critic_before = parameter_snapshot(model.critic)

        model.learn(total_timesteps=32)
        assert parameters_equal(actor_before, model.actor)
        assert not parameters_equal(critic_before, model.critic)
        assert model.critic_warmup_gradient_updates == 2

        model.learn(total_timesteps=16, reset_num_timesteps=False)
        assert not parameters_equal(actor_before, model.actor)
        assert model.num_timesteps == 48
    finally:
        env.close()
