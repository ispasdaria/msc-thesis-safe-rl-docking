"""Focused tests for the isolated LQR replay experiment."""

import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_lqr_replay_dataset import (  # noqa: E402
    combine_successful_episodes,
    load_demonstration_dataset,
    save_demonstration_dataset,
)
from train_sac_closing_lqr_replay import (  # noqa: E402
    initialize_vecnormalize_from_demonstrations,
    make_closing_env,
    prefill_replay_buffer,
    resolve_buffer_size,
)


def make_successful_episode(seed, length, observation_offset=0.0):
    observations = (
        np.arange(length * 12, dtype=np.float32).reshape(length, 12)
        + observation_offset
    )
    next_observations = observations + 0.25
    terminated = np.zeros(length, dtype=np.bool_)
    terminated[-1] = True
    return {
        "seed": seed,
        "success": True,
        "termination_reason": "success_P3_target",
        "observations": observations,
        "actions": np.linspace(
            -0.8,
            0.8,
            length * 3,
            dtype=np.float32,
        ).reshape(length, 3),
        "rewards": np.linspace(-2.0, 10.0, length, dtype=np.float32),
        "next_observations": next_observations,
        "terminated": terminated,
        "truncated": np.zeros(length, dtype=np.bool_),
        "initial_state": np.zeros(6, dtype=np.float64),
        "p3_target": np.ones(6, dtype=np.float64),
        "final_state": np.ones(6, dtype=np.float64),
        "episode_return": float(
            np.sum(np.linspace(-2.0, 10.0, length), dtype=np.float64)
        ),
        "saturation_fraction": 0.0,
    }


def test_complete_episodes_round_trip_without_subsampling(tmp_path):
    episodes = [
        make_successful_episode(seed=20_000, length=4),
        make_successful_episode(
            seed=20_001,
            length=6,
            observation_offset=1000.0,
        ),
    ]
    dataset = combine_successful_episodes(
        episodes,
        reward_version="v15",
        mode="training",
    )

    assert len(dataset["rewards"]) == 10
    assert np.array_equal(dataset["episode_offsets"], [0, 4, 10])
    assert np.array_equal(dataset["episode_ids"], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert np.array_equal(dataset["observations"][:4], episodes[0]["observations"])
    assert np.array_equal(dataset["observations"][4:], episodes[1]["observations"])

    path = save_demonstration_dataset(tmp_path / "demonstrations.npz", dataset)
    loaded, metadata = load_demonstration_dataset(
        path,
        expected_reward_version="v15",
    )
    assert metadata["successful_episodes"] == 2
    assert metadata["transition_count"] == 10
    assert np.array_equal(loaded["actions"], dataset["actions"])


def test_vecnormalize_initialization_and_replay_prefill():
    dataset = combine_successful_episodes(
        [make_successful_episode(seed=20_000, length=5)],
        reward_version="v15",
        mode="training",
    )
    vec_env = DummyVecEnv(
        [
            lambda: make_closing_env(
                mode="training",
                seed=0,
                reward_version="v15",
                monitor=True,
            )
        ]
    )
    vecnormalize = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        gamma=0.99,
    )
    try:
        initialize_vecnormalize_from_demonstrations(
            vecnormalize,
            dataset,
            gamma=0.99,
        )
        model = SAC(
            "MlpPolicy",
            vecnormalize,
            buffer_size=20,
            learning_starts=0,
            batch_size=2,
            seed=0,
            device="cpu",
        )
        count = prefill_replay_buffer(model, dataset, progress_every=0)

        assert count == 5
        assert model.replay_buffer.size() == 5
        assert np.array_equal(
            model.replay_buffer.observations[:5, 0],
            dataset["observations"],
        )
        assert np.array_equal(
            model.replay_buffer.actions[:5, 0],
            dataset["actions"],
        )
        assert model.replay_buffer.dones[4, 0] == 1.0
        assert model.replay_buffer.timeouts[4, 0] == 0.0

        # SB3 must retain the prefilled transitions when online learning starts.
        model.learn(total_timesteps=1)
        assert model.replay_buffer.size() == 6
        assert np.array_equal(
            model.replay_buffer.observations[:5, 0],
            dataset["observations"],
        )
    finally:
        vecnormalize.close()


def test_automatic_buffer_capacity_preserves_all_demonstrations():
    assert resolve_buffer_size(None, 345_000, 600_000, 50_000) == 1_000_000
    assert resolve_buffer_size(None, 575_000, 600_000, 50_000) == 1_300_000
