import copy
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from episode_stratified_replay_buffer import EpisodeStratifiedReplayBuffer
import train_sac_closing_uniform_dt01_actor_transfer as transfer
from train_sac_closing_uniform_dt01_actor_transfer import (
    ACTION_HOLD_STEPS,
    DT_SECONDS,
    FrozenObservationFreshRewardVecNormalize,
    ProtectedActorSAC,
    parse_args,
    tensor_state_sha256,
)
from stable_baselines3.common.vec_env import DummyVecEnv


def make_info(ratio, reset_distribution="near_p3"):
    return {
        "reset_distribution": reset_distribution,
        "reward_components": {
            "p3_axial_ratio": ratio,
            "p3_lateral_ratio": 0.1,
            "p3_vx_ratio": 0.1,
            "p3_vy_ratio": 0.1,
            "p3_attitude_ratio": 0.1,
            "p3_rate_ratio": 0.1,
        },
    }


def make_buffer(**overrides):
    kwargs = {
        "buffer_size": 100,
        "observation_space": gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        ),
        "action_space": gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        ),
        "device": "cpu",
        "target_le_2_capacity": 20,
        "target_2_to_5_capacity": 20,
        "target_5_to_10_capacity": 20,
        "approach_capacity": 20,
        "minimum_completed_episodes": 4,
        "minimum_standard_episodes": 1,
        "minimum_near_p3_episodes": 3,
        "minimum_target_le_2_episodes": 0,
        "minimum_target_2_to_5_episodes": 1,
        "minimum_target_5_to_10_episodes": 1,
        "minimum_approach_episodes": 1,
    }
    kwargs.update(overrides)
    return EpisodeStratifiedReplayBuffer(**kwargs)


def add_transition(buffer, ratio, *, done, reset_distribution="near_p3", marker=0):
    obs = np.full((1, 12), marker, dtype=np.float32)
    next_obs = obs + 1
    action = np.zeros((1, 3), dtype=np.float32)
    reward = np.array([1.0], dtype=np.float32)
    info = make_info(ratio, reset_distribution=reset_distribution)
    if done:
        info["TimeLimit.truncated"] = True
    buffer.add(
        obs,
        next_obs,
        action,
        reward,
        np.array([done]),
        [info],
    )


def test_ratio_strata_are_disjoint_and_include_approach():
    buffer = make_buffer(
        minimum_completed_episodes=0,
        minimum_standard_episodes=0,
        minimum_near_p3_episodes=0,
        minimum_target_2_to_5_episodes=0,
        minimum_target_5_to_10_episodes=0,
        minimum_approach_episodes=0,
    )
    for marker, ratio in enumerate((2.0, 5.0, 10.0, 10.01)):
        add_transition(buffer, ratio, done=True, marker=marker)

    assert buffer.partition_size(buffer.STRATUM_TARGET_LE_2) == 1
    assert buffer.partition_size(buffer.STRATUM_TARGET_2_TO_5) == 1
    assert buffer.partition_size(buffer.STRATUM_TARGET_5_TO_10) == 1
    assert buffer.partition_size(buffer.STRATUM_APPROACH) == 1


def test_replay_readiness_is_gated_by_completed_episode_diversity():
    buffer = make_buffer()
    add_transition(buffer, 20.0, done=True, reset_distribution="standard_p2")
    add_transition(buffer, 4.0, done=True)
    add_transition(buffer, 7.0, done=True)
    assert not buffer.replay_ready

    add_transition(buffer, 4.0, done=True)
    assert buffer.replay_ready
    statistics = buffer.replay_statistics()
    assert statistics["completed_episodes"] == 4
    assert statistics["standard_episodes"] == 1
    assert statistics["near_p3_episodes"] == 3


def test_sampling_redistributes_missing_close_stratum_and_tracks_counts():
    buffer = make_buffer()
    add_transition(buffer, 20.0, done=True, reset_distribution="standard_p2")
    add_transition(buffer, 4.0, done=True)
    add_transition(buffer, 7.0, done=True)
    add_transition(buffer, 4.0, done=True)

    samples = buffer.sample(100)
    assert samples.observations.shape[0] == 100
    counts = buffer.last_sample_counts
    assert counts[buffer.STRATUM_TARGET_LE_2] == 0
    assert sum(counts.values()) == 100
    assert counts[buffer.STRATUM_TARGET_2_TO_5] > counts[buffer.STRATUM_TARGET_5_TO_10]


class TinyEnv(gym.Env):
    metadata = {}

    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(12,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.ones(12, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        obs = np.full(12, 1.0 + 0.1 * self.steps, dtype=np.float32)
        return obs, 1.0, False, self.steps >= 4, {}


def test_frozen_observation_statistics_allow_fresh_reward_statistics():
    vec = FrozenObservationFreshRewardVecNormalize(
        DummyVecEnv([TinyEnv]), gamma=0.99, norm_obs=True, norm_reward=True
    )
    vec.obs_rms.mean[:] = 3.0
    vec.obs_rms.var[:] = 2.0
    obs_count = float(vec.obs_rms.count)
    reward_count = float(vec.ret_rms.count)
    expected_mean = copy.deepcopy(vec.obs_rms.mean)
    expected_var = copy.deepcopy(vec.obs_rms.var)

    vec.reset()
    vec.step(np.zeros((1, 3), dtype=np.float32))
    assert np.array_equal(vec.obs_rms.mean, expected_mean)
    assert np.array_equal(vec.obs_rms.var, expected_var)
    assert float(vec.obs_rms.count) == obs_count
    assert float(vec.ret_rms.count) > reward_count
    vec.close()


def test_actor_is_exactly_frozen_during_critic_only_training():
    vec = DummyVecEnv([TinyEnv])
    model = ProtectedActorSAC(
        "MlpPolicy",
        vec,
        learning_rate=1e-4,
        buffer_size=100,
        learning_starts=0,
        batch_size=8,
        train_freq=1,
        gradient_steps=1,
        ent_coef=1e-3,
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-4,
        actor_update_interval=1,
        device="cpu",
    )
    model.set_actor_updates_enabled(False)
    before = tensor_state_sha256(model.actor)
    critic_before = tensor_state_sha256(model.critic)
    model.learn(total_timesteps=12)

    assert tensor_state_sha256(model.actor) == before
    assert tensor_state_sha256(model.critic) != critic_before
    assert model.actor_gradient_updates == 0
    assert model.critic_gradient_updates > 0
    vec.close()


def test_default_experiment_is_uniform_point_one_second():
    args = parse_args(["--dry-run", "--device", "cpu"])
    assert DT_SECONDS == 0.1
    assert ACTION_HOLD_STEPS == 1
    assert np.isclose(args.gamma, 0.99**0.1)
    assert args.actor_learning_rate < args.critic_learning_rate
    assert args.actor_update_interval == 20


def test_output_directories_require_explicit_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(transfer, "MODEL_ROOT", tmp_path / "models")
    monkeypatch.setattr(transfer, "RESULT_ROOT", tmp_path / "results")
    monkeypatch.setattr(transfer, "LOG_ROOT", tmp_path / "logs")
    args = parse_args(
        ["--dry-run", "--device", "cpu", "--experiment-name", "same_name"]
    )
    existing = transfer.MODEL_ROOT / args.experiment_name
    existing.mkdir(parents=True)
    (existing / "partial.txt").write_text("partial", encoding="utf-8")

    try:
        transfer.prepare_output_directories(args)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing output was reused without --overwrite.")
    assert (existing / "partial.txt").is_file()

    args.overwrite = True
    paths = transfer.prepare_output_directories(args)
    assert not (existing / "partial.txt").exists()
    assert all(path.is_dir() for path in paths)
