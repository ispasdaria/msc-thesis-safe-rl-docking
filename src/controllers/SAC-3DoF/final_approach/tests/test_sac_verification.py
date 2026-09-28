"""Fast verification tests for the final-approach SAC interface and pipeline."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


REPO_ROOT = Path(__file__).resolve().parents[5]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs.docking3dof.gym_env import Docking3DoFGymEnv


def make_env(*, seed=0, max_time=4.0, mode="training"):
    return Docking3DoFGymEnv(
        phase="final", mode=mode, dt=1.0, max_time=max_time,
        normalize_action=True, seed=seed,
    )


def make_vec(*, seed=0, max_time=4.0):
    return VecNormalize(
        DummyVecEnv([lambda: make_env(seed=seed, max_time=max_time)]),
        norm_obs=True, norm_reward=True, gamma=0.98,
    )


def make_model(env, *, seed=0, learning_starts=4, train_freq=2,
               gradient_steps=2):
    return SAC(
        "MlpPolicy", env, seed=seed, device="cpu", verbose=0,
        learning_starts=learning_starts, train_freq=train_freq,
        gradient_steps=gradient_steps, batch_size=2, buffer_size=64,
        policy_kwargs={"net_arch": [8]}, ent_coef="auto_0.1",
    )


def load_evaluator():
    path = Path(__file__).parents[1] / "evaluate_sac_final_rollouts.py"
    spec = importlib.util.spec_from_file_location("evaluate_sac_final_rollouts_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_final_approach_scripts_use_explicit_environment_horizons():
    final_approach_dir = Path(__file__).parents[1]
    scripts = list(final_approach_dir.glob("*.py"))
    scripts.extend((final_approach_dir / "plotters").glob("*.py"))

    offenders = [
        path.name
        for path in scripts
        if "max_time=None" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_environment_contract_observations_and_action_normalization():
    env = make_env(seed=7)
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=7)
    assert obs.shape == env.observation_space.shape == (6,)
    assert obs.dtype == env.observation_space.dtype == np.float32
    assert np.isfinite(obs).all() and env.observation_space.contains(obs)
    assert info["observation_labels"] == env.final_observation_labels

    limits = env.get_physical_action_limits()
    for scalar in (-1.0, 0.0, 1.0):
        normalized = np.full(3, scalar)
        assert np.allclose(
            env.normalized_action_to_physical(normalized), scalar * limits
        )
    assert np.allclose(
        env.physical_action_to_normalized(limits), np.ones(3)
    )
    env.close()


def test_reset_seed_reproducibility_and_variation():
    env = make_env(seed=0)
    first, _ = env.reset(seed=123)
    repeated, _ = env.reset(seed=123)
    different, _ = env.reset(seed=124)
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, different)
    env.close()


@pytest.mark.parametrize(
    ("core_info", "expected"),
    [
        ({"success": True, "failure": False, "time_limit": False}, (True, False)),
        ({"success": False, "failure": True, "time_limit": False}, (True, False)),
        ({"success": False, "failure": False, "time_limit": True}, (False, True)),
    ],
    ids=("success", "hard-safety", "timeout"),
)
def test_termination_signals(monkeypatch, core_info, expected):
    env = make_env(seed=3)
    env.reset(seed=3)
    base_info = env.core_env.monitor_safety(env.core_env.state)
    base_info.update(core_info)
    base_info.setdefault("termination_reason", "forced-test-event")

    def forced_step(_action):
        return env.core_env.state.copy(), True, dict(base_info)

    monkeypatch.setattr(env.core_env, "step", forced_step)
    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert (terminated, truncated) == expected
    assert info["termination_reason"] == "forced-test-event"
    env.close()


def test_vectorized_environment_and_vecnormalize_roundtrip(tmp_path):
    vec = make_vec(seed=5)
    before = vec.obs_rms.count
    obs = vec.reset()
    obs, _, _, _ = vec.step(np.zeros((1, 3), dtype=np.float32))
    assert obs.shape == (1, 6) and np.isfinite(obs).all()
    assert vec.obs_rms.count > before

    stats = tmp_path / "vecnormalize.pkl"
    vec.save(stats)
    raw = DummyVecEnv([lambda: make_env(seed=5)])
    restored = VecNormalize.load(stats, raw)
    assert np.allclose(restored.obs_rms.mean, vec.obs_rms.mean)
    assert np.allclose(restored.obs_rms.var, vec.obs_rms.var)
    restored.close()
    vec.close()


class TrainingProbe(BaseCallback):
    def __init__(self):
        super().__init__()
        self.first_update_at = None
        self.max_updates = 0
        self.logged = {}

    def _on_step(self):
        updates = int(self.model._n_updates)
        if updates and self.first_update_at is None:
            self.first_update_at = self.num_timesteps
        self.max_updates = max(self.max_updates, updates)
        self.logged.update(self.model.logger.name_to_value)
        return True


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory):
    artifact_dir = tmp_path_factory.mktemp("sac-artifacts")
    vec = make_vec(seed=11, max_time=3.0)
    model = make_model(vec, seed=11, learning_starts=4,
                       train_freq=2, gradient_steps=2)
    probe = TrainingProbe()
    model.learn(total_timesteps=12, callback=probe, log_interval=1)
    model_path = artifact_dir / "checkpoint"
    stats_path = artifact_dir / "vecnormalize.pkl"
    model.save(model_path)
    vec.save(stats_path)
    yield model, vec, probe, model_path, stats_path
    vec.close()


def test_short_training_replay_schedule_entropy_losses_and_outputs(trained_artifacts):
    model, _vec, probe, model_path, stats_path = trained_artifacts
    assert model.replay_buffer.size() == model.num_timesteps == 12
    assert probe.first_update_at is not None
    assert probe.first_update_at > model.learning_starts
    assert model.train_freq.frequency == 2 and model.gradient_steps == 2
    # Four train calls execute two gradient steps each. The callback runs
    # before the training call at the end of its rollout, hence it sees six.
    assert model._n_updates == 8
    assert probe.max_updates == 6
    assert (model_path.with_suffix(".zip")).is_file() and stats_path.is_file()

    ent_coef = torch.exp(model.log_ent_coef.detach()).cpu().item()
    assert np.isfinite(ent_coef) and ent_coef > 0
    required = {"train/actor_loss", "train/critic_loss", "train/ent_coef"}
    assert required <= probe.logged.keys()
    assert all(np.isfinite(float(probe.logged[key])) for key in required)


def test_model_reload_prediction_bounds_and_equivalence(trained_artifacts):
    model, vec, _probe, model_path, stats_path = trained_artifacts
    raw_obs = vec.get_original_obs().copy()
    normalized = vec.normalize_obs(raw_obs)
    before, _ = model.predict(normalized, deterministic=True)

    restored_vec = VecNormalize.load(
        stats_path, DummyVecEnv([lambda: make_env(seed=11, max_time=3.0)])
    )
    restored_model = SAC.load(model_path, env=restored_vec, device="cpu")
    after, _ = restored_model.predict(
        restored_vec.normalize_obs(raw_obs), deterministic=True
    )
    assert np.all(before >= -1.0) and np.all(before <= 1.0)
    assert np.allclose(before, after, rtol=0, atol=1e-7)
    restored_vec.close()


def test_deterministic_evaluation_is_repeatable_and_freezes_statistics(
    trained_artifacts, monkeypatch
):
    model, vec, _probe, _model_path, _stats_path = trained_artifacts
    evaluator = load_evaluator()
    monkeypatch.setattr(
        evaluator, "make_raw_eval_env",
        lambda seed=0, mode="testing": make_env(
            seed=seed, mode=mode, max_time=2.0
        ),
    )
    vec.training = False
    vec.norm_reward = False
    mean_before, var_before, count_before = (
        vec.obs_rms.mean.copy(), vec.obs_rms.var.copy(), vec.obs_rms.count
    )
    first = evaluator.run_one_rollout(model, vec, seed=99, max_steps=2)
    second = evaluator.run_one_rollout(model, vec, seed=99, max_steps=2)
    assert np.allclose(first["normalized_actions"], second["normalized_actions"])
    assert np.allclose(first["states"], second["states"])
    assert first["total_reward"] == pytest.approx(second["total_reward"])
    assert np.array_equal(mean_before, vec.obs_rms.mean)
    assert np.array_equal(var_before, vec.obs_rms.var)
    assert count_before == vec.obs_rms.count


def test_evaluation_requests_deterministic_predictions(monkeypatch):
    evaluator = load_evaluator()
    vec = make_vec(seed=2, max_time=1.0)
    vec.training = False
    calls = []

    class RecordingModel:
        def predict(self, observation, deterministic=False):
            calls.append(deterministic)
            return np.zeros((1, 3), dtype=np.float32), None

    monkeypatch.setattr(
        evaluator, "make_raw_eval_env",
        lambda seed=0, mode="testing": make_env(
            seed=seed, mode=mode, max_time=1.0
        ),
    )
    evaluator.run_one_rollout(RecordingModel(), vec, seed=2, max_steps=1)
    assert calls and all(calls)
    vec.close()


def test_seeded_sac_initialization_is_reproducible():
    vec_a, vec_b = make_vec(seed=17), make_vec(seed=17)
    model_a = make_model(vec_a, seed=17)
    model_b = make_model(vec_b, seed=17)
    obs_a, obs_b = vec_a.reset(), vec_b.reset()
    action_a, _ = model_a.predict(obs_a, deterministic=True)
    action_b, _ = model_b.predict(obs_b, deterministic=True)
    assert np.array_equal(obs_a, obs_b)
    assert np.allclose(action_a, action_b, rtol=0, atol=1e-7)
    vec_a.close()
    vec_b.close()
