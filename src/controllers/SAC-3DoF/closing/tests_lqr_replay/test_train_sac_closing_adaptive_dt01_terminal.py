from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC


SRC_PATH = Path(__file__).resolve().parents[4]
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
CLOSING_PATH = Path(__file__).resolve().parents[1]
if str(CLOSING_PATH) not in sys.path:
    sys.path.insert(0, str(CLOSING_PATH))

from near_p3_reset_wrapper import FixedNearP3ResetWrapper
from train_sac_closing_adaptive_dt01_terminal import (
    DT_SECONDS,
    FAR_ACTION_HOLD_STEPS,
    GAMMA_PER_NEAR_DECISION,
    REWARD_VERSION,
    ResetSpecificTimeLimit,
    configure_model,
    install_deterministic_warmup_sampler,
    make_training_env,
    make_validation_env,
    parse_args,
    remove_deterministic_warmup_sampler,
    validate_args,
)
from train_sac_closing_v15_phase_balanced_replay import resolve_checkpoint_steps


def test_defaults_encode_the_adaptive_terminal_experiment():
    args = parse_args([])
    validate_args(args)

    assert args.training_timesteps == 200_000
    assert args.checkpoint_steps[-1] == args.training_timesteps
    assert args.adaptive_switch_distance_m == 200.0
    assert args.near_p3_reset_fraction == 0.35
    assert args.capture_batch_fraction == 0.50
    assert args.fixed_ent_coef == 1.7e-4
    assert np.isclose(args.gamma, GAMMA_PER_NEAR_DECISION)


def test_training_env_uses_dt01_adaptive_hold_and_augmented_resets():
    args = parse_args(
        [
            "--warmup-near-p3-reset-fraction",
            "1",
            "--near-p3-max-time",
            "1",
        ]
    )
    validate_args(args)
    env = make_training_env(args)
    observation, info = env.reset(seed=77)

    assert observation.shape == (12,)
    assert info["reset_distribution"] == FixedNearP3ResetWrapper.RESET_NEAR_P3
    assert np.isclose(env.unwrapped.core_env.dt, DT_SECONDS)
    assert env.unwrapped.closing_reward_version == REWARD_VERSION
    assert env.unwrapped.action_hold_steps == FAR_ACTION_HOLD_STEPS
    assert env.unwrapped.adaptive_action_hold_distance_m == 200.0

    _, _, _, _, step_info = env.step(np.zeros(3, dtype=np.float32))
    assert step_info["action_hold_steps_requested"] == 1
    assert np.isclose(step_info["control_interval_seconds"], 0.1)
    env.close()


def test_near_reset_time_limit_does_not_shorten_standard_p2_episodes():
    args = parse_args(
        [
            "--warmup-near-p3-reset-fraction",
            "0",
            "--near-p3-max-time",
            "0.1",
        ]
    )
    validate_args(args)
    env = make_training_env(args)
    env.reset(seed=78)
    _, _, terminated, truncated, info = env.step(
        np.zeros(3, dtype=np.float32)
    )

    assert not terminated
    assert not truncated
    assert not info["near_p3_training_timeout"]
    env.close()


def test_near_reset_time_limit_marks_a_training_only_timeout():
    args = parse_args(
        [
            "--warmup-near-p3-reset-fraction",
            "1",
            "--near-p3-max-time",
            "0.1",
        ]
    )
    validate_args(args)
    env = make_training_env(args)
    env.reset(seed=79)
    _, reward, terminated, truncated, info = env.step(
        np.zeros(3, dtype=np.float32)
    )

    assert not terminated
    assert truncated
    assert info["time_limit"]
    assert info["near_p3_training_timeout"]
    assert info["termination_reason"] == "near_p3_training_timeout"
    assert info["reward_components"]["reward_timeout"] == -200.0
    assert np.isfinite(reward)
    env.close()


def test_validation_env_has_no_reset_augmentation_and_matches_deployment():
    args = parse_args([])
    validate_args(args)
    env = make_validation_env(
        args=args,
        seed=100,
        closing_reward_version="ignored",
        action_hold_steps=1,
        adaptive_action_hold_distance_m=None,
    )
    _, info = env.reset(seed=100)

    assert "reset_distribution" not in info
    assert np.isclose(env.core_env.dt, 0.1)
    assert env.action_hold_steps == 10
    assert env.adaptive_action_hold_distance_m == 200.0
    env.close()


def test_model_configuration_fixes_entropy_and_temporal_parameters():
    args = parse_args(
        [
            "--learning-rate",
            "0.00001",
            "--batch-size",
            "256",
            "--train-freq",
            "8",
            "--gradient-steps",
            "2",
            "--fixed-ent-coef",
            "0.0002",
        ]
    )
    validate_args(args)
    env = make_validation_env(args=args, seed=100)
    model = SAC("MlpPolicy", env, ent_coef="auto_0.01", device="cpu")
    configure_model(model, args)

    assert model.learning_starts == 0
    assert model.batch_size == 256
    assert model.train_freq.frequency == 8
    assert model.gradient_steps == 2
    assert np.isclose(model.gamma, GAMMA_PER_NEAR_DECISION)
    assert model.ent_coef_optimizer is None
    assert model.log_ent_coef is None
    assert torch.isclose(model.ent_coef_tensor, torch.tensor(0.0002))
    env.close()


def test_temporary_warmup_sampler_is_removed_before_model_saving(tmp_path):
    args = parse_args([])
    validate_args(args)
    env = make_validation_env(args=args, seed=100)
    model = SAC("MlpPolicy", env, device="cpu")

    install_deterministic_warmup_sampler(model)
    assert "_sample_action" in model.__dict__
    remove_deterministic_warmup_sampler(model)
    assert "_sample_action" not in model.__dict__
    model.save(tmp_path / "serializable_model")
    env.close()


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--near-p3-max-time", "12000"], "smaller than"),
        (["--near-p3-other-max-ratio", "1"], "lie in"),
        (["--experiment-name", "../unsafe"], "unsafe"),
    ],
)
def test_invalid_last_experiment_configurations_are_rejected(arguments, message):
    args = parse_args(arguments)
    with pytest.raises((ValueError, RuntimeError), match=message):
        validate_args(args)


def test_final_training_checkpoint_is_always_added():
    args = parse_args(["--checkpoint-steps", "25000"])
    validate_args(args)

    assert resolve_checkpoint_steps(args) == [25_000, 200_000]
