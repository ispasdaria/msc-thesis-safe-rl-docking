from pathlib import Path
import json
import sys

import numpy as np


SRC_PATH = Path(__file__).resolve().parents[4]
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
CLOSING_PATH = Path(__file__).resolve().parents[1]
if str(CLOSING_PATH) not in sys.path:
    sys.path.insert(0, str(CLOSING_PATH))

from train_sac_closing_dt01 import (
    BASE_GAMMA,
    EXPERIMENT_DT,
    EXPERIMENT_GAMMA,
    build_experiment_config,
    make_env,
    parse_args,
    task_first_checkpoint_rank,
    validate_args,
)
from evaluate_sac_closing_dt01 import (
    ClosingV15Dt01EvaluationEnv,
    ClosingV15DtSafe01EvaluationEnv,
    load_and_validate_training_config,
    parse_args as parse_evaluation_args,
)


def test_dt01_defaults_preserve_physical_time_quantities():
    args = parse_args([])
    validate_args(args)
    config = build_experiment_config(args)

    assert config["dt_seconds"] == 0.1
    assert config["control_frequency_hz"] == 10.0
    assert config["action_hold_steps"] == 1
    assert config["reward"]["version"] == "v15_dt"
    assert config["reward"]["persistent_cost_scale"] == 0.1
    assert config["max_episode_steps"] == 120_000
    assert config["hyperparameters"]["learning_starts"] == 100_000
    assert config["hyperparameters"]["train_freq"] == 160
    assert config["hyperparameters"]["gradient_steps"] == 8
    assert np.isclose(EXPERIMENT_GAMMA, BASE_GAMMA**EXPERIMENT_DT)
    assert np.isclose(
        config["normalization"]["gamma"],
        EXPERIMENT_GAMMA,
    )


def test_dt01_environment_uses_ten_hz_v15_dt_reward():
    env = make_env("testing", seed=41, max_time=12_000.0)
    _, info = env.reset(seed=41)

    assert np.isclose(info["integration_step_seconds"], 0.1)
    assert np.isclose(info["control_interval_seconds"], 0.1)
    assert info["action_hold_steps"] == 1
    assert env.closing_reward_version == "v15_dt"

    _, _, _, _, step_info = env.step(np.zeros(3, dtype=np.float32))
    components = step_info["reward_components"]
    assert components["closing_reward_version"] == "v15_dt"
    assert np.isclose(components["closing_reward_time_scale"], 0.1)
    env.close()


def test_dt01_cli_overrides_are_recorded():
    args = parse_args(
        [
            "--seed",
            "7",
            "--total-timesteps",
            "4000000",
            "--buffer-size",
            "1000000",
            "--eval-seeds",
            "200",
            "201",
        ]
    )
    validate_args(args)
    config = build_experiment_config(args)

    assert config["seed"] == 7
    assert config["total_timesteps"] == 4_000_000
    assert config["simulated_training_time_seconds"] == 400_000.0
    assert config["hyperparameters"]["buffer_size"] == 1_000_000
    assert config["evaluation"]["seeds"] == [200, 201]


def test_explicit_recommended_pilot_cli_is_self_contained():
    args = parse_args(
        [
            "--device",
            "auto",
            "--seed",
            "0",
            "--dt",
            "0.1",
            "--reward-version",
            "v15_dt",
            "--integrator",
            "rk4",
            "--action-hold-steps",
            "1",
            "--total-timesteps",
            "500000",
            "--buffer-size",
            "3000000",
            "--learning-rate",
            "0.00026276728418436003",
            "--batch-size",
            "512",
            "--tau",
            "0.01",
            "--gamma",
            "0.99899547129175",
            "--learning-starts",
            "100000",
            "--train-freq",
            "160",
            "--gradient-steps",
            "8",
            "--ent-coef",
            "auto_0.05",
            "--target-update-interval",
            "1",
            "--net-arch",
            "256",
            "256",
            "--eval-freq",
            "500000",
            "--eval-seeds",
            "100",
            "101",
            "102",
            "103",
            "104",
            "--max-time",
            "12000",
            "--clip-obs",
            "10",
            "--clip-reward",
            "10",
        ]
    )
    validate_args(args)
    config = build_experiment_config(args)
    hyperparameters = config["hyperparameters"]

    assert config["total_timesteps"] == 500_000
    assert config["reward"]["version"] == "v15_dt"
    assert hyperparameters["learning_rate"] == 0.00026276728418436003
    assert hyperparameters["batch_size"] == 512
    assert hyperparameters["tau"] == 0.01
    assert np.isclose(hyperparameters["gamma"], EXPERIMENT_GAMMA)
    assert hyperparameters["learning_starts"] == 100_000
    assert hyperparameters["train_freq"] == 160
    assert hyperparameters["gradient_steps"] == 8
    assert hyperparameters["ent_coef"] == "auto_0.05"
    assert hyperparameters["target_update_interval"] == 1
    assert hyperparameters["policy_network"] == [256, 256]


def test_dt01_performance_evaluator_uses_heldout_ten_hz_mdp(tmp_path):
    evaluation_args = parse_evaluation_args([])
    assert evaluation_args.model_type == "best"
    assert evaluation_args.seeds == list(range(200, 210))

    env = ClosingV15Dt01EvaluationEnv(
        phase="closing",
        mode="testing",
        max_time=12_000.0,
        seed=200,
        closing_reward_version="v15",
        dt=1.0,
    )
    _, info = env.reset(seed=200)
    assert env.closing_reward_version == "v15_dt"
    assert np.isclose(info["integration_step_seconds"], 0.1)
    assert np.isclose(info["control_interval_seconds"], 0.1)
    env.close()

    config_path = tmp_path / "experiment_config.json"
    config = {
        "dt_seconds": 0.1,
        "action_hold_steps": 1,
        "reward": {"version": "v15_dt"},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert load_and_validate_training_config(config_path) == config

    safe_config = {
        "dt_seconds": 0.1,
        "action_hold_steps": 1,
        "reward": {"version": "v15_dt_safe"},
    }
    config_path.write_text(json.dumps(safe_config), encoding="utf-8")
    assert load_and_validate_training_config(config_path) == safe_config

    safe_env = ClosingV15DtSafe01EvaluationEnv(
        phase="closing",
        mode="testing",
        max_time=12_000.0,
        seed=200,
    )
    safe_env.reset(seed=200)
    assert safe_env.closing_reward_version == "v15_dt_safe"
    assert np.isclose(safe_env.core_env.dt, 0.1)
    safe_env.close()


def test_task_first_checkpoint_rank_never_prefers_fast_failure_return():
    fast_failure = {
        "success_rate": 0.0,
        "safety_failure_rate": 1.0,
        "mean_final_max_target_ratio": 200.0,
        "mean_final_position_error_m": 1400.0,
        "mean_return": -1_000.0,
    }
    safer_failure = {
        "success_rate": 0.0,
        "safety_failure_rate": 0.5,
        "mean_final_max_target_ratio": 150.0,
        "mean_final_position_error_m": 1300.0,
        "mean_return": -20_000.0,
    }
    successful = {
        "success_rate": 0.1,
        "safety_failure_rate": 0.9,
        "mean_final_max_target_ratio": 100.0,
        "mean_final_position_error_m": 1000.0,
        "mean_return": -50_000.0,
    }

    assert task_first_checkpoint_rank(safer_failure) > (
        task_first_checkpoint_rank(fast_failure)
    )
    assert task_first_checkpoint_rank(successful) > (
        task_first_checkpoint_rank(safer_failure)
    )
