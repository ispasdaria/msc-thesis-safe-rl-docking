from pathlib import Path
import sys

import numpy as np
import pytest


SRC_PATH = Path(__file__).resolve().parents[4]
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
CLOSING_PATH = Path(__file__).resolve().parents[1]
if str(CLOSING_PATH) not in sys.path:
    sys.path.insert(0, str(CLOSING_PATH))

from evaluate_sac_closing_timestep_transfer import (
    DEFAULT_VARIANTS,
    Dt01EvaluationEnv,
    NativeOneHzEvaluationEnv,
    VARIANTS,
    build_comparison_rows,
    build_paired_rows,
    parse_args,
    validate_args,
)


def test_defaults_run_the_control_and_both_timestep_transfer_cases():
    args = parse_args([])

    assert tuple(args.variants) == DEFAULT_VARIANTS
    assert args.seeds == list(range(200, 210))
    assert args.max_time == 12_000.0
    assert args.model_path.name == "model_0100000_steps.zip"
    assert args.vecnormalize_path.name == (
        "model_0100000_steps_vecnormalize.pkl"
    )


def test_variant_definitions_isolate_integration_and_policy_cadence():
    native = VARIANTS["native_1hz"]
    held = VARIANTS["dt0p1_hold_1s"]
    recompute = VARIANTS["dt0p1_recompute_10hz"]

    assert native.integration_dt_seconds == 1.0
    assert native.decision_interval_seconds == 1.0
    assert held.integration_dt_seconds == 0.1
    assert held.action_hold_steps == 10
    assert held.decision_interval_seconds == 1.0
    assert recompute.integration_dt_seconds == 0.1
    assert recompute.action_hold_steps == 1
    assert recompute.decision_interval_seconds == 0.1


def test_environment_classes_force_the_intended_mdp_parameters():
    native = NativeOneHzEvaluationEnv(
        phase="closing",
        mode="testing",
        max_time=2.0,
        seed=200,
        dt=0.1,
        integrator="euler",
        closing_reward_version="v15_dt_safe",
        action_hold_steps=1,
    )
    _, native_info = native.reset(seed=200)
    assert np.isclose(native.core_env.dt, 1.0)
    assert native.closing_reward_version == "v15"
    assert native_info["action_hold_steps"] == 1
    native.close()

    held = Dt01EvaluationEnv(
        phase="closing",
        mode="testing",
        max_time=2.0,
        seed=200,
        dt=1.0,
        closing_reward_version="v15",
        action_hold_steps=10,
    )
    _, held_info = held.reset(seed=200)
    assert np.isclose(held.core_env.dt, 0.1)
    assert held.closing_reward_version == "v15_dt"
    assert held_info["action_hold_steps"] == 10
    _, _, _, _, step_info = held.step(np.zeros(3, dtype=np.float32))
    assert step_info["action_hold_steps_executed"] == 10
    assert np.isclose(step_info["action_hold_seconds_executed"], 1.0)
    held.close()


def test_validation_rejects_duplicate_cases_and_unsafe_output_names():
    duplicate = parse_args(
        ["--variants", "native_1hz", "native_1hz"]
    )
    with pytest.raises(ValueError, match="variants must be unique"):
        validate_args(duplicate)

    unsafe_name = parse_args(["--output-name", "../outside"])
    with pytest.raises(ValueError, match="output-name"):
        validate_args(unsafe_name)


def fake_summary(offset):
    return {
        "success_rate": 0.1 + offset,
        "safety_failure_rate": 0.2 + offset,
        "timeout_rate": 0.7 - offset,
        "unsafe_episode_rate": 0.2 + offset,
        "mean_minimum_position_error_m": 10.0 + offset,
        "mean_final_position_error_m": 20.0 + offset,
        "mean_final_max_target_ratio": 3.0 + offset,
        "mean_terminal_port_speed_mps": 0.1 + offset,
        "mean_steps": 100.0 + offset,
    }


def test_comparison_rows_compute_deltas_against_native_control():
    summaries = {
        "native_1hz": fake_summary(0.0),
        "dt0p1_hold_1s": fake_summary(1.0),
    }
    rows = build_comparison_rows(list(summaries), summaries)

    assert rows[0]["delta_from_native__success_rate"] == 0.0
    assert rows[1]["delta_from_native__success_rate"] == 1.0
    assert rows[1]["delta_from_native__mean_final_position_error_m"] == 1.0


def test_paired_rows_keep_each_seed_aligned_across_variants():
    results = {
        "native_1hz": [
            {
                "seed": 201,
                **{metric: f"native_{metric}" for metric in (
                    "success",
                    "safety_failure",
                    "timeout",
                    "termination_reason",
                    "elapsed_time_seconds",
                    "decision_steps",
                    "simulator_steps",
                    "minimum_position_error_m",
                    "final_position_error_m",
                    "final_max_target_ratio",
                    "max_port_speed_mps",
                )},
            }
        ],
        "dt0p1_hold_1s": [
            {
                "seed": 201,
                **{metric: f"held_{metric}" for metric in (
                    "success",
                    "safety_failure",
                    "timeout",
                    "termination_reason",
                    "elapsed_time_seconds",
                    "decision_steps",
                    "simulator_steps",
                    "minimum_position_error_m",
                    "final_position_error_m",
                    "final_max_target_ratio",
                    "max_port_speed_mps",
                )},
            }
        ],
    }
    paired = build_paired_rows(
        [201],
        list(results),
        results,
    )

    assert paired[0]["seed"] == 201
    assert paired[0]["native_1hz__success"] == "native_success"
    assert paired[0]["dt0p1_hold_1s__success"] == "held_success"
