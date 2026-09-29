from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


SRC_PATH = Path(__file__).resolve().parents[5]
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
CLOSING_PATH = Path(__file__).resolve().parents[1]
if str(CLOSING_PATH) not in sys.path:
    sys.path.insert(0, str(CLOSING_PATH))

from her_closing_utils import make_her_env
from lqr_residual_env import ResidualLQRGoalConditionedClosingEnv
from residual_lqr_config import (
    CLOSING_LQR_DT_SECONDS,
    discount_for_dt,
    resolve_environment_dt,
    steps_for_duration,
)
from train_sac_closing_her_residual_lqr import (
    _validate_args as validate_action_residual_args,
    build_parser as build_action_residual_parser,
)
from train_sac_closing_reference_residual_lqr_no_her import (
    build_parser as build_reference_residual_parser,
    validate_args as validate_reference_residual_args,
)


def test_residual_factories_default_to_dt01_and_reject_legacy_dt():
    assert resolve_environment_dt("lqr_residual") == CLOSING_LQR_DT_SECONDS
    assert (
        resolve_environment_dt("lqr_reference_residual")
        == CLOSING_LQR_DT_SECONDS
    )
    assert resolve_environment_dt("v3") == 1.0

    with pytest.raises(ValueError, match="requires the dt=0.1 s closing LQR"):
        resolve_environment_dt("lqr_residual", 1.0)


def test_residual_factory_builds_retuned_dt01_environment():
    env = make_her_env(
        seed=0,
        mode="testing",
        max_time=1.0,
        reward_version="lqr_residual",
    )
    try:
        assert np.isclose(env.core_env.dt, CLOSING_LQR_DT_SECONDS)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
        assert np.isclose(info["persistent_reward_time_scale"], 0.1)
    finally:
        env.close()


def test_action_residual_persistent_reward_is_physical_time_scaled():
    common = {"mode": "testing", "seed": 2, "max_time": 2.0}
    env_1s = ResidualLQRGoalConditionedClosingEnv(dt=1.0, **common)
    env_01s = ResidualLQRGoalConditionedClosingEnv(dt=0.1, **common)
    try:
        observation_1s, _ = env_1s.reset(seed=2)
        observation_01s, _ = env_01s.reset(seed=2)
        reward_1s = env_1s.compute_reward(
            observation_1s["achieved_goal"],
            observation_1s["desired_goal"],
            {},
        )
        reward_01s = env_01s.compute_reward(
            observation_01s["achieved_goal"],
            observation_01s["desired_goal"],
            {},
        )
        assert np.isclose(reward_01s, 0.1 * reward_1s)
    finally:
        env_1s.close()
        env_01s.close()


def test_physical_time_conversions_are_explicit():
    assert steps_for_duration(12_000.0) == 120_000
    assert np.isclose(discount_for_dt(0.9999), 0.9999**0.1)


def test_action_residual_cli_defaults_are_dt01_consistent():
    parser = build_action_residual_parser()
    args = parser.parse_args([])
    validate_action_residual_args(parser, args)

    assert args.max_eval_steps == 120_000
    assert np.isclose(args.gamma, 0.999**0.1)
    assert "dt0p1" in args.experiment_name


def test_reference_residual_cli_exposes_only_dt01():
    parser = build_reference_residual_parser()
    args = parser.parse_args([])
    validate_reference_residual_args(parser, args)

    assert args.dt == CLOSING_LQR_DT_SECONDS
    assert args.max_eval_steps == 120_000
    assert np.isclose(args.gamma, 0.9999**0.1)

    with pytest.raises(SystemExit):
        parser.parse_args(["--dt", "1.0"])
