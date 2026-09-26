from pathlib import Path

import pytest

from controllers.run_mixed_full_mission_evaluations import (
    MissionConfiguration,
    PolicyFiles,
    make_env,
    run_phase,
    validate,
)


def test_environment_uses_the_requested_phase_horizon():
    env = make_env("closing", seed=3, dt=0.1, max_time=123.0)
    try:
        assert env.core_env.max_time == 123.0
    finally:
        env.close()


def test_run_phase_rejects_nonpositive_step_limits():
    with pytest.raises(ValueError, match="positive"):
        run_phase(None, None, None, 0)


def test_sac_configuration_requires_a_closing_policy(tmp_path: Path):
    final_policy = PolicyFiles(tmp_path / "final.zip", tmp_path / "final.pkl")
    config = MissionConfiguration(
        key="invalid",
        label="invalid",
        closing_kind="sac",
        closing_dt=1.0,
        final_policy=final_policy,
    )

    with pytest.raises(ValueError, match="closing policy"):
        validate([config])
