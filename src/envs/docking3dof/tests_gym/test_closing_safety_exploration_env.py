import numpy as np

from envs.docking3dof.closing_safety_exploration_env import (
    ClosingSafetyExplorationGymEnv,
    ClosingSafetyExplorationV2GymEnv,
)
from envs.docking3dof.gym_env import Docking3DoFGymEnv


def _set_velocity_unsafe_state(env):
    # Keep the sampled position/target, but exceed both closing hard velocity
    # limits.  A zero command leaves it unsafe after one one-second step.
    env.core_env.state[2] = -0.40
    env.core_env.state[3] = 0.0
    env.core_env.state[5] = 0.0


def test_standard_closing_environment_still_terminates_on_safety_violation():
    env = Docking3DoFGymEnv(
        phase="closing",
        mode="testing",
        closing_reward_version="v15",
        seed=7,
    )
    env.reset(seed=7)
    _set_velocity_unsafe_state(env)

    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert info["failure"] is True
    assert info["failure_reason"] == "safety_violation"
    assert "safety_termination_disabled" not in info


def test_experiment_continues_and_preserves_hard_safety_penalty():
    env = ClosingSafetyExplorationGymEnv(
        mode="testing",
        closing_reward_version="v15",
        seed=7,
    )
    env.reset(seed=7)
    _set_velocity_unsafe_state(env)

    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))

    assert terminated is False
    assert truncated is False
    assert info["failure"] is False
    assert bool(info["unsafe"]) is True
    assert bool(info["velocity"]["unsafe"]) is True
    assert info["safety_violation_observed"] is True
    assert info["safety_violation_ignored_for_termination"] is True
    assert info["observed_failure_reason"] == "safety_violation"
    assert info["termination_reason"] is None
    assert info["reward_components"]["reward_failure"] == -1000.0
    assert info["reward_components"]["safety_penalty_preserved"] is True


def test_experiment_still_terminates_on_p3_success():
    env = ClosingSafetyExplorationGymEnv(
        mode="testing",
        closing_reward_version="v15",
        seed=11,
    )
    env.reset(seed=11)
    env.core_env.state = env.core_env.p3_target.copy()

    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["termination_reason"] == "success_P3_target"


def test_experiment_still_truncates_at_time_limit():
    env = ClosingSafetyExplorationGymEnv(
        mode="testing",
        max_time=1.0,
        closing_reward_version="v15",
        seed=13,
    )
    env.reset(seed=13)

    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))

    assert terminated is False
    assert truncated is True
    assert info["time_limit"] is True
    assert info["termination_reason"] == "timeout"
    assert info["reward_components"]["reward_timeout"] == -200.0


def test_custom_reward_is_isolated_from_standard_v15_reward():
    standard = Docking3DoFGymEnv(
        phase="closing",
        mode="testing",
        closing_reward_version="v15",
        seed=17,
    )
    experiment = ClosingSafetyExplorationGymEnv(
        mode="testing",
        closing_reward_version="v15",
        seed=17,
    )
    standard.reset(seed=17)
    experiment.reset(seed=17)

    action = np.array([0.01, -0.02, 0.03], dtype=np.float32)
    standard_step = standard.step(action)
    experiment_step = experiment.step(action)

    np.testing.assert_allclose(standard_step[0], experiment_step[0])
    assert standard_step[2:4] == experiment_step[2:4]
    assert standard_step[4]["reward_components"]["closing_reward_version"] == "v15"
    assert "reward_scheme" not in standard_step[4]["reward_components"]
    assert experiment_step[4]["reward_components"]["reward_scheme"] == "p3_reach_v1"
    assert (
        experiment_step[4]["reward_components"]["shared_gym_reward_version_used"]
        is False
    )
    assert standard_step[1] != experiment_step[1]


def test_v2_capture_shaping_is_active_at_v1_stall_distance():
    env = ClosingSafetyExplorationV2GymEnv(mode="testing", seed=19)
    env.reset(seed=19)
    state = env.core_env.p3_target.copy()
    state[0] += 164.0

    reward_state = env._p3_reward_state(state)

    assert env.reward_scheme == "p3_reach_v2"
    assert reward_state["precision_gate"] > 0.90
    np.testing.assert_allclose(
        env.capture_scales,
        np.array(
            [
                100.0,
                20.0,
                0.05,
                0.02,
                np.deg2rad(10.0),
                np.deg2rad(0.1),
            ]
        ),
    )


def test_v2_unsafe_step_is_penalized_but_nonterminal():
    env = ClosingSafetyExplorationV2GymEnv(mode="testing", seed=23)
    env.reset(seed=23)
    _set_velocity_unsafe_state(env)

    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))

    assert terminated is False
    assert truncated is False
    assert info["reward_components"]["reward_scheme"] == "p3_reach_v2"
    assert info["reward_components"]["reward_failure"] == -1000.0
    assert info["reward_components"]["safety_penalty_preserved"] is True
    assert "capture_max_ratio" in info["reward_components"]
