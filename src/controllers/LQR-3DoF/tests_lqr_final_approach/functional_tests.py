import os
import sys

import numpy as np
import pytest

LQR_ROOT = os.path.dirname(__file__)
while not os.path.exists(os.path.join(LQR_ROOT, "lqr_3dof_final.py")):
    parent_dir = os.path.dirname(LQR_ROOT)
    if parent_dir == LQR_ROOT:
        break
    LQR_ROOT = parent_dir
SRC_PATH = os.path.abspath(os.path.join(LQR_ROOT, "..", ".."))
if LQR_ROOT not in sys.path:
    sys.path.insert(0, LQR_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from lqr_3dof_final import (
    LQRFinalController,
    analyze_saturation,
    build_linearized_3dof_matrices,
    build_port_error_mapping,
    build_Q_final,
    build_R_final,
    compute_lqr_gain,
)
from envs.docking3dof.core import Docking3DoFEnv


def _build_final_controller(env):
    Q_port = build_Q_final()
    R = build_R_final(env)

    A, B = build_linearized_3dof_matrices(env)
    H = build_port_error_mapping(env)

    Q_com = H.T @ Q_port @ H

    K, _ = compute_lqr_gain(A, B, Q_com, R)

    return LQRFinalController(K, H)


@pytest.mark.slow
def test_final_lqr_monte_carlo_rollouts_terminate():
    num_rollouts = 20
    max_time = 5000.0

    env_for_gain = Docking3DoFEnv(phase="final", max_time=max_time, seed=0)
    controller = _build_final_controller(env_for_gain)

    for seed in range(num_rollouts):
        env = Docking3DoFEnv(
            phase="final",
            max_time=max_time,
            seed=seed,
        )
        state = env.reset(mode="testing")

        done = False
        steps = 0
        max_steps = int(max_time / env.dt) + 1
        info = {}

        while not done and steps < max_steps:
            action = controller.compute_action(env, state)
            state, done, info = env.step(action)
            steps += 1

        assert done
        assert info["termination_reason"] is not None
        assert np.all(np.isfinite(state))


@pytest.mark.slow
def test_final_lqr_safety_rate_threshold_initial():
    """
    Initial functional safety test.

    This threshold is intentionally modest before final-approach tuning.
    After tuning, increase the threshold or add a separate success-rate test.
    """
    num_rollouts = 30
    max_time = 5000.0
    max_safety_rate = 0.0

    env_for_gain = Docking3DoFEnv(phase="final", max_time=max_time, seed=0)
    controller = _build_final_controller(env_for_gain)

    safety_violations = 0

    for seed in range(num_rollouts):
        env = Docking3DoFEnv(
            phase="final",
            max_time=max_time,
            seed=seed,
        )
        state = env.reset(mode="testing")

        done = False
        steps = 0
        max_steps = int(max_time / env.dt) + 1
        info = {}

        while not done and steps < max_steps:
            action = controller.compute_action(env, state)
            state, done, info = env.step(action)
            steps += 1

        if info.get("termination_reason") == "safety_violation":
            safety_violations += 1

    safety_rate = safety_violations / float(num_rollouts)

    assert safety_rate <= max_safety_rate


@pytest.mark.slow
def test_final_lqr_success_rate_threshold_after_tuning():
    """
    Use a low threshold initially.

    After tuning the final-approach LQR, update min_success_rate to something
    meaningful, e.g. 0.3, 0.5, or higher depending on the achieved performance.
    """
    num_rollouts = 30
    max_time = 5000.0
    min_success_rate = 0.8

    env_for_gain = Docking3DoFEnv(phase="final", max_time=max_time, seed=0)
    controller = _build_final_controller(env_for_gain)

    successes = 0

    for seed in range(num_rollouts):
        env = Docking3DoFEnv(
            phase="final",
            max_time=max_time,
            seed=seed,
        )
        state = env.reset(mode="testing")

        done = False
        steps = 0
        max_steps = int(max_time / env.dt) + 1
        info = {}

        while not done and steps < max_steps:
            action = controller.compute_action(env, state)
            state, done, info = env.step(action)
            steps += 1

        if info.get("success"):
            successes += 1

    success_rate = successes / float(num_rollouts)

    assert success_rate >= min_success_rate


def test_final_lqr_deterministic_rollout_with_fixed_seed():
    max_time = 5000.0
    seed = 7

    env_a = Docking3DoFEnv(phase="final", max_time=max_time, seed=seed)
    env_b = Docking3DoFEnv(phase="final", max_time=max_time, seed=seed)

    controller_a = _build_final_controller(env_a)
    controller_b = _build_final_controller(env_b)

    state_a = env_a.reset(mode="testing")
    state_b = env_b.reset(mode="testing")

    done_a = False
    done_b = False

    steps = 0
    max_steps = int(max_time / env_a.dt) + 1

    while not done_a and not done_b and steps < max_steps:
        action_a = controller_a.compute_action(env_a, state_a)
        action_b = controller_b.compute_action(env_b, state_b)

        state_a, done_a, _ = env_a.step(action_a)
        state_b, done_b, _ = env_b.step(action_b)

        steps += 1

    times_a, states_a, actions_raw_a, actions_clipped_a, actions_orb_a = (
        env_a.get_history_arrays()
    )
    times_b, states_b, actions_raw_b, actions_clipped_b, actions_orb_b = (
        env_b.get_history_arrays()
    )

    assert np.allclose(times_a, times_b)
    assert np.allclose(states_a, states_b)
    assert np.allclose(actions_raw_a, actions_raw_b)
    assert np.allclose(actions_clipped_a, actions_clipped_b)
    assert np.allclose(actions_orb_a, actions_orb_b)


def test_final_lqr_step_robust_under_clipping():
    env = Docking3DoFEnv(max_time=5_000.0, phase="final", seed=1)
    env.reset(mode="testing")

    action = np.array([10.0, -10.0, 10.0])
    state, _, info = env.step(action)

    assert np.all(np.isfinite(state))
    assert np.all(info["action_body_clipped"] <= env.action_high + 1e-12)
    assert np.all(info["action_body_clipped"] >= env.action_low - 1e-12)


def test_final_lqr_saturation_analysis_returns_finite_values(capsys):
    env = Docking3DoFEnv(phase="final", max_time=300.0, seed=0)
    state = env.reset(mode="testing")

    controller = _build_final_controller(env)

    for _ in range(10):
        action = controller.compute_action(env, state)
        state, done, _ = env.step(action)
        if done:
            break

    stats = analyze_saturation(env)
    capsys.readouterr()

    assert np.isfinite(stats["saturation_fraction_total"])
    assert 0.0 <= stats["saturation_fraction_total"] <= 1.0
    assert stats["saturation_fraction_channels"].shape == (3,)
    assert np.all(np.isfinite(stats["saturation_fraction_channels"]))
