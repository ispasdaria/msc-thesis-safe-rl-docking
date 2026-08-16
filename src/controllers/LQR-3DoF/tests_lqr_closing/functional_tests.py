import os
import sys

import numpy as np
import pytest

# Allow imports from the repo's src/ directory when running pytest from the root.
LQR_ROOT = os.path.dirname(__file__)
while not os.path.exists(os.path.join(LQR_ROOT, "lqr_3dof_closing.py")):
    parent_dir = os.path.dirname(LQR_ROOT)
    if parent_dir == LQR_ROOT:
        break
    LQR_ROOT = parent_dir
SRC_PATH = os.path.abspath(os.path.join(LQR_ROOT, "..", ".."))
if LQR_ROOT not in sys.path:
    sys.path.insert(0, LQR_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from lqr_3dof_closing import (
    LQRClosingController,
    build_linearized_3dof_matrices,
    build_Q_closing,
    build_R_closing,
    compute_lqr_gain,
    orbital_to_body_action,
)
from envs.docking3dof.core import Docking3DoFEnv


def _build_controller(env):
    Q = build_Q_closing()
    R = build_R_closing(env)
    A, B = build_linearized_3dof_matrices(env)
    K, _ = compute_lqr_gain(A, B, Q, R)

    return LQRClosingController(K)


def test_success_rate_threshold():
    num_rollouts = 50
    max_time = 12000.0
    min_success_rate = 0.5

    env_for_gain = Docking3DoFEnv(phase="closing", max_time=max_time, seed=0)
    controller = _build_controller(env_for_gain)

    successes = 0

    for seed in range(num_rollouts):
        env = Docking3DoFEnv(
            phase="closing",
            max_time=max_time,
            seed=seed,
        )
        state = env.reset(mode="testing")

        done = False
        steps = 0
        max_steps = int(max_time / env.dt) + 1

        while not done and steps < max_steps:
            action = controller.compute_action(env, state)
            state, done, info = env.step(action)
            steps += 1

        if info.get("success"):
            successes += 1

    success_rate = successes / num_rollouts

    assert success_rate >= min_success_rate


def test_nominal_closing_rollout_succeeds():
    max_time = 12000.0

    env = Docking3DoFEnv(
        phase="closing",
        max_time=max_time,
        seed=0,
    )
    state = env.reset(mode="nominal")

    Q = build_Q_closing()
    R = build_R_closing(env)
    A, B = build_linearized_3dof_matrices(env)
    K, _ = compute_lqr_gain(A, B, Q, R)
    controller = LQRClosingController(K)

    done = False
    steps = 0
    max_steps = int(max_time / env.dt) + 1

    while not done and steps < max_steps:
        action = controller.compute_action(env, state)
        state, done, info = env.step(action)
        steps += 1

    assert done
    assert info["success"]


def test_qr_scaling_keeps_closed_loop_stable():
    env = Docking3DoFEnv(phase="closing")
    A, B = build_linearized_3dof_matrices(env)

    for q_scale in [0.5, 2.0]:
        for r_scale in [0.5, 2.0]:
            Q = build_Q_closing() * q_scale
            R = build_R_closing(env) * r_scale

            K, _ = compute_lqr_gain(A, B, Q, R)

            cl_eigs = np.linalg.eigvals(A - B @ K)
            assert np.all(np.real(cl_eigs) < 0.0)


def test_orbital_to_body_inverse_transform():
    env = Docking3DoFEnv()
    psi = 0.3
    action_orbital = np.array([0.05, -0.02, 0.01])

    action_body = orbital_to_body_action(env, action_orbital, psi=psi)
    action_orbital_back = env.transform_action_to_orbital_frame(action_body, psi=psi)

    assert np.allclose(action_orbital_back, action_orbital)


def test_step_robust_under_clipping():
    env = Docking3DoFEnv(phase="closing", seed=1)
    env.reset(mode="testing")

    action = np.array([10.0, -10.0, 10.0])
    state, _, info = env.step(action)

    assert np.all(np.isfinite(state))
    assert np.all(info["action_body_clipped"] <= env.action_high + 1e-12)
    assert np.all(info["action_body_clipped"] >= env.action_low - 1e-12)


def test_deterministic_rollout_with_fixed_seed():
    max_time = 120.0
    seed = 7

    env_a = Docking3DoFEnv(phase="closing", max_time=max_time, seed=seed)
    env_b = Docking3DoFEnv(phase="closing", max_time=max_time, seed=seed)

    controller_a = _build_controller(env_a)
    controller_b = _build_controller(env_b)

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
