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
    analyze_saturation,
    build_linearized_3dof_matrices,
    build_Q_closing,
    build_R_closing,
    closing_error,
    compute_lqr_gain,
    orbital_to_body_action,
)
from envs.docking3dof.core import Docking3DoFEnv


def _finite_difference_jacobian(fun, x, eps):
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(fun(x), dtype=float)
    jac = np.zeros((f0.size, x.size), dtype=float)

    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        fp = np.asarray(fun(x + dx), dtype=float)
        fm = np.asarray(fun(x - dx), dtype=float)
        jac[:, i] = (fp - fm) / (2.0 * eps)

    return jac


def test_build_Q_closing_is_psd():
    Q = build_Q_closing()

    assert Q.shape == (6, 6)
    assert np.allclose(Q, Q.T)

    eigvals = np.linalg.eigvalsh(Q)
    assert np.all(eigvals >= 0.0)


def test_build_R_closing_is_pd():
    env = Docking3DoFEnv()
    R = build_R_closing(env)

    assert R.shape == (3, 3)
    assert np.allclose(R, R.T)

    eigvals = np.linalg.eigvalsh(R)
    assert np.all(eigvals > 0.0)


def test_build_linearized_3dof_matrices_entries():
    env = Docking3DoFEnv()
    A, B = build_linearized_3dof_matrices(env)

    assert A.shape == (6, 6)
    assert B.shape == (6, 3)

    assert A[0, 2] == pytest.approx(1.0)
    assert A[1, 3] == pytest.approx(1.0)
    assert A[3, 1] == pytest.approx(-env.n**2)

    assert B[2, 0] == pytest.approx(1.0)
    assert B[3, 1] == pytest.approx(1.0)
    assert B[5, 2] == pytest.approx(1.0)


def test_linear_model_matches_local_dynamics():
    env = Docking3DoFEnv(phase="closing")
    A, B = build_linearized_3dof_matrices(env)

    state = np.array([1e-3, -2e-3, 0.0, 0.0, 0.0, 0.0])
    action_body = np.array([0.0, 0.0, 0.0])

    def dynamics_state(s):
        return env.dynamics(s, action_body)

    def dynamics_action(u):
        return env.dynamics(state, u)

    A_fd = _finite_difference_jacobian(dynamics_state, state, eps=1e-4)
    B_fd = _finite_difference_jacobian(dynamics_action, action_body, eps=1e-4)

    u_orbital_from_body = np.diag([-1.0, -1.0, 1.0])
    B_body = B @ u_orbital_from_body

    assert np.allclose(A_fd, A, atol=1e-5)
    assert np.allclose(B_fd, B_body, atol=1e-5)


def test_closing_error_wraps_psi():
    env = Docking3DoFEnv()

    x = np.array([0.0, 0.0, 0.0, 0.0, np.pi - 0.01, 0.0])
    x_ref = np.array([0.0, 0.0, 0.0, 0.0, -np.pi + 0.01, 0.0])

    e = closing_error(env, x, x_ref)

    assert e.shape == (6,)
    assert e[4] == pytest.approx(-0.02)


def test_orbital_to_body_action_zero_angle():
    env = Docking3DoFEnv()
    action_orbital = np.array([0.1, -0.2, 0.3])

    action_body = orbital_to_body_action(env, action_orbital, psi=0.0)

    expected = np.array([-0.1, 0.2, 0.3])
    assert np.allclose(action_body, expected)


def test_orbital_to_body_action_pi_over_2():
    env = Docking3DoFEnv()
    action_orbital = np.array([0.1, 0.2, -0.3])

    action_body = orbital_to_body_action(env, action_orbital, psi=np.pi / 2)

    expected = np.array([-0.2, 0.1, -0.3])
    assert np.allclose(action_body, expected)


def test_compute_lqr_gain_stable_and_shapes():
    env = Docking3DoFEnv()
    A, B = build_linearized_3dof_matrices(env)
    Q = build_Q_closing()
    R = build_R_closing(env)

    K, P = compute_lqr_gain(A, B, Q, R)

    assert K.shape == (3, 6)
    assert P.shape == (6, 6)

    cl_eigs = np.linalg.eigvals(A - B @ K)
    assert np.all(np.real(cl_eigs) < 0.0)


def test_compute_lqr_gain_rejects_non_pd_R():
    env = Docking3DoFEnv()
    A, B = build_linearized_3dof_matrices(env)
    Q = build_Q_closing()
    R_bad = np.zeros((3, 3))

    with pytest.raises(ValueError, match="R must be positive definite"):
        compute_lqr_gain(A, B, Q, R_bad)


def test_compute_lqr_gain_rejects_non_psd_Q():
    env = Docking3DoFEnv()
    A, B = build_linearized_3dof_matrices(env)
    Q_bad = build_Q_closing()
    Q_bad[0, 0] = -1.0
    R = build_R_closing(env)

    with pytest.raises(ValueError, match="Q must be positive semidefinite"):
        compute_lqr_gain(A, B, Q_bad, R)


def test_linearized_pair_is_controllable():
    env = Docking3DoFEnv(phase="closing")
    A, B = build_linearized_3dof_matrices(env)

    n = A.shape[0]

    controllability = B
    for i in range(1, n):
        controllability = np.hstack((controllability, np.linalg.matrix_power(A, i) @ B))

    rank = np.linalg.matrix_rank(controllability)

    assert rank == n


def test_linearized_pair_is_stabilizable_pbh():
    env = Docking3DoFEnv(phase="closing")
    A, B = build_linearized_3dof_matrices(env)

    n = A.shape[0]
    eigvals = np.linalg.eigvals(A)

    tol = 1e-9

    for lam in eigvals:
        if np.real(lam) >= -tol:
            pbh = np.hstack((A - lam * np.eye(n), B)).astype(complex)
            assert np.linalg.matrix_rank(pbh, tol=tol) == n


def test_linearized_pair_is_detectable_from_Q_pbh():
    env = Docking3DoFEnv(phase="closing")
    A, _ = build_linearized_3dof_matrices(env)
    Q = build_Q_closing()

    n = A.shape[0]
    eigvals = np.linalg.eigvals(A)

    # For diagonal positive definite Q, this is simply sqrt of diagonal.
    C = np.diag(np.sqrt(np.diag(Q)))

    tol = 1e-9

    for lam in eigvals:
        if np.real(lam) >= -tol:
            pbh = np.vstack((A - lam * np.eye(n), C)).astype(complex)
            assert np.linalg.matrix_rank(pbh, tol=tol) == n


def test_controller_rejects_wrong_phase():
    controller = LQRClosingController(np.zeros((3, 6)))
    env = Docking3DoFEnv(phase="final")

    with pytest.raises(ValueError, match="closing phase"):
        controller.compute_action(env, np.zeros(6))


def test_controller_requires_p3_target():
    controller = LQRClosingController(np.zeros((3, 6)))
    env = Docking3DoFEnv(phase="closing")

    with pytest.raises(RuntimeError, match="p3_target is None"):
        controller.compute_action(env, np.zeros(6))


def test_controller_rejects_invalid_state_shape():
    controller = LQRClosingController(np.zeros((3, 6)))
    env = Docking3DoFEnv(phase="closing")
    env.reset(mode="testing")

    with pytest.raises(ValueError, match="state must have shape"):
        controller.compute_action(env, np.zeros(5))


def test_controller_action_sanity():
    env = Docking3DoFEnv(phase="closing", seed=1)
    state = env.reset(mode="testing")

    Q = build_Q_closing()
    R = build_R_closing(env)
    A, B = build_linearized_3dof_matrices(env)
    K, _ = compute_lqr_gain(A, B, Q, R)

    controller = LQRClosingController(K)
    action = controller.compute_action(env, state)

    assert action.shape == (3,)
    assert np.all(np.isfinite(action))

    clipped = env.clip_action(action)
    assert np.all(clipped <= env.action_high + 1e-12)
    assert np.all(clipped >= env.action_low - 1e-12)


def test_analyze_saturation_empty_history(capsys):
    env = Docking3DoFEnv()

    stats = analyze_saturation(env)
    capsys.readouterr()

    assert stats["saturation_fraction_total"] == 0.0
    assert np.allclose(stats["saturation_fraction_channels"], np.zeros(3))


def test_analyze_saturation_detects_clipping(capsys):
    env = Docking3DoFEnv()
    env.reset(mode="testing")

    action = np.array([10.0, -10.0, 10.0])
    env.step(action)

    stats = analyze_saturation(env)
    capsys.readouterr()

    assert stats["saturation_fraction_total"] > 0.0
    assert np.all(stats["max_raw"] >= stats["max_clipped"])


def test_saturation_analysis_returns_finite_values_after_rollout(capsys):
    env = Docking3DoFEnv(phase="closing", max_time=300.0, seed=0)
    state = env.reset(mode="testing")

    Q = build_Q_closing()
    R = build_R_closing(env)
    A, B = build_linearized_3dof_matrices(env)
    K, _ = compute_lqr_gain(A, B, Q, R)
    controller = LQRClosingController(K)

    for _ in range(10):
        action = controller.compute_action(env, state)
        state, done, _ = env.step(action)
        if done:
            break

    stats = analyze_saturation(env)
    capsys.readouterr()

    assert np.isfinite(stats["saturation_fraction_total"])
    assert stats["saturation_fraction_total"] >= 0.0
    assert stats["saturation_fraction_total"] <= 1.0


def test_monte_carlo_rollouts_terminate():
    num_rollouts = 50
    max_time = 600.0

    env_for_gain = Docking3DoFEnv(phase="closing", max_time=max_time, seed=0)
    Q = build_Q_closing()
    R = build_R_closing(env_for_gain)
    A, B = build_linearized_3dof_matrices(env_for_gain)
    K, _ = compute_lqr_gain(A, B, Q, R)
    controller = LQRClosingController(K)

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

        assert done
        assert info["termination_reason"] is not None
        assert np.all(np.isfinite(state))
