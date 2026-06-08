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


def test_build_linearized_3dof_matrices_entries_final():
    env = Docking3DoFEnv(phase="final")
    A, B = build_linearized_3dof_matrices(env)

    assert A.shape == (6, 6)
    assert B.shape == (6, 3)

    assert A[0, 2] == pytest.approx(1.0)
    assert A[1, 3] == pytest.approx(1.0)
    assert A[3, 1] == pytest.approx(-env.n**2)

    assert B[2, 0] == pytest.approx(1.0)
    assert B[3, 1] == pytest.approx(1.0)
    assert B[5, 2] == pytest.approx(1.0)


def test_Q_com_is_symmetric_psd():
    env = Docking3DoFEnv(phase="final")

    Q_port = build_Q_final()
    H = build_port_error_mapping(env)

    Q_com = H.T @ Q_port @ H

    assert Q_com.shape == (6, 6)
    assert np.allclose(Q_com, Q_com.T)

    eigvals = np.linalg.eigvalsh(Q_com)
    assert np.all(eigvals >= -1e-10)


def test_linearized_pair_is_controllable_final():
    env = Docking3DoFEnv(phase="final")
    A, B = build_linearized_3dof_matrices(env)

    n = A.shape[0]

    controllability = B
    for i in range(1, n):
        controllability = np.hstack(
            [controllability, np.linalg.matrix_power(A, i) @ B]
        )

    rank = np.linalg.matrix_rank(controllability)

    assert rank == n


def test_linearized_pair_is_stabilizable_pbh_final():
    env = Docking3DoFEnv(phase="final")
    A, B = build_linearized_3dof_matrices(env)

    n = A.shape[0]
    eigvals = np.linalg.eigvals(A)
    tol = 1e-9

    for lam in eigvals:
        if np.real(lam) >= -tol:
            pbh = np.hstack([A - lam * np.eye(n), B]).astype(complex)
            assert np.linalg.matrix_rank(pbh, tol=tol) == n


def test_linearized_pair_is_detectable_from_Q_com_pbh_final():
    env = Docking3DoFEnv(phase="final")

    A, _ = build_linearized_3dof_matrices(env)
    H = build_port_error_mapping(env)
    Q_port = build_Q_final()
    Q_com = H.T @ Q_port @ H

    n = A.shape[0]
    eigvals = np.linalg.eigvals(A)
    tol = 1e-9

    eigvals_Q, eigvecs_Q = np.linalg.eigh(0.5 * (Q_com + Q_com.T))
    eigvals_Q = np.clip(eigvals_Q, 0.0, None)
    C = (eigvecs_Q * np.sqrt(eigvals_Q)) @ eigvecs_Q.T

    for lam in eigvals:
        if np.real(lam) >= -tol:
            pbh = np.vstack([A - lam * np.eye(n), C]).astype(complex)
            assert np.linalg.matrix_rank(pbh, tol=tol) == n


def test_compute_lqr_gain_final_stable_and_shapes():
    env = Docking3DoFEnv(phase="final")

    A, B = build_linearized_3dof_matrices(env)
    H = build_port_error_mapping(env)
    Q_port = build_Q_final()
    R = build_R_final(env)

    Q_com = H.T @ Q_port @ H

    K, P = compute_lqr_gain(A, B, Q_com, R)

    assert K.shape == (3, 6)
    assert P.shape == (6, 6)

    cl_eigs = np.linalg.eigvals(A - B @ K)
    assert np.all(np.real(cl_eigs) < 0.0)


def test_compute_lqr_gain_rejects_non_pd_R_final():
    env = Docking3DoFEnv(phase="final")

    A, B = build_linearized_3dof_matrices(env)
    H = build_port_error_mapping(env)
    Q_port = build_Q_final()
    Q_com = H.T @ Q_port @ H

    R_bad = np.zeros((3, 3))

    with pytest.raises(ValueError, match="R must be positive definite"):
        compute_lqr_gain(A, B, Q_com, R_bad)


def test_compute_lqr_gain_rejects_non_psd_Q_final():
    env = Docking3DoFEnv(phase="final")

    A, B = build_linearized_3dof_matrices(env)
    R = build_R_final(env)

    Q_bad = build_Q_final()
    Q_bad[0, 0] = -1.0

    with pytest.raises(ValueError, match="Q must be positive semidefinite"):
        compute_lqr_gain(A, B, Q_bad, R)


def test_compute_lqr_gain_rejects_undetectable_Q_final():
    env = Docking3DoFEnv(phase="final")

    A, B = build_linearized_3dof_matrices(env)
    R = build_R_final(env)

    Q_bad = np.zeros((6, 6))

    with pytest.raises(ValueError, match="not detectable"):
        compute_lqr_gain(A, B, Q_bad, R)


def test_compute_lqr_gain_rejects_non_stabilizable_B_final():
    env = Docking3DoFEnv(phase="final")

    A, B = build_linearized_3dof_matrices(env)
    H = build_port_error_mapping(env)
    Q_port = build_Q_final()
    R = build_R_final(env)

    Q_com = H.T @ Q_port @ H

    B_bad = B.copy()
    B_bad[:, 2] = 0.0  # remove angular acceleration authority

    with pytest.raises(ValueError, match="not stabilizable"):
        compute_lqr_gain(A, B_bad, Q_com, R)


def test_controller_rejects_wrong_phase_final():
    controller = LQRFinalController(np.zeros((3, 6)), np.eye(6))
    env = Docking3DoFEnv(phase="closing")

    with pytest.raises(ValueError, match="final phase"):
        controller.compute_action(env, np.zeros(6))


def test_controller_rejects_invalid_state_shape_final():
    controller = LQRFinalController(np.zeros((3, 6)), np.eye(6))
    env = Docking3DoFEnv(phase="final")

    with pytest.raises(ValueError, match="state must have shape"):
        controller.compute_action(env, np.zeros(5))


def test_controller_rejects_bad_K_shape_final():
    with pytest.raises(ValueError, match="K must have shape"):
        LQRFinalController(np.zeros((2, 6)), np.eye(6))


def test_controller_rejects_bad_H_shape_final():
    with pytest.raises(ValueError, match="H must have shape"):
        LQRFinalController(np.zeros((3, 6)), np.eye(5))


def test_controller_action_sanity_final():
    env = Docking3DoFEnv(phase="final", seed=1)
    state = env.reset(mode="testing")

    controller = _build_final_controller(env)

    action = controller.compute_action(env, state)

    assert action.shape == (3,)
    assert np.all(np.isfinite(action))

    clipped = env.clip_action(action)

    assert np.all(clipped <= env.action_high + 1e-12)
    assert np.all(clipped >= env.action_low - 1e-12)