import numpy as np
import pytest

from envs.docking3dof.core import Docking3DoFEnv
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


def test_build_Q_closing_is_psd():
    Q = build_Q_closing()

    assert Q.shape == (6, 6)
    assert np.allclose(Q, Q.T)
    assert np.all(np.linalg.eigvalsh(Q) >= 0.0)


def test_build_Q_closing_selects_only_supported_dt_weights():
    Q_default = build_Q_closing()
    Q_dt1 = build_Q_closing(dt=1.0)
    Q_dt01 = build_Q_closing(dt=0.1)

    assert np.array_equal(Q_default, Q_dt1)
    assert Q_dt1[3, 3] == pytest.approx(845_000.0)
    assert Q_dt01[3, 3] == pytest.approx(718_250.0)
    assert np.array_equal(np.delete(np.diag(Q_dt01), 3), np.delete(np.diag(Q_dt1), 3))

    with pytest.raises(ValueError, match="Unsupported closing-controller dt"):
        build_Q_closing(dt=0.2)


def test_build_R_closing_is_pd():
    env = Docking3DoFEnv(max_time=12_000.0)
    R = build_R_closing(env)

    assert R.shape == (3, 3)
    assert np.allclose(R, R.T)
    assert np.all(np.linalg.eigvalsh(R) > 0.0)


def test_build_linearized_3dof_matrices_entries():
    env = Docking3DoFEnv(max_time=12_000.0)
    A, B = build_linearized_3dof_matrices(env)

    assert A.shape == (6, 6)
    assert B.shape == (6, 3)
    assert A[0, 2] == pytest.approx(1.0)
    assert A[1, 3] == pytest.approx(1.0)
    assert A[3, 1] == pytest.approx(-env.n**2)
    assert B[2, 0] == pytest.approx(1.0)
    assert B[3, 1] == pytest.approx(1.0)
    assert B[5, 2] == pytest.approx(1.0)


def test_closing_error_wraps_psi():
    env = Docking3DoFEnv(max_time=12_000.0)
    state = np.array([0.0, 0.0, 0.0, 0.0, np.pi - 0.01, 0.0])
    reference = np.array([0.0, 0.0, 0.0, 0.0, -np.pi + 0.01, 0.0])

    error = closing_error(env, state, reference)

    assert error.shape == (6,)
    assert error[4] == pytest.approx(-0.02)


def test_orbital_to_body_action_zero_angle():
    env = Docking3DoFEnv(max_time=12_000.0)
    action_orbital = np.array([0.1, -0.2, 0.3])

    action_body = orbital_to_body_action(env, action_orbital, psi=0.0)

    assert np.allclose(action_body, np.array([-0.1, 0.2, 0.3]))


def test_orbital_to_body_action_pi_over_2():
    env = Docking3DoFEnv(max_time=12_000.0)
    action_orbital = np.array([0.1, 0.2, -0.3])

    action_body = orbital_to_body_action(env, action_orbital, psi=np.pi / 2)

    assert np.allclose(action_body, np.array([-0.2, 0.1, -0.3]))


def test_compute_lqr_gain_rejects_non_pd_R():
    env = Docking3DoFEnv(max_time=12_000.0)
    A, B = build_linearized_3dof_matrices(env)

    with pytest.raises(ValueError, match="R must be positive definite"):
        compute_lqr_gain(A, B, build_Q_closing(), np.zeros((3, 3)))


def test_compute_lqr_gain_rejects_non_psd_Q():
    env = Docking3DoFEnv(max_time=12_000.0)
    A, B = build_linearized_3dof_matrices(env)
    Q_bad = build_Q_closing()
    Q_bad[0, 0] = -1.0

    with pytest.raises(ValueError, match="Q must be positive semidefinite"):
        compute_lqr_gain(A, B, Q_bad, build_R_closing(env))


def test_controller_rejects_wrong_phase():
    controller = LQRClosingController(np.zeros((3, 6)))
    env = Docking3DoFEnv(max_time=5_000.0, phase="final")

    with pytest.raises(ValueError, match="closing phase"):
        controller.compute_action(env, np.zeros(6))


def test_controller_requires_p3_target():
    controller = LQRClosingController(np.zeros((3, 6)))
    env = Docking3DoFEnv(max_time=12_000.0, phase="closing")

    with pytest.raises(RuntimeError, match="p3_target is None"):
        controller.compute_action(env, np.zeros(6))


def test_controller_rejects_invalid_state_shape():
    controller = LQRClosingController(np.zeros((3, 6)))
    env = Docking3DoFEnv(max_time=12_000.0, phase="closing")
    env.reset(mode="testing")

    with pytest.raises(ValueError, match="state must have shape"):
        controller.compute_action(env, np.zeros(5))


def test_analyze_saturation_empty_history(capsys):
    env = Docking3DoFEnv(max_time=12_000.0)

    stats = analyze_saturation(env)
    capsys.readouterr()

    assert stats["saturation_fraction_total"] == 0.0
    assert np.allclose(stats["saturation_fraction_channels"], np.zeros(3))


def test_analyze_saturation_detects_clipping(capsys):
    env = Docking3DoFEnv(max_time=12_000.0)
    env.reset(mode="testing")
    env.step(np.array([10.0, -10.0, 10.0]))

    stats = analyze_saturation(env)
    capsys.readouterr()

    assert stats["saturation_fraction_total"] > 0.0
    assert np.all(stats["max_raw"] >= stats["max_clipped"])
