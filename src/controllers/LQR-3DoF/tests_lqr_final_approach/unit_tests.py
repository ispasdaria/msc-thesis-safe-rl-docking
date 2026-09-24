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
    build_port_error_mapping,
    build_Q_final,
    build_R_final,
    final_error,
    orbital_to_body_action,
)
from envs.docking3dof.core import Docking3DoFEnv


def test_build_Q_final_is_symmetric_psd():
    Q = build_Q_final()

    assert Q.shape == (6, 6)
    assert np.allclose(Q, Q.T)

    eigvals = np.linalg.eigvalsh(Q)
    assert np.all(eigvals >= 0.0)


def test_build_R_final_is_symmetric_pd():
    env = Docking3DoFEnv(max_time=5_000.0, phase="final")
    R = build_R_final(env)

    assert R.shape == (3, 3)
    assert np.allclose(R, R.T)

    eigvals = np.linalg.eigvalsh(R)
    assert np.all(eigvals > 0.0)


def test_port_error_mapping_shape_and_conditioning():
    env = Docking3DoFEnv(max_time=5_000.0, phase="final")
    H = build_port_error_mapping(env)

    assert H.shape == (6, 6)
    assert np.all(np.isfinite(H))

    cond_H = np.linalg.cond(H)
    assert cond_H < 1e8


def test_port_error_mapping_contains_expected_entries():
    env = Docking3DoFEnv(max_time=5_000.0, phase="final")
    H = build_port_error_mapping(env)

    x_c, y_c = env.r_c_body

    assert H[0, 0] == pytest.approx(1.0)
    assert H[0, 4] == pytest.approx(y_c)

    assert H[1, 1] == pytest.approx(1.0)
    assert H[1, 4] == pytest.approx(-x_c)

    assert H[2, 2] == pytest.approx(1.0)
    assert H[2, 5] == pytest.approx(y_c)

    assert H[3, 3] == pytest.approx(1.0)
    assert H[3, 5] == pytest.approx(-x_c)

    assert H[4, 4] == pytest.approx(1.0)
    assert H[5, 5] == pytest.approx(1.0)


def test_final_error_shape_and_finiteness():
    env = Docking3DoFEnv(max_time=5_000.0, phase="final", seed=0)
    state = env.reset(mode="testing")

    e = final_error(env, state)

    assert e.shape == (6,)
    assert np.all(np.isfinite(e))


def test_final_error_rejects_invalid_state_shape():
    env = Docking3DoFEnv(max_time=5_000.0, phase="final")

    with pytest.raises(ValueError, match="state must have shape"):
        final_error(env, np.zeros(5))


def test_orbital_to_body_action_zero_angle():
    env = Docking3DoFEnv(max_time=5_000.0, phase="final")
    action_orbital = np.array([0.1, -0.2, 0.3])

    action_body = orbital_to_body_action(env, action_orbital, psi=0.0)

    expected = np.array([-0.1, 0.2, 0.3])
    assert np.allclose(action_body, expected)


def test_orbital_to_body_action_inverse_transform():
    env = Docking3DoFEnv(max_time=5_000.0, phase="final")
    psi = 0.3
    action_orbital = np.array([0.05, -0.02, 0.01])

    action_body = orbital_to_body_action(env, action_orbital, psi=psi)
    action_orbital_back = env.transform_action_to_orbital_frame(action_body, psi=psi)

    assert np.allclose(action_orbital_back, action_orbital)


def test_orbital_to_body_action_rejects_wrong_shape():
    env = Docking3DoFEnv(max_time=5_000.0, phase="final")

    with pytest.raises(ValueError, match="Orbital-frame action"):
        orbital_to_body_action(env, np.zeros(2), psi=0.0)