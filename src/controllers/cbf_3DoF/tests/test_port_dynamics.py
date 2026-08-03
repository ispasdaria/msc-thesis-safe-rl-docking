import numpy as np

from controllers.cbf_3DoF.port_dynamics import (
    compute_port_acceleration_affine,
    compute_port_kinematics,
    evaluate_port_acceleration,
)
from envs.docking3dof.core import Docking3DoFEnv


def make_env():
    return Docking3DoFEnv(phase="final", dt=1.0, integrator="rk4", seed=0)


def test_zero_attitude_port_kinematics_matches_environment_geometry():
    env = make_env()
    state = np.array([300.0, 2.0, -0.04, 0.01, 0.0, 0.0])

    port = compute_port_kinematics(env, state)

    assert np.isclose(port.s, 300.0 - 4.1 - 20.0)
    assert np.isclose(port.y_p, 2.0)
    assert np.allclose(port.v_p, [-0.04, 0.01])
    assert np.isclose(port.s_dot, port.v_p[0])
    assert np.isclose(port.y_p_dot, port.v_p[1])


def test_angular_velocity_changes_port_velocity():
    env = make_env()
    omega = 0.02
    state = np.array([300.0, 0.0, 0.0, 0.0, 0.0, omega])

    port = compute_port_kinematics(env, state)

    assert np.isclose(port.s_dot, 0.0)
    assert np.isclose(port.y_p_dot, -4.1 * omega)


def test_affine_port_acceleration_matches_finite_difference():
    env = make_env()
    state = np.array(
        [285.0, -1.5, -0.06, 0.025, np.deg2rad(1.7), np.deg2rad(0.08)]
    )
    action = np.array([0.004, -0.006, 0.0015])
    affine = compute_port_acceleration_affine(env, state)
    analytic = evaluate_port_acceleration(affine, action)

    dt = 1.0e-5
    state_derivative = env.dynamics(state, action)
    state_next = state + dt * state_derivative
    velocity_now = env.get_port_relative_velocity(state)
    velocity_next = env.get_port_relative_velocity(state_next)
    finite_difference = (velocity_next - velocity_now) / dt

    assert np.allclose(analytic, finite_difference, rtol=2.0e-4, atol=2.0e-7)


def test_angular_control_column_is_present_for_offset_port():
    env = make_env()
    state = np.array([300.0, 0.0, -0.03, 0.0, np.deg2rad(20.0), 0.0])
    affine = compute_port_acceleration_affine(env, state)

    assert affine.control_matrix.shape == (2, 3)
    assert np.linalg.norm(affine.control_matrix[:, 2]) > 4.0

