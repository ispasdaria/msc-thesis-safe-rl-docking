import numpy as np

from controllers.cbf_3DoF.barriers import (
    FinalApproachCBFConfig,
    alignment_barrier_values,
    build_final_approach_constraints,
    cone_barrier_derivatives,
    cone_barrier_values,
    speed_barrier_value,
)
from controllers.cbf_3DoF.closing_barriers import (
    ClosingCBFConfig,
    axial_speed_barrier_value,
    build_closing_constraints,
)
from envs.docking3dof.core import Docking3DoFEnv


def make_env():
    return Docking3DoFEnv(phase="final", dt=1.0, integrator="rk4", seed=0)


def make_closing_env():
    return Docking3DoFEnv(phase="closing", dt=1.0, integrator="rk4", seed=0)


def test_cone_barrier_signs():
    assert np.allclose(cone_barrier_values(100.0, 5.0, 0.1), [5.0, 15.0])
    assert np.allclose(cone_barrier_values(100.0, 10.0, 0.1), [0.0, 20.0])
    assert cone_barrier_values(100.0, 11.0, 0.1)[0] < 0.0
    assert cone_barrier_values(100.0, -11.0, 0.1)[1] < 0.0


def test_cone_barrier_derivative_signs():
    upper_dot, lower_dot = cone_barrier_derivatives(-0.05, 0.02, 0.1)
    assert np.isclose(upper_dot, -0.025)
    assert np.isclose(lower_dot, 0.015)


def test_speed_barrier_values():
    assert speed_barrier_value(np.array([0.10, 0.0]), 0.15) > 0.0
    assert np.isclose(speed_barrier_value(np.array([0.15, 0.0]), 0.15), 0.0)
    assert speed_barrier_value(np.array([0.20, 0.0]), 0.15) < 0.0


def test_axial_speed_barrier_is_symmetric_and_has_expected_sign():
    assert axial_speed_barrier_value(0.10, 0.15) > 0.0
    assert np.isclose(axial_speed_barrier_value(-0.15, 0.15), 0.0)
    assert axial_speed_barrier_value(0.20, 0.15) < 0.0
    assert np.isclose(
        axial_speed_barrier_value(-0.10, 0.15),
        axial_speed_barrier_value(0.10, 0.15),
    )


def test_closing_constraints_match_direct_cbf_expressions():
    env = make_closing_env()
    state = np.array(
        [1700.0, 8.0, -0.14, 0.08, np.deg2rad(2.0), np.deg2rad(0.05)]
    )
    action = np.array([0.003, -0.004, 0.001])
    config = ClosingCBFConfig(
        port_speed_alpha=0.4,
        axial_port_speed_alpha=0.6,
    )
    constraints = build_closing_constraints(env, state, config)
    acceleration = (
        constraints.port_acceleration.drift
        + constraints.port_acceleration.control_matrix @ action
    )
    port = constraints.kinematics

    direct = {
        "port_speed": -2.0 * port.v_p @ acceleration
        + config.port_speed_alpha * constraints.barrier_values["port_speed"],
        "axial_port_speed": -2.0 * port.s_dot * acceleration[0]
        + config.axial_port_speed_alpha
        * constraints.barrier_values["axial_port_speed"],
    }

    assert constraints.names == ("port_speed", "axial_port_speed")
    for index, name in enumerate(constraints.names):
        assert np.isclose(
            constraints.A[index] @ action - constraints.b[index],
            direct[name],
        )


def test_closing_axial_barrier_derivative_matches_finite_difference():
    env = make_closing_env()
    state = np.array(
        [1700.0, -4.0, -0.13, 0.03, np.deg2rad(-1.0), np.deg2rad(0.04)]
    )
    action = np.array([-0.002, 0.006, -0.001])
    config = ClosingCBFConfig()
    constraints = build_closing_constraints(env, state, config)
    acceleration = (
        constraints.port_acceleration.drift
        + constraints.port_acceleration.control_matrix @ action
    )
    analytic = -2.0 * constraints.kinematics.s_dot * acceleration[0]

    dt = 1.0e-5
    state_next = state + dt * env.dynamics(state, action)
    s_dot_now = env.get_port_relative_velocity(state)[0]
    s_dot_next = env.get_port_relative_velocity(state_next)[0]
    finite_difference = (
        axial_speed_barrier_value(s_dot_next, config.axial_port_speed_max)
        - axial_speed_barrier_value(s_dot_now, config.axial_port_speed_max)
    ) / dt

    assert np.isclose(analytic, finite_difference, rtol=3.0e-4, atol=2.0e-7)


def test_alignment_barrier_signs():
    limits = alignment_barrier_values(
        np.deg2rad(2.0),
        np.deg2rad(-0.1),
        np.deg2rad(3.0),
        np.deg2rad(0.15),
    )

    assert limits["attitude_upper"] > 0.0
    assert limits["attitude_lower"] > 0.0
    assert limits["angular_rate_upper"] > 0.0
    assert limits["angular_rate_lower"] > 0.0
    outside = alignment_barrier_values(
        np.deg2rad(3.1),
        np.deg2rad(0.16),
        np.deg2rad(3.0),
        np.deg2rad(0.15),
    )
    assert outside["attitude_upper"] < 0.0
    assert outside["angular_rate_upper"] < 0.0


def test_constraint_rows_match_direct_cbf_expressions():
    env = make_env()
    state = np.array(
        [120.0, 4.0, -0.08, 0.03, np.deg2rad(1.0), np.deg2rad(0.04)]
    )
    action = np.array([0.003, -0.004, 0.001])
    config = FinalApproachCBFConfig(cone_k1=0.2, cone_k2=0.3, speed_alpha=0.4)
    constraints = build_final_approach_constraints(env, state, config)
    acceleration = (
        constraints.port_acceleration.drift
        + constraints.port_acceleration.control_matrix @ action
    )
    port = constraints.kinematics

    h_plus = constraints.barrier_values["cone_upper"]
    h_minus = constraints.barrier_values["cone_lower"]
    h_plus_dot = constraints.barrier_derivatives["cone_upper"]
    h_minus_dot = constraints.barrier_derivatives["cone_lower"]
    gain_sum = config.cone_k1 + config.cone_k2
    gain_product = config.cone_k1 * config.cone_k2
    attitude_gain_sum = config.attitude_k1 + config.attitude_k2
    attitude_gain_product = config.attitude_k1 * config.attitude_k2

    direct = {
        "cone_upper": config.kappa * acceleration[0]
        - acceleration[1]
        + gain_sum * h_plus_dot
        + gain_product * h_plus,
        "cone_lower": config.kappa * acceleration[0]
        + acceleration[1]
        + gain_sum * h_minus_dot
        + gain_product * h_minus,
        "port_speed": -2.0 * port.v_p @ acceleration
        + config.speed_alpha * constraints.barrier_values["port_speed"],
        "attitude_upper": -action[2]
        - attitude_gain_sum * state[5]
        + attitude_gain_product * constraints.barrier_values["attitude_upper"],
        "attitude_lower": action[2]
        + attitude_gain_sum * state[5]
        + attitude_gain_product * constraints.barrier_values["attitude_lower"],
        "angular_rate_upper": -action[2]
        + config.angular_rate_alpha
        * constraints.barrier_values["angular_rate_upper"],
        "angular_rate_lower": action[2]
        + config.angular_rate_alpha
        * constraints.barrier_values["angular_rate_lower"],
    }

    for index, name in enumerate(constraints.names):
        assert np.isclose(constraints.A[index] @ action - constraints.b[index], direct[name])


def test_speed_barrier_derivative_matches_finite_difference():
    env = make_env()
    state = np.array(
        [260.0, 1.0, -0.08, 0.04, np.deg2rad(-1.2), np.deg2rad(0.05)]
    )
    action = np.array([-0.002, 0.006, -0.001])
    config = FinalApproachCBFConfig()
    constraints = build_final_approach_constraints(env, state, config)
    acceleration = (
        constraints.port_acceleration.drift
        + constraints.port_acceleration.control_matrix @ action
    )
    analytic = -2.0 * constraints.kinematics.v_p @ acceleration

    dt = 1.0e-5
    state_next = state + dt * env.dynamics(state, action)
    h_now = speed_barrier_value(env.get_port_relative_velocity(state), 0.15)
    h_next = speed_barrier_value(env.get_port_relative_velocity(state_next), 0.15)
    finite_difference = (h_next - h_now) / dt

    assert np.isclose(analytic, finite_difference, rtol=3.0e-4, atol=2.0e-7)


def test_cone_second_derivatives_match_finite_difference():
    env = make_env()
    state = np.array(
        [180.0, 7.0, -0.09, -0.02, np.deg2rad(2.0), np.deg2rad(-0.08)]
    )
    action = np.array([0.004, -0.003, 0.0015])
    config = FinalApproachCBFConfig()
    constraints = build_final_approach_constraints(env, state, config)
    acceleration = (
        constraints.port_acceleration.drift
        + constraints.port_acceleration.control_matrix @ action
    )
    analytic = np.array(
        [
            config.kappa * acceleration[0] - acceleration[1],
            config.kappa * acceleration[0] + acceleration[1],
        ]
    )

    dt = 1.0e-5
    state_next = state + dt * env.dynamics(state, action)
    next_constraints = build_final_approach_constraints(env, state_next, config)
    h_dot_now = np.array(
        [
            constraints.barrier_derivatives["cone_upper"],
            constraints.barrier_derivatives["cone_lower"],
        ]
    )
    h_dot_next = np.array(
        [
            next_constraints.barrier_derivatives["cone_upper"],
            next_constraints.barrier_derivatives["cone_lower"],
        ]
    )

    assert np.allclose(analytic, (h_dot_next - h_dot_now) / dt, rtol=3e-4, atol=2e-7)


def test_alignment_barrier_derivatives_match_finite_difference():
    env = make_env()
    state = np.array(
        [180.0, 0.0, -0.03, 0.0, np.deg2rad(2.0), np.deg2rad(0.08)]
    )
    action = np.array([0.0, 0.0, -0.0015])
    config = FinalApproachCBFConfig()
    constraints = build_final_approach_constraints(env, state, config)

    dt = 1.0e-6
    state_next = state + dt * env.dynamics(state, action)
    next_constraints = build_final_approach_constraints(env, state_next, config)

    for name, expected_derivative in (
        ("attitude_upper", -state[5]),
        ("attitude_lower", state[5]),
        ("angular_rate_upper", -action[2]),
        ("angular_rate_lower", action[2]),
    ):
        finite_difference = (
            next_constraints.barrier_values[name]
            - constraints.barrier_values[name]
        ) / dt
        assert np.isclose(
            finite_difference,
            expected_derivative,
            rtol=1.0e-6,
            atol=1.0e-9,
        )
