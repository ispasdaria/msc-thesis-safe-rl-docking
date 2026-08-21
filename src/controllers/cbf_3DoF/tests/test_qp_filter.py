from dataclasses import replace

import numpy as np

from controllers.cbf_3DoF.barriers import FinalApproachCBFConfig
from controllers.cbf_3DoF.closing_barriers import ClosingCBFConfig
from controllers.cbf_3DoF.qp_filter import ClosingCBFFilter, FinalApproachCBFFilter
from envs.docking3dof.gym_env import Docking3DoFGymEnv


def make_env():
    env = Docking3DoFGymEnv(
        phase="final",
        mode="testing",
        dt=1.0,
        integrator="rk4",
        normalize_action=True,
        seed=0,
    )
    env.reset(seed=0)
    return env


def scipy_config(**changes):
    base = FinalApproachCBFConfig(solver_backend="scipy")
    return replace(base, **changes)


def make_closing_env():
    env = Docking3DoFGymEnv(
        phase="closing",
        mode="testing",
        dt=1.0,
        integrator="rk4",
        normalize_action=True,
        seed=0,
    )
    env.reset(seed=0)
    return env


def closing_scipy_config(**changes):
    base = ClosingCBFConfig(solver_backend="scipy")
    return replace(base, **changes)


def test_public_state_and_action_conversions_round_trip():
    env = make_env()
    state = env.get_physical_state()
    normalized = np.array([0.4, -0.2, 0.7])
    physical = env.normalized_action_to_physical(normalized)

    assert state.shape == (6,)
    assert state.dtype == np.float64
    assert np.allclose(env.physical_action_to_normalized(physical), normalized)
    assert np.allclose(env.get_physical_action_limits(), [0.01, 0.01, 0.003])


def test_feasible_nominal_action_passes_through():
    env = make_env()
    state = np.array([300.0, 0.0, -0.03, 0.0, 0.0, 0.0])
    nominal = np.zeros(3)
    cbf = FinalApproachCBFFilter(env.core_env, scipy_config())

    result = cbf.filter(state, nominal)

    assert not result.qp_failed
    assert not result.intervened
    assert np.allclose(result.action_filtered, nominal, atol=1.0e-8)
    assert min(result.relaxed_cbf_residuals.values()) >= -1.0e-7


def test_speed_increasing_action_is_filtered_near_speed_boundary():
    env = make_env()
    state = np.array([300.0, 0.0, -0.149, 0.0, 0.0, 0.0])
    nominal = np.array([0.01, 0.0, 0.0])
    cbf = FinalApproachCBFFilter(
        env.core_env,
        scipy_config(
            enable_cone=False,
            enable_speed=True,
            enable_alignment=False,
        ),
    )

    result = cbf.filter(state, nominal)

    assert not result.qp_failed
    assert result.intervened
    assert result.action_filtered[0] < nominal[0]
    assert result.relaxed_cbf_residuals["port_speed"] >= -1.0e-7


def test_outward_action_is_filtered_near_upper_cone_boundary():
    env = make_env()
    # psi=0 gives port position [rx - 24.1, ry].
    state = np.array([124.1, 9.9, -0.03, 0.04, 0.0, 0.0])
    nominal = np.array([0.0, -0.01, 0.0])
    cbf = FinalApproachCBFFilter(
        env.core_env,
        scipy_config(
            enable_cone=True,
            enable_speed=False,
            enable_alignment=False,
        ),
    )

    result = cbf.filter(state, nominal)

    assert not result.qp_failed
    assert result.intervened
    assert result.relaxed_cbf_residuals["cone_upper"] >= -1.0e-7
    assert np.all(np.abs(result.action_filtered) <= env.get_physical_action_limits() + 1.0e-12)


def test_soft_qp_reports_slack_for_unrecoverable_constraint_state():
    env = make_env()
    # Far outside the upper cone with a large outward lateral velocity.
    state = np.array([124.1, 30.0, -0.03, 0.2, 0.0, 0.0])
    cbf = FinalApproachCBFFilter(
        env.core_env,
        scipy_config(
            enable_cone=True,
            enable_speed=False,
            enable_alignment=False,
            cone_slack_penalty=1.0e6,
        ),
    )

    result = cbf.filter(state, np.zeros(3))

    assert not result.qp_failed
    assert result.slack_values["cone_upper"] > 0.0
    assert result.relaxed_cbf_residuals["cone_upper"] >= -1.0e-6


def test_outward_attitude_action_is_filtered_near_boundary():
    env = make_env()
    state = np.array(
        [300.0, 0.0, -0.03, 0.0, np.deg2rad(2.9), np.deg2rad(0.1)]
    )
    nominal = np.array([0.0, 0.0, env.core_env.alphaz_max])
    config = scipy_config(
        enable_cone=False,
        enable_speed=False,
        enable_alignment=True,
        allow_slack=False,
    )
    result = FinalApproachCBFFilter(env.core_env, config).filter(state, nominal)

    assert not result.qp_failed
    assert result.intervened
    assert result.action_filtered[2] < nominal[2]
    assert result.cbf_residuals["attitude_upper"] >= -1.0e-7


def test_outward_angular_acceleration_is_filtered_at_rate_boundary():
    env = make_env()
    state = np.array(
        [300.0, 0.0, -0.03, 0.0, 0.0, env.core_env.omega_safe]
    )
    nominal = np.array([0.0, 0.0, env.core_env.alphaz_max])
    config = scipy_config(
        enable_cone=False,
        enable_speed=False,
        enable_alignment=True,
        allow_slack=False,
    )
    result = FinalApproachCBFFilter(env.core_env, config).filter(state, nominal)
    next_state = env.core_env.integrate(state, result.action_filtered)

    assert not result.qp_failed
    assert result.intervened
    assert result.action_filtered[2] <= 1.0e-7
    assert abs(next_state[5]) <= config.angular_rate_max + 1.0e-7


def test_osqp_backend_solves_the_seven_row_filter():
    env = make_env()
    state = np.array([180.0, 8.0, -0.145, 0.03, np.deg2rad(1.0), 0.0])
    config = FinalApproachCBFConfig(solver_backend="osqp")
    cbf = FinalApproachCBFFilter(env.core_env, config)

    result = cbf.filter(state, np.array([0.01, -0.01, 0.003]))

    assert not result.qp_failed
    assert result.solver_backend == "osqp"
    assert result.solver_status.lower() in {"solved", "solved inaccurate"}
    assert min(result.relaxed_cbf_residuals.values()) >= -1.0e-6


def test_closing_filter_builds_only_the_two_requested_speed_constraints():
    env = make_closing_env()
    cbf = ClosingCBFFilter(
        env.core_env,
        closing_scipy_config(enable_cone=False, enable_alignment=False),
    )
    state = np.array([1700.0, 0.0, -0.10, 0.02, 0.0, 0.0])

    result = cbf.filter(state, np.zeros(3))

    assert not result.qp_failed
    assert set(result.barrier_values) == {"port_speed", "axial_port_speed"}
    assert set(result.cbf_residuals) == {"port_speed", "axial_port_speed"}


def test_closing_total_port_speed_cbf_filters_lateral_acceleration():
    env = make_closing_env()
    state = np.array([1700.0, 0.0, -0.05, 0.244, 0.0, 0.0])
    nominal = np.array([0.0, -0.01, 0.0])
    config = closing_scipy_config(
        enable_cone=False,
        enable_port_speed=True,
        enable_axial_port_speed=False,
        enable_alignment=False,
        allow_slack=False,
    )

    result = ClosingCBFFilter(env.core_env, config).filter(state, nominal)

    assert not result.qp_failed
    assert result.intervened
    assert result.action_filtered[1] > nominal[1]
    assert result.cbf_residuals["port_speed"] >= -1.0e-7


def test_closing_axial_port_speed_cbf_filters_excessive_closing_acceleration():
    env = make_closing_env()
    state = np.array([1700.0, 0.0, -0.149, 0.0, 0.0, 0.0])
    nominal = np.array([0.01, 0.0, 0.0])
    config = closing_scipy_config(
        enable_cone=False,
        enable_port_speed=False,
        enable_axial_port_speed=True,
        enable_alignment=False,
        allow_slack=False,
    )

    result = ClosingCBFFilter(env.core_env, config).filter(state, nominal)

    assert not result.qp_failed
    assert result.intervened
    assert result.action_filtered[0] < nominal[0]
    assert result.cbf_residuals["axial_port_speed"] >= -1.0e-7


def test_closing_cone_cbf_filters_outward_lateral_acceleration():
    env = make_closing_env()
    # At psi=0, the port is at [rx - 24.1, ry].  This puts it just inside
    # the upper lateral boundary y_p = cone_slope * s.
    state = np.array([1024.1, 99.0, -0.03, 0.04, 0.0, 0.0])
    nominal = np.array([0.0, -0.01, 0.0])
    config = closing_scipy_config(
        enable_cone=True,
        enable_port_speed=False,
        enable_axial_port_speed=False,
        enable_alignment=False,
        allow_slack=False,
    )

    result = ClosingCBFFilter(env.core_env, config).filter(state, nominal)

    assert not result.qp_failed
    assert result.intervened
    assert result.action_filtered[1] > nominal[1]
    assert result.cbf_residuals["cone_upper"] >= -1.0e-7


def test_closing_cone_cbf_filters_acceleration_past_axial_cap():
    env = make_closing_env()
    # At psi=0 this puts the port just inside s = cone_length.
    state = np.array([2023.1, 0.0, 0.03, 0.0, 0.0, 0.0])
    nominal = np.array([-0.01, 0.0, 0.0])
    config = closing_scipy_config(
        enable_cone=True,
        enable_port_speed=False,
        enable_axial_port_speed=False,
        enable_alignment=False,
        allow_slack=False,
    )

    result = ClosingCBFFilter(env.core_env, config).filter(state, nominal)

    assert not result.qp_failed
    assert result.intervened
    assert result.action_filtered[0] > nominal[0]
    assert result.cbf_residuals["cone_axial_upper"] >= -1.0e-7


def test_closing_alignment_cbf_filters_outward_angular_acceleration():
    env = make_closing_env()
    state = np.array(
        [1700.0, 0.0, -0.10, 0.0, np.deg2rad(2.9), np.deg2rad(0.1)]
    )
    nominal = np.array([0.0, 0.0, env.core_env.alphaz_max])
    config = closing_scipy_config(
        enable_cone=False,
        enable_port_speed=False,
        enable_axial_port_speed=False,
        enable_alignment=True,
        allow_slack=False,
    )

    result = ClosingCBFFilter(env.core_env, config).filter(state, nominal)

    assert not result.qp_failed
    assert result.intervened
    assert result.action_filtered[2] < nominal[2]
    assert result.cbf_residuals["attitude_upper"] >= -1.0e-7
    assert result.cbf_residuals["angular_rate_upper"] >= -1.0e-7


def test_closing_filter_rejects_non_closing_environment():
    env = make_env()

    with np.testing.assert_raises_regex(ValueError, "closing-phase"):
        ClosingCBFFilter(env.core_env)
