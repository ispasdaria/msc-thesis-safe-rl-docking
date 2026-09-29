"""Hand-selected action scenarios for the final-approach CBF-QP filter.

These tests exercise the safety filter without a neural controller.  Each test
corresponds to one of the manual validation cases used before controller/CBF
integration.
"""

from dataclasses import replace

import numpy as np

from cbf_3dof.barriers import (
    FinalApproachCBFConfig,
    build_final_approach_constraints,
)
from cbf_3dof.diagnostics import (
    cone_hocbf_admissibility,
    hard_cbf_feasibility_margin,
)
from cbf_3dof.qp_filter import FinalApproachCBFFilter
from envs.docking3dof.core import Docking3DoFEnv


J = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
RESIDUAL_TOL = 1.0e-7
BARRIER_TOL = 1.0e-6


def make_env():
    return Docking3DoFEnv(max_time=5_000.0, phase="final", dt=1.0, integrator="rk4", seed=123)


def hard_config(**changes):
    base = FinalApproachCBFConfig(
        allow_slack=False,
        solver_backend="osqp",
    )
    return replace(base, **changes)


def state_from_port_quantities(env, s, y_p, v_p, psi=0.0, omega=0.0):
    """Invert the simulator's offset-port position and velocity maps."""
    rotation = env.rotation_body_to_orbital_2d(psi)
    port_offset = rotation @ env.r_c_body
    com_position = np.array([s, y_p]) - port_offset + env.r_t_orbital
    com_velocity = np.asarray(v_p) - omega * (J @ port_offset)
    return np.concatenate((com_position, com_velocity, [psi, omega]))


def test_01_safe_interior_action_passes_through_unchanged():
    """A safe action pointing away from the upper cone side is non-invasive."""
    env = make_env()
    config = hard_config()
    state = state_from_port_quantities(
        env,
        s=100.0,
        y_p=5.0,
        v_p=np.array([-0.03, 0.0]),
    )
    # At psi=0, positive body-y acceleration points toward negative orbital y
    # and therefore toward the cone centreline from this upper-half state.
    nominal = np.array([0.0, 0.25 * env.ay_max, 0.0])

    result = FinalApproachCBFFilter(env, config).filter(state, nominal)

    assert not result.qp_failed
    assert not result.intervened
    assert np.allclose(result.action_filtered, nominal, atol=1.0e-8)
    assert min(result.cbf_residuals.values()) >= -RESIDUAL_TOL


def test_02_safe_near_boundary_outward_action_is_corrected():
    """A safe near-boundary state rejects an outward manual command."""
    env = make_env()
    config = hard_config(
        enable_speed=False,
        enable_alignment=False,
    )
    state = state_from_port_quantities(
        env,
        s=20.0,
        y_p=1.99,
        v_p=np.array([-0.03, -0.003]),
    )
    nominal = np.array([env.ax_max, -env.ay_max, -env.alphaz_max])
    constraints = build_final_approach_constraints(env, state, config)

    assert 0.0 < constraints.barrier_values["cone_upper"] < 0.02
    assert (constraints.A @ nominal - constraints.b)[0] < 0.0

    result = FinalApproachCBFFilter(env, config).filter(state, nominal)

    assert not result.qp_failed
    assert result.intervened
    assert result.cbf_residuals["cone_upper"] >= -RESIDUAL_TOL


def test_03_exact_boundary_outward_action_preserves_cone_for_one_step():
    """The QP handles an exact boundary value without sign or tolerance errors."""
    env = make_env()
    config = hard_config(
        enable_speed=False,
        enable_alignment=False,
    )
    state = state_from_port_quantities(
        env,
        s=20.0,
        y_p=config.kappa * 20.0,
        v_p=np.array([-0.03, -config.kappa * 0.03]),
    )
    nominal = np.array([env.ax_max, -env.ay_max, -env.alphaz_max])
    constraints = build_final_approach_constraints(env, state, config)

    assert abs(constraints.barrier_values["cone_upper"]) <= 1.0e-12

    result = FinalApproachCBFFilter(env, config).filter(state, nominal)
    next_state = env.integrate(state, result.action_filtered)
    next_constraints = build_final_approach_constraints(env, next_state, config)

    assert not result.qp_failed
    assert result.intervened
    assert result.cbf_residuals["cone_upper"] >= -RESIDUAL_TOL
    assert next_constraints.barrier_values["cone_upper"] >= -BARRIER_TOL


def test_04_safe_position_with_excessive_outward_velocity_is_corrected():
    """Position safety alone is insufficient when HOCBF velocity is inadmissible."""
    env = make_env()
    config = hard_config(
        enable_speed=False,
        enable_alignment=False,
    )
    state = state_from_port_quantities(
        env,
        s=20.0,
        y_p=1.8,
        v_p=np.array([-0.03, 0.03]),
    )
    nominal = np.array([env.ax_max, -env.ay_max, -env.alphaz_max])
    constraints = build_final_approach_constraints(env, state, config)
    hocbf_state = cone_hocbf_admissibility(constraints, config)

    assert constraints.barrier_values["cone_upper"] > 0.0
    assert hocbf_state["cone_upper"] < 0.0

    result = FinalApproachCBFFilter(env, config).filter(state, nominal)

    assert not result.qp_failed
    assert result.intervened
    assert result.cbf_residuals["cone_upper"] >= -RESIDUAL_TOL


def test_05_unsafe_initial_state_uses_reported_soft_constraint_slack():
    """An unsafe state is handled visibly, without claiming hard invariance."""
    env = make_env()
    config = replace(
        hard_config(enable_speed=False, enable_alignment=False),
        allow_slack=True,
        cone_slack_penalty=1.0e6,
    )
    state = state_from_port_quantities(
        env,
        s=100.0,
        y_p=30.0,
        v_p=np.array([-0.03, 0.20]),
    )
    constraints = build_final_approach_constraints(env, state, config)

    assert constraints.barrier_values["cone_upper"] < 0.0

    result = FinalApproachCBFFilter(env, config).filter(state, np.zeros(3))

    assert not result.qp_failed
    assert result.slack_values["cone_upper"] > 0.0
    assert result.cbf_residuals["cone_upper"] < 0.0
    assert result.relaxed_cbf_residuals["cone_upper"] >= -BARRIER_TOL
    assert np.all(np.abs(result.action_filtered) <= env.action_high + 1.0e-12)


def test_06_simultaneous_cone_and_speed_boundaries_are_enforced():
    """One QP command satisfies two simultaneously active barrier families."""
    env = make_env()
    config = hard_config(enable_alignment=False)
    tangent_velocity = np.array([-1.0, -config.kappa])
    tangent_velocity *= config.port_speed_max / np.linalg.norm(tangent_velocity)
    state = state_from_port_quantities(
        env,
        s=20.0,
        y_p=config.kappa * 20.0,
        v_p=tangent_velocity,
    )
    nominal = np.array([env.ax_max, -env.ay_max, -env.alphaz_max])
    constraints = build_final_approach_constraints(env, state, config)

    assert abs(constraints.barrier_values["cone_upper"]) <= 1.0e-12
    assert abs(constraints.barrier_values["port_speed"]) <= 1.0e-12

    result = FinalApproachCBFFilter(env, config).filter(state, nominal)

    assert not result.qp_failed
    assert result.intervened
    assert result.cbf_residuals["cone_upper"] >= -RESIDUAL_TOL
    assert result.cbf_residuals["port_speed"] >= -RESIDUAL_TOL


def test_07_barrier_actuator_conflict_is_reported_as_hard_infeasibility():
    """The filter exposes a CBF demand that exceeds physical actuator limits."""
    env = make_env()
    config = hard_config(
        enable_speed=False,
        enable_alignment=False,
    )
    state = state_from_port_quantities(
        env,
        s=0.2,
        y_p=config.kappa * 0.2,
        v_p=np.array([-0.10, -config.kappa * 0.10]),
    )
    nominal = np.zeros(3)
    constraints = build_final_approach_constraints(env, state, config)
    physical_feasibility = hard_cbf_feasibility_margin(
        constraints,
        env.action_high,
    )
    doubled_actuator_feasibility = hard_cbf_feasibility_margin(
        constraints,
        2.0 * env.action_high,
    )

    assert not physical_feasibility.feasible
    assert physical_feasibility.margin < 0.0
    assert doubled_actuator_feasibility.feasible

    result = FinalApproachCBFFilter(env, config).filter(state, nominal)

    assert result.qp_failed
    # The documented failed-QP policy returns the clipped nominal action; it
    # must not be mistaken for a CBF-certified command.
    assert np.array_equal(result.action_filtered, nominal)
    assert min(result.cbf_residuals.values()) < 0.0


def test_08_attitude_transform_rotates_the_required_body_frame_correction():
    """Equivalent outward orbital commands are corrected at two attitudes."""
    env = make_env()
    config = hard_config(
        enable_speed=False,
        enable_alignment=False,
    )
    filtered_body_actions = []

    for psi in (0.0, np.pi / 2.0):
        state = state_from_port_quantities(
            env,
            s=20.0,
            y_p=1.99,
            v_p=np.array([-0.03, -0.003]),
            psi=psi,
        )
        rotation = env.rotation_body_to_orbital_2d(psi)
        outward_orbital_acceleration = np.array([0.0, env.ay_max])
        nominal = np.concatenate(
            (rotation.T @ outward_orbital_acceleration, [0.0])
        )
        constraints = build_final_approach_constraints(env, state, config)

        assert np.allclose(
            rotation @ nominal[:2],
            outward_orbital_acceleration,
            atol=1.0e-12,
        )

        result = FinalApproachCBFFilter(env, config).filter(state, nominal)
        port_acceleration_correction = (
            constraints.port_acceleration.control_matrix
            @ (result.action_filtered - nominal)
        )

        assert not result.qp_failed
        assert result.intervened
        assert result.cbf_residuals["cone_upper"] >= -RESIDUAL_TOL
        assert port_acceleration_correction[1] < 0.0
        filtered_body_actions.append(result.action_filtered)

    # The same orbital-frame hazard maps to different body-frame commands.
    assert not np.allclose(filtered_body_actions[0], filtered_body_actions[1])
