import numpy as np
import pytest

from controllers.cbf_3DoF.barriers import (
    FinalApproachCBFConfig,
    build_final_approach_constraints,
)
from controllers.cbf_3DoF.diagnostics import (
    attitude_hocbf_admissibility,
    cone_hocbf_admissibility,
    hard_cbf_feasibility_margin,
)
from controllers.cbf_3DoF.qp_filter import FinalApproachCBFFilter
from envs.docking3dof.core import Docking3DoFEnv


J = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)


def make_env(dt=1.0):
    return Docking3DoFEnv(max_time=5_000.0, phase="final", dt=dt, integrator="rk4", seed=123)


def hard_config():
    return FinalApproachCBFConfig(allow_slack=False, solver_backend="osqp")


def state_from_port_quantities(env, s, y_p, v_p, psi=0.0, omega=0.0):
    """Invert the environment's offset-port position and velocity maps."""
    rotation = env.rotation_body_to_orbital_2d(psi)
    port_offset = rotation @ env.r_c_body
    com_position = np.array([s, y_p]) - port_offset + env.r_t_orbital
    com_velocity = np.asarray(v_p) - omega * (J @ port_offset)
    return np.concatenate((com_position, com_velocity, [psi, omega]))


@pytest.mark.parametrize("mode", ["testing", "training"])
def test_sampled_p3_initial_states_are_strictly_hocbf_admissible(mode):
    env = make_env()
    config = hard_config()

    for _ in range(500):
        state = env.sample_X_P3(mode)
        constraints = build_final_approach_constraints(env, state, config)
        psi_1 = cone_hocbf_admissibility(constraints, config)
        psi_1.update(attitude_hocbf_admissibility(constraints, config))

        assert constraints.barrier_values["cone_upper"] > 0.0
        assert constraints.barrier_values["cone_lower"] > 0.0
        assert psi_1["cone_upper"] > 0.0
        assert psi_1["cone_lower"] > 0.0


@pytest.mark.parametrize("s", [300.0, 100.0, 20.0, 5.0, 1.0])
@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_tangent_cone_boundary_has_a_hard_feasible_control(s, side):
    env = make_env()
    config = hard_config()
    state = state_from_port_quantities(
        env,
        s=s,
        y_p=side * config.kappa * s,
        v_p=np.array([-0.1, -side * config.kappa * 0.1]),
    )
    constraints = build_final_approach_constraints(env, state, config)
    psi_1 = cone_hocbf_admissibility(constraints, config)
    feasibility = hard_cbf_feasibility_margin(
        constraints,
        env.action_high,
    )

    active_name = "cone_upper" if side > 0.0 else "cone_lower"
    assert abs(constraints.barrier_values[active_name]) <= 1.0e-12
    assert psi_1[active_name] >= -1.0e-12
    assert feasibility.feasible


def test_simultaneous_cone_and_speed_boundaries_are_hard_feasible():
    env = make_env()
    config = hard_config()
    tangent_velocity = np.array([-1.0, -config.kappa])
    tangent_velocity *= config.port_speed_max / np.linalg.norm(tangent_velocity)
    state = state_from_port_quantities(
        env,
        s=20.0,
        y_p=config.kappa * 20.0,
        v_p=tangent_velocity,
    )
    constraints = build_final_approach_constraints(env, state, config)
    feasibility = hard_cbf_feasibility_margin(constraints, env.action_high)

    assert abs(constraints.barrier_values["cone_upper"]) <= 1.0e-12
    assert abs(constraints.barrier_values["port_speed"]) <= 1.0e-12
    assert feasibility.feasible
    assert feasibility.margin > 0.0


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_worst_outward_action_is_corrected_and_survives_one_rk4_step(side):
    env = make_env(dt=1.0)
    config = hard_config()
    state = state_from_port_quantities(
        env,
        s=20.0,
        y_p=side * config.kappa * 20.0,
        v_p=np.array([-0.1, -side * config.kappa * 0.1]),
    )
    # At psi=0 the body axes are anti-aligned with the orbital axes.  Both the
    # lateral translation and offset-port angular acceleration point outward.
    nominal = np.array(
        [env.ax_max, -side * env.ay_max, -side * env.alphaz_max]
    )
    result = FinalApproachCBFFilter(env, config).filter(state, nominal)
    next_state = env.integrate(state, result.action_filtered)
    next_constraints = build_final_approach_constraints(env, next_state, config)

    assert not result.qp_failed
    assert result.intervened
    assert min(result.cbf_residuals.values()) >= -1.0e-7
    assert next_constraints.barrier_values["cone_upper"] >= -1.0e-6
    assert next_constraints.barrier_values["cone_lower"] >= -1.0e-6
    assert next_constraints.barrier_values["port_speed"] >= -1.0e-6


def test_speed_boundary_increasing_action_is_corrected_for_one_rk4_step():
    env = make_env(dt=1.0)
    config = hard_config()
    state = state_from_port_quantities(
        env,
        s=100.0,
        y_p=0.0,
        v_p=np.array([-config.port_speed_max, 0.0]),
    )
    # Positive body-x acceleration is negative orbital-x acceleration at
    # psi=0, so it increases the magnitude of this closing port velocity.
    nominal = np.array([env.ax_max, 0.0, 0.0])
    result = FinalApproachCBFFilter(env, config).filter(state, nominal)
    next_state = env.integrate(state, result.action_filtered)
    next_constraints = build_final_approach_constraints(env, next_state, config)

    assert not result.qp_failed
    assert result.intervened
    assert result.cbf_residuals["port_speed"] >= -1.0e-7
    assert next_constraints.barrier_values["port_speed"] >= -1.0e-6


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_attitude_boundary_outward_action_is_corrected_for_one_rk4_step(side):
    env = make_env(dt=1.0)
    config = hard_config()
    state = state_from_port_quantities(
        env,
        s=100.0,
        y_p=0.0,
        v_p=np.array([-0.03, 0.0]),
        psi=side * config.attitude_max,
        omega=0.0,
    )
    nominal = np.array([0.0, 0.0, side * env.alphaz_max])
    result = FinalApproachCBFFilter(env, config).filter(state, nominal)
    next_state = env.integrate(state, result.action_filtered)
    next_constraints = build_final_approach_constraints(env, next_state, config)

    active = "attitude_upper" if side > 0.0 else "attitude_lower"
    assert not result.qp_failed
    assert result.intervened
    assert result.cbf_residuals[active] >= -1.0e-7
    assert next_constraints.barrier_values[active] >= -1.0e-6


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_angular_rate_boundary_outward_action_is_corrected_for_one_rk4_step(side):
    env = make_env(dt=1.0)
    config = hard_config()
    state = state_from_port_quantities(
        env,
        s=100.0,
        y_p=0.0,
        v_p=np.array([-0.03, 0.0]),
        psi=0.0,
        omega=side * config.angular_rate_max,
    )
    nominal = np.array([0.0, 0.0, side * env.alphaz_max])
    result = FinalApproachCBFFilter(env, config).filter(state, nominal)
    next_state = env.integrate(state, result.action_filtered)
    next_constraints = build_final_approach_constraints(env, next_state, config)

    active = "angular_rate_upper" if side > 0.0 else "angular_rate_lower"
    assert not result.qp_failed
    assert result.intervened
    assert result.cbf_residuals[active] >= -1.0e-7
    assert next_constraints.barrier_values[active] >= -1.0e-6


def test_fast_near_apex_state_is_outside_hocbf_set_and_infeasible():
    """Document a real limitation, rather than silently weakening the cone."""
    env = make_env()
    config = hard_config()
    state = state_from_port_quantities(
        env,
        s=0.2,
        y_p=config.kappa * 0.2,
        v_p=np.array([-0.1, -config.kappa * 0.1]),
    )
    constraints = build_final_approach_constraints(env, state, config)
    psi_1 = cone_hocbf_admissibility(constraints, config)
    feasibility = hard_cbf_feasibility_margin(constraints, env.action_high)
    result = FinalApproachCBFFilter(env, config).filter(state, np.zeros(3))

    assert constraints.barrier_values["cone_upper"] >= -1.0e-12
    assert psi_1["cone_lower"] < 0.0
    assert not feasibility.feasible
    assert feasibility.margin < 0.0
    assert result.qp_failed


def test_terminal_state_is_inside_both_barriers_without_an_extra_step():
    env = make_env()
    config = hard_config()
    state = state_from_port_quantities(
        env,
        s=0.05,
        y_p=0.0,
        v_p=np.array([-0.05, 0.0]),
    )
    constraints = build_final_approach_constraints(env, state, config)

    assert env.is_in_X_dock(state)
    assert constraints.barrier_values["cone_upper"] > 0.0
    assert constraints.barrier_values["cone_lower"] > 0.0
    assert constraints.barrier_values["port_speed"] > 0.0


def test_randomized_admissible_boundary_states_remain_feasible_for_one_step():
    env = make_env(dt=1.0)
    config = hard_config()
    rng = np.random.default_rng(7)

    for _ in range(200):
        s = rng.uniform(2.0, 300.0)
        y_fraction = rng.uniform(-1.0, 1.0)
        y_p = y_fraction * config.kappa * s
        s_dot = -rng.uniform(0.005, min(0.14, 0.075 * s))

        h_upper = config.kappa * s - y_p
        h_lower = config.kappa * s + y_p
        y_dot_lower = -config.kappa * s_dot - config.cone_k1 * h_lower
        y_dot_upper = config.kappa * s_dot + config.cone_k1 * h_upper
        # Stay away from the exact psi_1 boundaries while sampling both signs.
        y_p_dot = rng.uniform(
            0.8 * y_dot_lower + 0.2 * y_dot_upper,
            0.2 * y_dot_lower + 0.8 * y_dot_upper,
        )
        v_p = np.array([s_dot, y_p_dot])
        if np.linalg.norm(v_p) >= 0.145:
            v_p *= 0.145 / np.linalg.norm(v_p)

        state = state_from_port_quantities(
            env,
            s=s,
            y_p=y_p,
            v_p=v_p,
            psi=rng.uniform(-np.deg2rad(2.0), np.deg2rad(2.0)),
            omega=rng.uniform(-np.deg2rad(0.1), np.deg2rad(0.1)),
        )
        constraints = build_final_approach_constraints(env, state, config)
        psi_1 = cone_hocbf_admissibility(constraints, config)
        psi_1.update(attitude_hocbf_admissibility(constraints, config))
        feasibility = hard_cbf_feasibility_margin(constraints, env.action_high)
        nominal = rng.uniform(-env.action_high, env.action_high)
        result = FinalApproachCBFFilter(env, config).filter(state, nominal)

        assert min(constraints.barrier_values.values()) >= -1.0e-9
        assert min(psi_1.values()) >= -1.0e-9
        assert feasibility.feasible
        assert not result.qp_failed

        next_state = env.integrate(state, result.action_filtered)
        next_constraints = build_final_approach_constraints(env, next_state, config)
        assert min(next_constraints.barrier_values.values()) >= -1.0e-6
