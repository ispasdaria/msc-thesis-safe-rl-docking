import numpy as np

from controllers.cbf_3DoF.barriers import FinalApproachCBFConfig
from controllers.cbf_3DoF.evaluate_final_approach_cbf import (
    classify_protected_barrier_violations,
    diagnose_qp_failure,
    summarize_variant,
)
from envs.docking3dof.core import Docking3DoFEnv


J = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)


def state_from_port_quantities(env, s, y_p, v_p, psi=0.0, omega=0.0):
    rotation = env.rotation_body_to_orbital_2d(psi)
    port_offset = rotation @ env.r_c_body
    com_position = np.array([s, y_p]) - port_offset + env.r_t_orbital
    com_velocity = np.asarray(v_p) - omega * (J @ port_offset)
    return np.concatenate((com_position, com_velocity, [psi, omega]))


def test_protected_barrier_reporting_separates_strict_and_tolerance_crossings():
    strict, beyond = classify_protected_barrier_violations(
        {
            "cone_upper": -0.5e-6,
            "cone_lower": -2.0e-6,
            "port_speed": 0.1,
        }
    )

    assert strict == ("cone_upper", "cone_lower")
    assert beyond == ("cone_lower",)


def test_qp_failure_diagnosis_identifies_minimal_infeasible_cone_family():
    env = Docking3DoFEnv(phase="final", dt=1.0, integrator="rk4", seed=123)
    config = FinalApproachCBFConfig(allow_slack=False, solver_backend="osqp")
    state = state_from_port_quantities(
        env,
        s=0.2,
        y_p=config.kappa * 0.2,
        v_p=np.array([-0.1, -config.kappa * 0.1]),
    )

    diagnosis = diagnose_qp_failure(env, state, config, "primal infeasible")

    assert diagnosis["reason"] == "hard_constraints_infeasible[cone]"
    assert diagnosis["minimal_infeasible_constraint_groups"] == "cone"
    assert diagnosis["hard_feasibility_margin"] < 0.0


def test_qp_failure_diagnosis_distinguishes_solver_failure_from_infeasibility():
    env = Docking3DoFEnv(phase="final", dt=1.0, integrator="rk4", seed=123)
    config = FinalApproachCBFConfig(allow_slack=False, solver_backend="osqp")
    state = state_from_port_quantities(
        env,
        s=100.0,
        y_p=0.0,
        v_p=np.array([-0.03, 0.0]),
    )

    diagnosis = diagnose_qp_failure(env, state, config, "maximum iterations reached")

    assert diagnosis["reason"] == (
        "solver_failed_despite_hard_feasibility[maximum iterations reached]"
    )
    assert diagnosis["minimal_infeasible_constraint_groups"] == ""
    assert diagnosis["hard_feasibility_margin"] >= 0.0


def test_variant_summary_aggregates_explicit_diagnostic_counters():
    common = {
        "variant": "all",
        "success": False,
        "failure": True,
        "steps": 10,
        "intervention_fraction": 0.1,
        "max_slack": 0.0,
        "termination_reason": "safety_violation",
        "violated_constraints": "geometry",
    }
    rows = [
        {
            **common,
            "qp_failure_count": 2,
            "qp_failure_reasons": "hard_constraints_infeasible[cone]:2",
            "qp_failure_minimal_infeasible_constraint_groups": "cone:2",
            "minimum_qp_failure_hard_feasibility_margin": -0.01,
            "strict_protected_barrier_violation_count": 1,
            "strict_protected_barriers_violated": "cone_lower",
            "protected_barrier_violation_beyond_tolerance_count": 0,
            "protected_barriers_violated_beyond_tolerance": "",
        },
        {
            **common,
            "qp_failure_count": 1,
            "qp_failure_reasons": "hard_constraints_infeasible[cone]:1",
            "qp_failure_minimal_infeasible_constraint_groups": "cone:1",
            "minimum_qp_failure_hard_feasibility_margin": -0.02,
            "strict_protected_barrier_violation_count": 1,
            "strict_protected_barriers_violated": "cone_lower",
            "protected_barrier_violation_beyond_tolerance_count": 1,
            "protected_barriers_violated_beyond_tolerance": "cone_lower",
        },
    ]

    summary = summarize_variant("all", rows)

    assert summary["qp_failure_count"] == 3
    assert summary["qp_failure_reasons"] == "hard_constraints_infeasible[cone]:3"
    assert summary["qp_failure_minimal_infeasible_constraint_groups"] == "cone:3"
    assert summary["minimum_qp_failure_hard_feasibility_margin"] == -0.02
    assert summary["strict_protected_barrier_violation_count"] == 2
    assert summary["protected_barrier_violation_beyond_tolerance_count"] == 1

