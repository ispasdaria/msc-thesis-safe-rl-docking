import numpy as np

from controllers.cbf_3DoF.closing_barriers import ClosingCBFConfig
from controllers.cbf_3DoF.evaluate_closing_cbf import (
    build_paired_comparison,
    classify_protected_barrier_violations,
    diagnose_qp_failure,
    run_rollout,
)
from envs.docking3dof.core import Docking3DoFEnv


class ZeroPolicy:
    def predict(self, observation, deterministic=True):
        assert deterministic
        return np.zeros((1, 3), dtype=np.float64), None


class IdentityVecNormalize:
    def normalize_obs(self, observation):
        return observation


def test_closing_barrier_reporting_separates_strict_and_tolerant_crossings():
    strict, beyond = classify_protected_barrier_violations(
        {
            "port_speed": -0.5e-6,
            "axial_port_speed": -2.0e-6,
        }
    )

    assert strict == ("port_speed", "axial_port_speed")
    assert beyond == ("axial_port_speed",)


def test_closing_qp_failure_diagnosis_identifies_axial_family():
    env = Docking3DoFEnv(max_time=12_000.0, phase="closing", dt=1.0, integrator="rk4", seed=0)
    config = ClosingCBFConfig(
        enable_cone=False,
        enable_port_speed=False,
        enable_axial_port_speed=True,
        enable_alignment=False,
        allow_slack=False,
        solver_backend="scipy",
    )
    state = np.array([1700.0, 0.0, -0.30, 0.0, 0.0, 0.0])

    diagnosis = diagnose_qp_failure(env, state, config, "infeasible")

    assert diagnosis["reason"] == "hard_constraints_infeasible[axial_speed]"
    assert diagnosis["minimal_infeasible_constraint_groups"] == "axial_speed"
    assert diagnosis["hard_feasibility_margin"] < 0.0


def test_paired_comparison_matches_rows_by_seed():
    baseline = {
        "variant": "baseline",
        "eval_seed": 7,
        "success": False,
        "failure": False,
        "final_p3_position_error": 50.0,
        "minimum_p3_position_error": 40.0,
        "velocity_warning_fraction": 0.2,
        "cone_exit_fraction": 0.3,
        "alignment_unsafe_fraction": 0.5,
        "protected_barrier_violation_beyond_tolerance_count": 3,
    }
    both = {
        "variant": "both",
        "eval_seed": 7,
        "success": False,
        "failure": False,
        "final_p3_position_error": 45.0,
        "minimum_p3_position_error": 35.0,
        "velocity_warning_fraction": 0.0,
        "cone_exit_fraction": 0.0,
        "alignment_unsafe_fraction": 0.0,
        "protected_barrier_violation_beyond_tolerance_count": 0,
        "intervention_fraction": 0.1,
        "qp_failure_count": 0,
    }

    paired = build_paired_comparison([both, baseline])

    assert len(paired) == 1
    assert paired[0]["eval_seed"] == 7
    assert paired[0]["filtered_variant"] == "both"
    assert paired[0]["cbf_minus_baseline_final_p3_position_error"] == -5.0
    assert paired[0]["cbf_alignment_unsafe_fraction"] == 0.0


def test_short_paired_rollouts_save_post_step_and_raw_residual_diagnostics(
    tmp_path,
):
    config = ClosingCBFConfig(allow_slack=False, solver_backend="scipy")
    common = {
        "model": ZeroPolicy(),
        "vecnormalize": IdentityVecNormalize(),
        "eval_seed": 0,
        "mode": "testing",
        "max_steps": 2,
        "base_config": config,
        "output_dir": tmp_path,
    }

    baseline = run_rollout(variant="baseline", **common)
    translational = run_rollout(variant="all", **common)
    filtered = run_rollout(variant="full", **common)

    assert baseline["steps"] == translational["steps"] == filtered["steps"] == 2
    assert baseline["initial_protected_set_satisfied"]
    assert filtered["initial_protected_set_satisfied"]
    assert baseline["intervention_fraction"] == 0.0
    assert filtered["qp_failure_count"] == 0
    with np.load(tmp_path / filtered["step_log"]) as data:
        assert "post_h_port_speed" in data
        assert "post_h_axial_port_speed" in data
        assert "post_h_cone_upper" in data
        assert "post_h_cone_lower" in data
        assert "post_h_cone_axial_upper" in data
        assert "post_h_attitude_upper" in data
        assert "post_h_attitude_lower" in data
        assert "post_h_angular_rate_upper" in data
        assert "post_h_angular_rate_lower" in data
        assert "residual_port_speed" in data
        assert "residual_axial_port_speed" in data
        assert "residual_cone_upper" in data
        assert "residual_cone_lower" in data
        assert "residual_cone_axial_upper" in data
        assert "residual_attitude_upper" in data
        assert "residual_attitude_lower" in data
        assert "residual_angular_rate_upper" in data
        assert "residual_angular_rate_lower" in data
        assert "qp_failure_hard_feasibility_margin" in data
