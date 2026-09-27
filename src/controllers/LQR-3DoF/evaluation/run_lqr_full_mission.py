import argparse
import csv
import json
import math
import os
import sys
from collections import Counter

import numpy as np

LQR_ROOT = os.path.abspath(os.path.dirname(__file__))
while not os.path.exists(os.path.join(LQR_ROOT, "lqr_3dof_closing.py")):
    parent_dir = os.path.dirname(LQR_ROOT)
    if parent_dir == LQR_ROOT:
        break
    LQR_ROOT = parent_dir

SRC_PATH = os.path.abspath(os.path.join(LQR_ROOT, "..", ".."))
if LQR_ROOT not in sys.path:
    sys.path.insert(0, LQR_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.docking3dof.runners import DockingMissionRunner

from lqr_3dof_closing import (
    LQRClosingController,
    build_Q_closing,
    build_R_closing,
    build_linearized_3dof_matrices as build_closing_matrices,
    compute_lqr_gain as compute_closing_lqr_gain,
    analyze_saturation as analyze_closing_saturation,
)

from lqr_3dof_final import (
    LQRFinalController,
    build_Q_final,
    build_R_final,
    build_port_error_mapping,
    build_linearized_3dof_matrices as build_final_matrices,
    compute_lqr_gain as compute_final_lqr_gain,
    analyze_saturation as analyze_final_saturation,
)


def make_json_safe(value):
    """Convert NumPy values/arrays and nested containers to JSON-safe objects."""
    if isinstance(value, np.ndarray):
        return [make_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return str(value)


def build_closing_controller(env):
    Q = build_Q_closing(dt=env.dt)
    R = build_R_closing(env)
    A, B = build_closing_matrices(env)
    K, _ = compute_closing_lqr_gain(A, B, Q, R)

    # Important: do not override v_fast, d_fast, d_slow here.
    # The controller uses its own tuned/default configuration.
    return LQRClosingController(K)


def build_final_controller(env):
    Q_port = build_Q_final()
    R = build_R_final(env)
    A, B = build_final_matrices(env)
    H = build_port_error_mapping(env)

    Q_com = H.T @ Q_port @ H

    K, _ = compute_final_lqr_gain(A, B, Q_com, R)

    return LQRFinalController(K, H)


class ControllerMetricTracker:
    """Track applied-command metrics without changing a controller's output."""

    def __init__(self, controller):
        self.controller = controller
        # Match the environment-history convention, which includes the reset
        # state with a zero command as its first sample.
        self.samples = 1
        self.saturated_any = 0
        self.saturated_channels = np.zeros(3, dtype=np.int64)
        self.translational_effort = 0.0
        self.angular_effort = 0.0

    def compute_action(self, env, state):
        action = np.asarray(self.controller.compute_action(env, state), dtype=float)
        applied = env.clip_action(action)
        saturated = np.abs(action - applied) > 1e-12
        self.samples += 1
        self.saturated_any += int(np.any(saturated))
        self.saturated_channels += saturated
        self.translational_effort += float(np.linalg.norm(applied[:2]) * env.dt)
        self.angular_effort += float(abs(applied[2]) * env.dt)
        return action

    def metrics(self):
        denominator = float(self.samples)
        return {
            "saturation_fraction_total": self.saturated_any / denominator,
            "saturation_fraction_channels": self.saturated_channels / denominator,
            "translational_effort": self.translational_effort,
            "angular_effort": self.angular_effort,
        }


def print_phase_result(name, result):
    print(f"\n{name.upper()} RESULT")
    print("-" * 50)
    print("success:", result["success"])
    print("failure:", result["failure"])
    print("time_limit:", result["time_limit"])
    print("termination_reason:", result["termination_reason"])
    print("num_steps:", result["num_steps"])
    print("terminal_state:", result["terminal_state"])

    final_info = result["final_info"]
    print("final_info:", final_info)


def create_runner(
    seed,
    closing_max_time,
    final_max_time,
    integrator="rk4",
    closing_dt=1.0,
    final_dt=1.0,
):
    return DockingMissionRunner(
        closing_env_kwargs={
            "integrator": integrator,
            "seed": seed,
            "max_time": closing_max_time,
            "dt": closing_dt,
        },
        final_env_kwargs={
            "integrator": integrator,
            "seed": seed,
            "max_time": final_max_time,
            "dt": final_dt,
        },
    )


def configure_runner(runner):
    closing_tracker = ControllerMetricTracker(
        build_closing_controller(runner.closing_env)
    )
    final_tracker = ControllerMetricTracker(build_final_controller(runner.final_env))

    runner.set_controller("closing", closing_tracker)
    runner.set_controller("final", final_tracker)
    runner.phase_metric_trackers = {
        "closing": closing_tracker,
        "final": final_tracker,
    }

    return runner


def run_full_mission(
    seed,
    mode,
    closing_max_time,
    final_max_time,
    integrator="rk4",
    closing_dt=1.0,
    final_dt=1.0,
    retain_history=True,
):
    runner = create_runner(
        seed,
        closing_max_time,
        final_max_time,
        closing_dt=closing_dt,
        final_dt=final_dt,
        integrator=integrator,
    )
    configure_runner(runner)
    if not retain_history:
        # The 600-mission dt=0.1 campaign can contain tens of millions of
        # control steps. Metrics are accumulated by ControllerMetricTracker,
        # so retaining every per-step dictionary is unnecessary.
        runner.closing_env.log_step = lambda **_: None
        runner.final_env.log_step = lambda **_: None
    mission_result = runner.run_mission_with_registered(mode=mode)
    mission_result["seed"] = seed
    return runner, mission_result


def run_one_mission(
    seed=0,
    mode="testing",
    closing_max_time=12000.0,
    final_max_time=5000.0,
    closing_dt=1.0,
    final_dt=1.0,
):
    runner, mission_result = run_full_mission(
        seed=seed,
        mode=mode,
        closing_max_time=closing_max_time,
        final_max_time=final_max_time,
        closing_dt=closing_dt,
        final_dt=final_dt,
    )

    print("\nFULL MISSION RESULT")
    print("=" * 60)
    print("mission_success:", mission_result["mission_success"])
    print("failed_phase:", mission_result["failed_phase"])

    print_phase_result("closing", mission_result["closing_result"])

    closing_sat = analyze_closing_saturation(runner.closing_env)
    print("closing_saturation_total:", closing_sat["saturation_fraction_total"])

    if mission_result["final_result"] is not None:
        print_phase_result("final", mission_result["final_result"])

        final_sat = analyze_final_saturation(runner.final_env)
        print("final_saturation_total:", final_sat["saturation_fraction_total"])

        print("\nTransition state from closing to final:")
        print(mission_result["transition_state"])

    return mission_result


def summarize_missions(mission_results):
    n = len(mission_results)

    mission_successes = sum(r["mission_success"] for r in mission_results)
    closing_successes = sum(r["closing_result"]["success"] for r in mission_results)
    final_attempts = sum(r["final_result"] is not None for r in mission_results)
    final_successes = sum(
        (r["final_result"] is not None and r["final_result"]["success"])
        for r in mission_results
    )

    failed_phases = Counter(r["failed_phase"] for r in mission_results)
    closing_reasons = Counter(
        r["closing_result"]["termination_reason"] for r in mission_results
    )
    final_reasons = Counter(
        r["final_result"]["termination_reason"]
        for r in mission_results
        if r["final_result"] is not None
    )

    closing_safety = sum(
        r["closing_result"]["termination_reason"] == "safety_violation"
        for r in mission_results
    )
    final_safety = sum(
        r["final_result"] is not None
        and r["final_result"]["termination_reason"] == "safety_violation"
        for r in mission_results
    )

    return {
        "num_missions": n,
        "mission_successes": mission_successes,
        "mission_success_rate": mission_successes / n,
        "closing_successes": closing_successes,
        "closing_success_rate": closing_successes / n,
        "final_attempts": final_attempts,
        "final_attempt_rate": final_attempts / n,
        "final_successes": final_successes,
        "final_success_rate_given_attempt": (
            final_successes / final_attempts if final_attempts > 0 else 0.0
        ),
        "overall_final_success_rate": final_successes / n,
        "failed_phase_counts": dict(failed_phases),
        "closing_termination_reasons": dict(closing_reasons),
        "final_termination_reasons": dict(final_reasons),
        "closing_safety_rate": closing_safety / n,
        "final_safety_rate": final_safety / n,
    }


def wilson_interval(count, total, z=1.959963984540054):
    """Return the two-sided 95% Wilson interval for a binomial proportion."""
    if total <= 0:
        return float("nan"), float("nan")
    p = count / float(total)
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    half_width = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total**2))
    half_width /= denominator
    return center - half_width, center + half_width


def finite_mean(records, key, predicate=lambda _: True):
    values = [
        float(record[key])
        for record in records
        if predicate(record) and np.isfinite(float(record[key]))
    ]
    return float(np.mean(values)) if values else float("nan")


def summarize_mission_records(records, configuration):
    n = len(records)
    mission_successes = sum(record["mission_success"] for record in records)
    closing_successes = sum(record["closing_success"] for record in records)
    closing_timeouts = sum(record["closing_reason"] == "timeout" for record in records)
    closing_safety = sum(record["closing_hard_safety"] for record in records)
    final_attempts = sum(record["final_attempted"] for record in records)
    final_successes = sum(record["final_success"] for record in records)
    final_timeouts = sum(record["final_reason"] == "timeout" for record in records)
    final_safety = sum(record["final_hard_safety"] for record in records)
    overall_safety = sum(record["mission_hard_safety"] for record in records)
    success_low, success_high = wilson_interval(mission_successes, n)
    safety_low, safety_high = wilson_interval(overall_safety, n)

    return {
        "configuration": configuration,
        "counts_and_rates": {
            "num_missions": n,
            "closing_successes": closing_successes,
            "closing_success_rate": closing_successes / n,
            "closing_timeouts": closing_timeouts,
            "closing_hard_safety_violations": closing_safety,
            "missions_reaching_final_approach": final_attempts,
            "final_successes": final_successes,
            "final_success_rate_given_attempt": (
                final_successes / final_attempts if final_attempts else float("nan")
            ),
            "final_timeouts": final_timeouts,
            "final_hard_safety_violations": final_safety,
            "overall_mission_successes": mission_successes,
            "overall_mission_success_rate": mission_successes / n,
            "overall_mission_success_wilson_95": {
                "lower": success_low,
                "upper": success_high,
            },
            "missions_with_hard_safety_violation": overall_safety,
            "overall_hard_safety_violation_rate": overall_safety / n,
            "overall_hard_safety_wilson_95": {
                "lower": safety_low,
                "upper": safety_high,
            },
        },
        "mean_completion_times_seconds": {
            "closing_all": finite_mean(records, "closing_elapsed_time_s"),
            "closing_successful": finite_mean(
                records, "closing_elapsed_time_s", lambda record: record["closing_success"]
            ),
            "final_attempted": finite_mean(
                records, "final_elapsed_time_s", lambda record: record["final_attempted"]
            ),
            "final_successful": finite_mean(
                records, "final_elapsed_time_s", lambda record: record["final_success"]
            ),
        },
        "mean_terminal_errors": {
            "closing_error_norm": finite_mean(records, "closing_final_error_norm"),
            "closing_abs_axial_position_m": finite_mean(records, "closing_abs_ex_m"),
            "closing_abs_lateral_position_m": finite_mean(records, "closing_abs_ey_m"),
            "closing_abs_axial_velocity_mps": finite_mean(records, "closing_abs_evx_mps"),
            "closing_abs_lateral_velocity_mps": finite_mean(records, "closing_abs_evy_mps"),
            "closing_abs_attitude_rad": finite_mean(records, "closing_abs_epsi_rad"),
            "closing_abs_angular_rate_radps": finite_mean(records, "closing_abs_eomega_radps"),
            "final_port_distance_m": finite_mean(
                records, "final_port_distance_m", lambda record: record["final_attempted"]
            ),
            "final_port_speed_mps": finite_mean(
                records, "final_port_speed_mps", lambda record: record["final_attempted"]
            ),
            "final_abs_attitude_rad": finite_mean(
                records, "final_abs_attitude_rad", lambda record: record["final_attempted"]
            ),
            "final_abs_angular_rate_radps": finite_mean(
                records, "final_abs_angular_rate_radps", lambda record: record["final_attempted"]
            ),
        },
        "mean_actuator_saturation_fractions": {
            "closing_any": finite_mean(records, "closing_sat_any"),
            "closing_ax": finite_mean(records, "closing_sat_ax"),
            "closing_ay": finite_mean(records, "closing_sat_ay"),
            "closing_alphaz": finite_mean(records, "closing_sat_alphaz"),
            "final_any_given_attempt": finite_mean(
                records, "final_sat_any", lambda record: record["final_attempted"]
            ),
            "final_ax_given_attempt": finite_mean(
                records, "final_sat_ax", lambda record: record["final_attempted"]
            ),
            "final_ay_given_attempt": finite_mean(
                records, "final_sat_ay", lambda record: record["final_attempted"]
            ),
            "final_alphaz_given_attempt": finite_mean(
                records, "final_sat_alphaz", lambda record: record["final_attempted"]
            ),
        },
    }


def get_closing_final_error_norm(result, runner):
    terminal_state = result["closing_result"].get("terminal_state")
    p3_target = getattr(runner.closing_env, "p3_target", None)

    if terminal_state is None or p3_target is None:
        return float("nan")

    e = np.asarray(terminal_state, dtype=float) - np.asarray(p3_target, dtype=float)

    # Wrap attitude error.
    e[4] = (e[4] + np.pi) % (2.0 * np.pi) - np.pi

    return float(np.linalg.norm(e))


def get_closing_final_error_components(result, runner):
    terminal_state = result["closing_result"].get("terminal_state")
    p3_target = getattr(runner.closing_env, "p3_target", None)
    if terminal_state is None or p3_target is None:
        return np.full(6, np.nan)
    error = np.asarray(terminal_state, dtype=float) - np.asarray(p3_target, dtype=float)
    error[4] = (error[4] + np.pi) % (2.0 * np.pi) - np.pi
    return error


def build_mission_record(result, runner):
    final_attempted = result["final_result"] is not None
    closing_metrics = runner.phase_metric_trackers["closing"].metrics()
    final_metrics = runner.phase_metric_trackers["final"].metrics()
    closing_channels = closing_metrics["saturation_fraction_channels"]
    final_channels = final_metrics["saturation_fraction_channels"]
    closing_error = get_closing_final_error_components(result, runner)

    closing_reason = result["closing_result"]["termination_reason"]
    final_reason = (
        None
        if result["final_result"] is None
        else result["final_result"]["termination_reason"]
    )
    closing_info = result["closing_result"]["final_info"]
    final_info = {} if not final_attempted else result["final_result"]["final_info"]
    closing_constraints = list(closing_info.get("violated_constraints", []) or [])
    final_constraints = list(final_info.get("violated_constraints", []) or [])
    closing_hard_safety = closing_reason == "safety_violation"
    final_hard_safety = final_reason == "safety_violation"

    return {
        "seed": int(result["seed"]),
        "mission_success": bool(result["mission_success"]),
        "failed_phase": result["failed_phase"],
        "closing_success": bool(result["closing_result"]["success"]),
        "final_attempted": bool(final_attempted),
        "final_success": bool(final_attempted and result["final_result"]["success"]),
        "closing_dt_s": float(runner.closing_env.dt),
        "final_dt_s": float(runner.final_env.dt),
        "closing_reason": closing_reason,
        "final_reason": final_reason,
        "closing_hard_safety": bool(closing_hard_safety),
        "final_hard_safety": bool(final_hard_safety),
        "mission_hard_safety": bool(closing_hard_safety or final_hard_safety),
        "closing_violated_constraints": ";".join(closing_constraints),
        "final_violated_constraints": ";".join(final_constraints),
        "closing_steps": int(result["closing_result"]["num_steps"]),
        "final_steps": 0 if not final_attempted else int(result["final_result"]["num_steps"]),
        "closing_elapsed_time_s": float(runner.closing_env.time),
        "final_elapsed_time_s": float(runner.final_env.time) if final_attempted else float("nan"),
        "closing_sat_any": float(closing_metrics["saturation_fraction_total"]),
        "closing_sat_ax": float(closing_channels[0]),
        "closing_sat_ay": float(closing_channels[1]),
        "closing_sat_alphaz": float(closing_channels[2]),
        "final_sat_any": float(final_metrics["saturation_fraction_total"]) if final_attempted else float("nan"),
        "final_sat_ax": float(final_channels[0]) if final_attempted else float("nan"),
        "final_sat_ay": float(final_channels[1]) if final_attempted else float("nan"),
        "final_sat_alphaz": float(final_channels[2]) if final_attempted else float("nan"),
        # Backward-compatible aliases used by representative-mission selection.
        "closing_sat": float(closing_metrics["saturation_fraction_total"]),
        "final_sat": float(final_metrics["saturation_fraction_total"]) if final_attempted else 0.0,
        "closing_translational_effort_mps": float(closing_metrics["translational_effort"]),
        "closing_angular_effort_radps": float(closing_metrics["angular_effort"]),
        "final_translational_effort_mps": float(final_metrics["translational_effort"]) if final_attempted else float("nan"),
        "final_angular_effort_radps": float(final_metrics["angular_effort"]) if final_attempted else float("nan"),
        "closing_final_error_norm": get_closing_final_error_norm(result, runner),
        "closing_abs_ex_m": float(abs(closing_error[0])),
        "closing_abs_ey_m": float(abs(closing_error[1])),
        "closing_abs_evx_mps": float(abs(closing_error[2])),
        "closing_abs_evy_mps": float(abs(closing_error[3])),
        "closing_abs_epsi_rad": float(abs(closing_error[4])),
        "closing_abs_eomega_radps": float(abs(closing_error[5])),
        "final_port_distance_m": float(final_info.get("port_distance", np.nan)),
        "final_port_speed_mps": float(final_info.get("port_speed", np.nan)),
        "final_abs_attitude_rad": float(abs(final_info.get("psi_e", np.nan))),
        "final_abs_angular_rate_radps": float(abs(final_info.get("omega", np.nan))),
    }


def write_mission_records_csv(records, output_path):
    """Write one reproducible, analysis-ready row per mission."""
    if not records:
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(output_path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def select_representative_missions(records):
    """Select only representative missions. Duplicate seeds get combined labels."""
    selected_by_seed = {}

    def add(label, record):
        if record is None:
            return
        seed = record["seed"]
        if seed not in selected_by_seed:
            selected_by_seed[seed] = {"labels": [label], "record": record}
        else:
            selected_by_seed[seed]["labels"].append(label)

    full_success = [r for r in records if r["mission_success"]]
    closing_failures = [r for r in records if r["failed_phase"] == "closing"]
    final_failures = [r for r in records if r["failed_phase"] == "final"]
    closing_timeouts = [r for r in records if r["closing_reason"] == "timeout"]
    finite_closing_err = [
        r for r in records if np.isfinite(r.get("closing_final_error_norm", np.nan))
    ]
    final_attempts = [r for r in records if r["final_attempted"]]

    if full_success:
        add("first_full_success", full_success[0])
        if len(full_success) >= 3:
            ordered = sorted(full_success, key=lambda r: r["seed"])
            add("median_full_success", ordered[len(ordered) // 2])

    if closing_failures:
        add("first_closing_failure", closing_failures[0])

    if closing_timeouts:
        ordered = sorted(closing_timeouts, key=lambda r: r["seed"])
        add("typical_closing_timeout", ordered[len(ordered) // 2])

    if finite_closing_err:
        add("worst_closing_final_error", max(finite_closing_err, key=lambda r: r["closing_final_error_norm"]))

    if final_failures:
        add("first_final_failure", final_failures[0])

    if records:
        add("highest_closing_saturation", max(records, key=lambda r: r["closing_sat"]))

    if final_attempts:
        add("highest_final_saturation", max(final_attempts, key=lambda r: r["final_sat"]))

    selected = []
    for item in selected_by_seed.values():
        selected.append(("__".join(item["labels"]), item["record"]))
    return selected


def save_full_mission_history(runner, result, record, label, output_dir, mode, closing_max_time, final_max_time):
    os.makedirs(output_dir, exist_ok=True)

    c = runner.closing_env.get_history_arrays()
    f = runner.final_env.get_history_arrays() if result["final_result"] is not None else None

    (
        closing_times,
        closing_states,
        closing_body_raw,
        closing_body_clipped,
        closing_orbital,
    ) = c

    if f is None:
        final_times = np.asarray([], dtype=float)
        final_states = np.empty((0, 6), dtype=float)
        final_body_raw = np.empty((0, 3), dtype=float)
        final_body_clipped = np.empty((0, 3), dtype=float)
        final_orbital = np.empty((0, 3), dtype=float)
    else:
        (
            final_times,
            final_states,
            final_body_raw,
            final_body_clipped,
            final_orbital,
        ) = f

    metadata = {
        "label": label,
        "seed": int(record["seed"]),
        "mode": mode,
        "mission_success": bool(result["mission_success"]),
        "failed_phase": result["failed_phase"],
        "closing_reason": result["closing_result"]["termination_reason"],
        "final_reason": None if result["final_result"] is None else result["final_result"]["termination_reason"],
        "closing_success": bool(result["closing_result"]["success"]),
        "final_attempted": result["final_result"] is not None,
        "final_success": bool(result["final_result"] is not None and result["final_result"]["success"]),
        "closing_saturation": record["closing_sat"],
        "final_saturation": record["final_sat"],
        "closing_max_time": float(closing_max_time),
        "final_max_time": float(final_max_time),
        "closing_dt": float(runner.closing_env.dt),
        "final_dt": float(runner.final_env.dt),
        "transition_state": result.get("transition_state"),
        "closing_result": result["closing_result"],
        "final_result": result["final_result"],
        "record": record,
    }

    safe_label = label.replace("/", "_").replace(" ", "_")
    status = "success" if result["mission_success"] else f"failed_{result['failed_phase']}"
    filename = f"seed{record['seed']:04d}_{safe_label}_{status}.npz"
    path = os.path.join(output_dir, filename)

    np.savez_compressed(
        path,
        closing_times=closing_times,
        closing_states=closing_states,
        closing_actions_body_raw=closing_body_raw,
        closing_actions_body_clipped=closing_body_clipped,
        closing_actions_orbital=closing_orbital,
        closing_p3_target=(
            np.asarray(runner.closing_env.p3_target, dtype=float)
            if getattr(runner.closing_env, "p3_target", None) is not None
            else np.full(6, np.nan)
        ),
        closing_p3_target_tolerance=(
            np.asarray(runner.closing_env.p3_target_tolerance, dtype=float)
            if getattr(runner.closing_env, "p3_target_tolerance", None) is not None
            else np.full(6, np.nan)
        ),
        final_times=final_times,
        final_states=final_states,
        final_actions_body_raw=final_body_raw,
        final_actions_body_clipped=final_body_clipped,
        final_actions_orbital=final_orbital,
        transition_state=(
            np.asarray(result.get("transition_state"), dtype=float)
            if result.get("transition_state") is not None
            else np.full(6, np.nan)
        ),
        metadata_json=json.dumps(make_json_safe(metadata), indent=2),
    )
    return path


def run_many_missions(
    num_missions=100,
    seed_start=0,
    mode="testing",
    closing_max_time=12000.0,
    final_max_time=5000.0,
    closing_dt=1.0,
    final_dt=1.0,
    save_mission_histories=False,
    history_dir=os.path.join(
        LQR_ROOT,
        "results",
        "lqr_mission",
        "representative_full_mission_rollout_histories_dt1p0",
    ),
    results_csv=None,
    summary_json=None,
):
    mission_results = []
    records = []

    if save_mission_histories:
        os.makedirs(history_dir, exist_ok=True)
        print(f"selected mission histories will be saved to: {history_dir}")

    for seed in range(seed_start, seed_start + num_missions):
        runner, result = run_full_mission(
            seed=seed,
            mode=mode,
            closing_max_time=closing_max_time,
            final_max_time=final_max_time,
            closing_dt=closing_dt,
            final_dt=final_dt,
            retain_history=save_mission_histories,
        )

        record = build_mission_record(result, runner)
        records.append(record)
        # Keep only aggregate/terminal data across the campaign. Selected
        # histories, when requested, are reproduced in a separate rerun below.
        result["closing_result"]["history"] = None
        if result["final_result"] is not None:
            result["final_result"]["history"] = None
        mission_results.append(result)

        print(
            f"seed={seed:03d} | "
            f"mission_success={result['mission_success']} | "
            f"failed_phase={result['failed_phase']} | "
            f"closing_reason={result['closing_result']['termination_reason']} | "
            f"final_reason="
            f"{None if result['final_result'] is None else result['final_result']['termination_reason']} | "
            f"closing_sat={record['closing_sat']:.3f} | final_sat={record['final_sat']:.3f}"
        )

    configuration = {
        "phase_controllers": "frozen_selected_LQR",
        "mode": mode,
        "seed_start": int(seed_start),
        "seed_end": int(seed_start + num_missions - 1),
        "num_missions": int(num_missions),
        "closing_dt_seconds": float(closing_dt),
        "final_dt_seconds": float(final_dt),
        "closing_max_time_seconds": float(closing_max_time),
        "final_max_time_seconds": float(final_max_time),
        "handover": "actual closing terminal state passed directly as final initial state",
        "closing_controller": {
            "Q": build_Q_closing(dt=runner.closing_env.dt),
            "R": build_R_closing(runner.closing_env),
            "K": runner.phase_metric_trackers["closing"].controller.K,
            "v_fast_mps": runner.phase_metric_trackers["closing"].controller.v_fast,
            "d_fast_m": runner.phase_metric_trackers["closing"].controller.d_fast,
            "d_slow_m": runner.phase_metric_trackers["closing"].controller.d_slow,
        },
        "final_controller": {
            "Q_port": build_Q_final(),
            "R": build_R_final(runner.final_env),
            "K": runner.phase_metric_trackers["final"].controller.K,
        },
    }
    summary = summarize_mission_records(records, configuration)

    if results_csv:
        write_mission_records_csv(records, results_csv)
        print(f"mission records written to: {results_csv}")

    if summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(summary_json)), exist_ok=True)
        with open(summary_json, "x", encoding="utf-8") as handle:
            json.dump(make_json_safe(summary), handle, indent=2, allow_nan=True)
            handle.write("\n")
        print(f"mission summary written to: {summary_json}")

    print("\nMISSION SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")

    if save_mission_histories:
        selected = select_representative_missions(records)
        print("\nSaving selected representative mission histories")
        print("=" * 60)
        for label, record in selected:
            # Regenerate the selected history without retaining every campaign trajectory.
            selected_runner, selected_result = run_full_mission(
                seed=record["seed"],
                mode=mode,
                closing_max_time=closing_max_time,
                final_max_time=final_max_time,
                closing_dt=closing_dt,
                final_dt=final_dt,
                retain_history=True,
            )
            path = save_full_mission_history(
                runner=selected_runner,
                result=selected_result,
                record=record,
                label=label,
                output_dir=history_dir,
                mode=mode,
                closing_max_time=closing_max_time,
                final_max_time=final_max_time,
            )
            print(f"  {label}: seed={record['seed']:03d} -> {path}")


    return mission_results, summary


def run_runner_closing_only(
    num_episodes=100,
    seed_start=0,
    mode="testing",
    closing_max_time=12000.0,
    final_max_time=5000.0,
    closing_dt=1.0,
    final_dt=1.0,
):
    results = []

    for seed in range(seed_start, seed_start + num_episodes):
        runner = create_runner(
            seed,
            closing_max_time,
            final_max_time,
            closing_dt=closing_dt,
            final_dt=final_dt,
        )
        closing_controller = build_closing_controller(runner.closing_env)
        runner.set_controller("closing", closing_controller)

        result = runner.run_phase_with_registered(phase="closing", mode=mode)
        result["seed"] = seed
        results.append(result)

        print(
            f"seed={seed:03d} | "
            f"success={result['success']} | "
            f"reason={result['termination_reason']} | "
            f"steps={result['num_steps']}"
        )

    successes = sum(r["success"] for r in results)
    timeouts = sum(r["termination_reason"] == "timeout" for r in results)

    print("\nRUNNER CLOSING-ONLY SUMMARY")
    print("=" * 60)
    print("num_episodes:", num_episodes)
    print("success_rate:", successes / num_episodes)
    print("timeout_rate:", timeouts / num_episodes)

    return results


def inspect_final_failures(mission_results):
    print("\nFINAL-PHASE FAILURES")
    print("=" * 60)

    for r in mission_results:
        if r["failed_phase"] != "final":
            continue

        seed = r.get("seed", None)
        transition_state = r["transition_state"]
        final_result = r["final_result"]
        final_info = final_result["final_info"]

        print(f"\nseed={seed}")
        print("transition_state:", transition_state)
        print("final termination:", final_result["termination_reason"])
        print("violated_constraints:", final_info.get("violated_constraints"))
        print("port_distance:", final_info.get("port_distance"))
        print("port_speed:", final_info.get("port_speed"))
        print("s:", final_info.get("s"))
        print("rho:", final_info.get("rho"))
        print("s_dot:", final_info.get("s_dot"))
        print("psi_e:", final_info.get("psi_e"))
        print("omega:", final_info.get("omega"))


def main():
    parser = argparse.ArgumentParser(description="Evaluate full LQR docking missions and optionally save selected histories.")
    parser.add_argument("--num-missions", type=int, default=300)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--mode", type=str, default="testing", choices=["nominal", "testing", "training"])
    parser.add_argument("--closing-dt", type=float, default=1.0)
    parser.add_argument("--final-dt", type=float, default=1.0)
    parser.add_argument("--closing-max-time", type=float, default=12000.0)
    parser.add_argument("--final-max-time", type=float, default=5000.0)
    parser.add_argument("--skip-closing-only", action="store_true")
    parser.add_argument("--skip-one-mission", action="store_true")
    parser.add_argument("--save-mission-histories", action="store_true")
    parser.add_argument(
        "--history-dir",
        type=str,
        default=os.path.join(
            LQR_ROOT,
            "results",
            "lqr_mission",
            "representative_full_mission_rollout_histories_dt1p0",
        ),
    )
    parser.add_argument("--results-csv", type=str, default=None)
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Build both phase controllers, print configuration, and exit without missions.",
    )
    args = parser.parse_args()

    if args.num_missions <= 0:
        parser.error("--num-missions must be positive")
    if args.closing_max_time <= 0.0 or args.final_max_time <= 0.0:
        parser.error("phase maximum times must be positive")

    check_runner = create_runner(
        seed=args.seed_start,
        closing_max_time=args.closing_max_time,
        final_max_time=args.final_max_time,
        closing_dt=args.closing_dt,
        final_dt=args.final_dt,
    )
    configure_runner(check_runner)
    print("FULL-MISSION LQR CONFIGURATION")
    print("=" * 60)
    print(f"seeds: {args.seed_start}-{args.seed_start + args.num_missions - 1}")
    print(f"closing dt: {check_runner.closing_env.dt:g} s")
    print(f"final dt: {check_runner.final_env.dt:g} s")
    print(f"closing max physical time: {check_runner.closing_env.max_time:g} s")
    print(f"final max physical time: {check_runner.final_env.max_time:g} s")
    print(
        "closing Q lateral-velocity weight: "
        f"{build_Q_closing(dt=check_runner.closing_env.dt)[3, 3]:g}"
    )
    print("closing K:")
    print(check_runner.phase_metric_trackers["closing"].controller.K)
    print("final K:")
    print(check_runner.phase_metric_trackers["final"].controller.K)
    print("handover: actual closing terminal state -> final initial_state")
    print("final handover reset: no P3 state resampling")

    if args.check_only:
        print("Check-only mode: no missions were run and no files were written.")
        return

    existing_outputs = [
        path
        for path in (args.results_csv, args.summary_json)
        if path is not None and os.path.exists(path)
    ]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite existing campaign output: "
            + ", ".join(existing_outputs)
        )

    if not args.skip_closing_only:
        run_runner_closing_only(
            num_episodes=args.num_missions,
            seed_start=args.seed_start,
            mode=args.mode,
            closing_max_time=args.closing_max_time,
            final_max_time=args.final_max_time,
            closing_dt=args.closing_dt,
            final_dt=args.final_dt,
        )

    if not args.skip_one_mission:
        run_one_mission(
            seed=args.seed_start,
            mode=args.mode,
            closing_max_time=args.closing_max_time,
            final_max_time=args.final_max_time,
            closing_dt=args.closing_dt,
            final_dt=args.final_dt,
        )

    mission_results, summary = run_many_missions(
        num_missions=args.num_missions,
        seed_start=args.seed_start,
        mode=args.mode,
        closing_max_time=args.closing_max_time,
        final_max_time=args.final_max_time,
        closing_dt=args.closing_dt,
        final_dt=args.final_dt,
        save_mission_histories=args.save_mission_histories,
        history_dir=args.history_dir,
        results_csv=args.results_csv,
        summary_json=args.summary_json,
    )
    inspect_final_failures(mission_results)


if __name__ == "__main__":
    main()
