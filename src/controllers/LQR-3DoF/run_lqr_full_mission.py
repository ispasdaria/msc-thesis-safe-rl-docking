import argparse
import json
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
    Q = build_Q_closing()
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


def create_runner(seed, closing_max_time, final_max_time, integrator="rk4"):
    return DockingMissionRunner(
        closing_env_kwargs={
            "integrator": integrator,
            "seed": seed,
            "max_time": closing_max_time,
        },
        final_env_kwargs={
            "integrator": integrator,
            "seed": seed,
            "max_time": final_max_time,
        },
    )


def configure_runner(runner):
    closing_controller = build_closing_controller(runner.closing_env)
    final_controller = build_final_controller(runner.final_env)

    runner.set_controller("closing", closing_controller)
    runner.set_controller("final", final_controller)

    return runner


def run_full_mission(seed, mode, closing_max_time, final_max_time, integrator="rk4"):
    runner = create_runner(seed, closing_max_time, final_max_time, integrator=integrator)
    configure_runner(runner)
    mission_result = runner.run_mission_with_registered(mode=mode)
    mission_result["seed"] = seed
    return runner, mission_result


def run_one_mission(seed=0, mode="testing", closing_max_time=12000.0, final_max_time=5000.0):
    runner, mission_result = run_full_mission(
        seed=seed,
        mode=mode,
        closing_max_time=closing_max_time,
        final_max_time=final_max_time,
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
        "mission_success_rate": mission_successes / n,
        "closing_success_rate": closing_successes / n,
        "final_attempt_rate": final_attempts / n,
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


def get_saturation_fraction(env, phase):
    if phase == "closing":
        sat = analyze_closing_saturation(env)
    else:
        sat = analyze_final_saturation(env)
    return float(sat.get("saturation_fraction_total", 0.0))


def get_closing_final_error_norm(result, runner):
    terminal_state = result["closing_result"].get("terminal_state")
    p3_target = getattr(runner.closing_env, "p3_target", None)

    if terminal_state is None or p3_target is None:
        return float("nan")

    e = np.asarray(terminal_state, dtype=float) - np.asarray(p3_target, dtype=float)

    # Wrap attitude error.
    e[4] = (e[4] + np.pi) % (2.0 * np.pi) - np.pi

    return float(np.linalg.norm(e))


def build_mission_record(result, runner):
    final_attempted = result["final_result"] is not None
    closing_sat = get_saturation_fraction(runner.closing_env, "closing")
    final_sat = get_saturation_fraction(runner.final_env, "final") if final_attempted else 0.0

    closing_reason = result["closing_result"]["termination_reason"]
    final_reason = (
        None
        if result["final_result"] is None
        else result["final_result"]["termination_reason"]
    )

    return {
        "seed": int(result["seed"]),
        "mission_success": bool(result["mission_success"]),
        "failed_phase": result["failed_phase"],
        "closing_success": bool(result["closing_result"]["success"]),
        "final_attempted": bool(final_attempted),
        "final_success": bool(final_attempted and result["final_result"]["success"]),
        "closing_reason": closing_reason,
        "final_reason": final_reason,
        "closing_steps": int(result["closing_result"]["num_steps"]),
        "final_steps": 0 if not final_attempted else int(result["final_result"]["num_steps"]),
        "closing_sat": closing_sat,
        "final_sat": final_sat,
        "closing_final_error_norm": get_closing_final_error_norm(result, runner),
    }


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
    mode="testing",
    closing_max_time=12000.0,
    final_max_time=5000.0,
    save_mission_histories=False,
    history_dir=os.path.join(LQR_ROOT, "results", "lqr_mission", "mission_histories"),
):
    mission_results = []
    records = []

    if save_mission_histories:
        os.makedirs(history_dir, exist_ok=True)
        print(f"selected mission histories will be saved to: {history_dir}")

    for seed in range(num_missions):
        runner, result = run_full_mission(
            seed=seed,
            mode=mode,
            closing_max_time=closing_max_time,
            final_max_time=final_max_time,
        )

        record = build_mission_record(result, runner)
        records.append(record)
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

    summary = summarize_missions(mission_results)

    print("\nMISSION SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")

    if save_mission_histories:
        selected = select_representative_missions(records)
        print("\nSaving selected representative mission histories")
        print("=" * 60)
        for label, record in selected:
            # Re-run the selected mission so we do not need to store every trajectory in RAM.
            selected_runner, selected_result = run_full_mission(
                seed=record["seed"],
                mode=mode,
                closing_max_time=closing_max_time,
                final_max_time=final_max_time,
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


def run_runner_closing_only(num_episodes=100, mode="testing", closing_max_time=12000.0, final_max_time=5000.0):
    results = []

    for seed in range(num_episodes):
        runner = create_runner(seed, closing_max_time, final_max_time)
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
    parser.add_argument("--mode", type=str, default="testing", choices=["nominal", "testing", "training"])
    parser.add_argument("--closing-max-time", type=float, default=12000.0)
    parser.add_argument("--final-max-time", type=float, default=5000.0)
    parser.add_argument("--skip-closing-only", action="store_true")
    parser.add_argument("--skip-one-mission", action="store_true")
    parser.add_argument("--save-mission-histories", action="store_true")
    parser.add_argument(
        "--history-dir",
        type=str,
        default=os.path.join(LQR_ROOT, "results", "lqr_mission", "mission_histories"),
    )
    args = parser.parse_args()

    if not args.skip_closing_only:
        run_runner_closing_only(
            num_episodes=args.num_missions,
            mode=args.mode,
            closing_max_time=args.closing_max_time,
            final_max_time=args.final_max_time,
        )

    if not args.skip_one_mission:
        run_one_mission(
            seed=0,
            mode=args.mode,
            closing_max_time=args.closing_max_time,
            final_max_time=args.final_max_time,
        )

    mission_results, summary = run_many_missions(
        num_missions=args.num_missions,
        mode=args.mode,
        closing_max_time=args.closing_max_time,
        final_max_time=args.final_max_time,
        save_mission_histories=args.save_mission_histories,
        history_dir=args.history_dir,
    )
    inspect_final_failures(mission_results)


if __name__ == "__main__":
    main()
