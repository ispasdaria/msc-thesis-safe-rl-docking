import os
import sys
from collections import Counter

import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.docking3dof.core import Docking3DoFEnv

from envs.docking3dof.runners import DockingMissionRunner

from controllers.lqr_3dof_closing import (
    LQRClosingController,
    build_Q_closing,
    build_R_closing,
    build_linearized_3dof_matrices as build_closing_matrices,
    compute_lqr_gain as compute_closing_lqr_gain,
    analyze_saturation as analyze_closing_saturation,
)

from controllers.lqr_3dof_final import (
    LQRFinalController,
    build_Q_final,
    build_R_final,
    build_port_error_mapping,
    build_linearized_3dof_matrices as build_final_matrices,
    compute_lqr_gain as compute_final_lqr_gain,
    analyze_saturation as analyze_final_saturation,
)


def build_closing_controller(env):
    Q = build_Q_closing()
    R = build_R_closing(env)
    A, B = build_closing_matrices(env)
    K, _ = compute_closing_lqr_gain(A, B, Q, R)

    return LQRClosingController(
        K,
        v_fast=-0.08,
        d_fast=100.0,
        d_slow=10.0,
    )


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


def run_one_mission(seed=0, mode="testing"):
    runner = DockingMissionRunner(
        closing_env_kwargs={
            "integrator": "rk4",
            "seed": seed,
            "max_time": 12000.0,
        },
        final_env_kwargs={
            "integrator": "rk4",
            "seed": seed,
            "max_time": 5000.0,
        },
    )

    closing_controller = build_closing_controller(runner.closing_env)
    final_controller = build_final_controller(runner.final_env)

    runner.set_controller("closing", closing_controller)
    runner.set_controller("final", final_controller)

    mission_result = runner.run_mission_with_registered(mode=mode)

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

    closing_successes = sum(
        r["closing_result"]["success"] for r in mission_results
    )

    final_attempts = sum(
        r["final_result"] is not None for r in mission_results
    )

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

    summary = {
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
        "final_safety_rate": final_safety / n
    }

    return summary


def run_many_missions(num_missions=100, mode="testing"):
    mission_results = []

    for seed in range(num_missions):
        runner = DockingMissionRunner(
            closing_env_kwargs={
                "integrator": "rk4",
                "seed": seed,
                "max_time": 12000.0,
            },
            final_env_kwargs={
                "integrator": "rk4",
                "seed": seed,
                "max_time": 5000.0,
            },
        )

        closing_controller = build_closing_controller(runner.closing_env)
        final_controller = build_final_controller(runner.final_env)

        runner.set_controller("closing", closing_controller)
        runner.set_controller("final", final_controller)

        result = runner.run_mission_with_registered(mode=mode)
        result["seed"] = seed

        mission_results.append(result)

        print(
            f"seed={seed:03d} | "
            f"mission_success={result['mission_success']} | "
            f"failed_phase={result['failed_phase']} | "
            f"closing_reason={result['closing_result']['termination_reason']} | "
            f"final_reason="
            f"{None if result['final_result'] is None else result['final_result']['termination_reason']}"
        )

    summary = summarize_missions(mission_results)

    print("\nMISSION SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")

    return mission_results, summary

def run_runner_closing_only(num_episodes=100, mode="testing"):
    results = []

    for seed in range(num_episodes):
        runner = DockingMissionRunner(
            closing_env_kwargs={
                "integrator": "rk4",
                "seed": seed,
                "max_time": 12000.0,
            },
            final_env_kwargs={
                "integrator": "rk4",
                "seed": seed,
                "max_time": 5000.0,
            },
        )
        closing_controller = build_closing_controller(runner.closing_env)

        runner.set_controller("closing", closing_controller)

        result = runner.run_phase_with_registered(
            phase="closing",
            mode=mode,
        )

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


if __name__ == "__main__":
    # 1. Check that closing-only inside the runner matches the closing sweep.
    run_runner_closing_only(num_episodes=300, mode="testing")

    # 2. Inspect one detailed full mission.
    run_one_mission(seed=0, mode="testing")

    # 3. Run 100 full missions and inspect final-phase failures.
    mission_results, summary = run_many_missions(num_missions=300, mode="testing")
    inspect_final_failures(mission_results)