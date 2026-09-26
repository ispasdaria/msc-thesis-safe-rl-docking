"""Compare the original Trial-16 closing SAC with the port-only variant."""

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from closing_port_observation_env import ClosingDockingPortObservationEnv
from envs.docking3dof.gym_env import Docking3DoFGymEnv


SEEDS = tuple(range(10))
MODEL_DIR = SCRIPT_DIR / "models" / "reward_shaping_trials"
OUTPUT_DIR = SCRIPT_DIR / "results" / "observation_comparison"

CONTROLLERS = {
    "docking_port_only": {
        "label": "Docking-port-only observation",
        "model": MODEL_DIR / "closing_sac_trial16_docking_port_obs_only.zip",
        "vecnorm": MODEL_DIR / "closing_sac_trial16_docking_port_obs_only_vecnormalize.pkl",
        "env_class": ClosingDockingPortObservationEnv,
    },
    "adopted": {
        "label": "Docking-port + P3 target-error observation",
        "model": MODEL_DIR / "sac_closing_p3augobs_rewardv15_optuna_trial16_400ksteps_seed0_buffer300k.zip",
        "vecnorm": MODEL_DIR / "sac_closing_p3augobs_rewardv15_optuna_trial16_400ksteps_seed0_buffer300k_vecnormalize.pkl",
        "env_class": Docking3DoFGymEnv,
    },
}


def make_env(env_class, seed=0):
    kwargs = {
        "mode": "testing",
        "dt": 1.0,
        "max_time": None,
        "integrator": "rk4",
        "normalize_action": True,
        "seed": seed,
        "closing_reward_version": "v15",
    }
    if env_class is Docking3DoFGymEnv:
        kwargs["phase"] = "closing"
    return env_class(**kwargs)


def load_controller(spec):
    for name in ("model", "vecnorm"):
        if not spec[name].is_file():
            raise FileNotFoundError(f"Missing {name}: {spec[name]}")

    dummy_env = DummyVecEnv([lambda: make_env(spec["env_class"], seed=999)])
    vecnorm = VecNormalize.load(str(spec["vecnorm"]), dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    model = SAC.load(str(spec["model"]), device="cpu")
    return model, vecnorm, dummy_env


def target_error(state, p3_target, wrap_to_pi):
    error = np.asarray(state, dtype=np.float64) - np.asarray(p3_target, dtype=np.float64)
    error[4] = wrap_to_pi(error[4])
    return error


def rollout(model, vecnorm, env_class, seed, initial_state, p3_target, max_steps=12000):
    env = make_env(env_class, seed=seed)
    obs, reset_info = env.reset(
        seed=seed,
        options={
            "initial_state": np.asarray(initial_state, dtype=np.float64).copy(),
            "p3_target": np.asarray(p3_target, dtype=np.float64).copy(),
        },
    )
    total_reward = 0.0
    final_info = reset_info

    for step in range(max_steps):
        normalized_obs = vecnorm.normalize_obs(np.asarray(obs).reshape(1, -1))
        action, _ = model.predict(normalized_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action[0])
        total_reward += float(reward)
        final_info = info
        if terminated or truncated:
            break

    times, states, actions_raw, actions_clipped, actions_orbital = env.get_history_arrays()
    wrap_to_pi = env.core_env.wrap_to_pi
    errors = np.asarray(
        [target_error(state, p3_target, wrap_to_pi) for state in states],
        dtype=np.float64,
    )
    env.close()

    position_errors = np.linalg.norm(errors[:, :2], axis=1)
    velocity_errors = np.linalg.norm(errors[:, 2:4], axis=1)
    axial_errors = np.abs(errors[:, 0])
    return {
        "seed": seed,
        "steps": int(step + 1),
        "return": total_reward,
        "success": bool(final_info.get("success", False)),
        "failure": bool(final_info.get("failure", False)),
        "time_limit": bool(final_info.get("time_limit", False)),
        "termination_reason": str(final_info.get("termination_reason", "")),
        "violated_constraints": list(final_info.get("violated_constraints", [])),
        "final_position_error": float(position_errors[-1]),
        "final_velocity_error": float(velocity_errors[-1]),
        "final_attitude_error_deg": float(np.rad2deg(abs(errors[-1, 4]))),
        "final_rate_error_deg_s": float(np.rad2deg(abs(errors[-1, 5]))),
        "times": np.asarray(times, dtype=float),
        "states": np.asarray(states, dtype=float),
        "target_errors": errors,
        "position_errors": position_errors,
        "axial_errors": axial_errors,
        "velocity_errors": velocity_errors,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scalar_row(controller_key, result):
    return {
        "controller": controller_key,
        "observation_space": CONTROLLERS[controller_key]["label"],
        "seed": result["seed"],
        "steps": result["steps"],
        "success": result["success"],
        "failure": result["failure"],
        "time_limit": result["time_limit"],
        "termination_reason": result["termination_reason"],
        "violated_constraints": ";".join(result["violated_constraints"]),
        "final_position_error_m": result["final_position_error"],
        "final_velocity_error_mps": result["final_velocity_error"],
        "final_attitude_error_deg": result["final_attitude_error_deg"],
        "final_rate_error_deg_s": result["final_rate_error_deg_s"],
        "episode_return": result["return"],
    }


def aggregate(controller_key, results):
    return {
        "controller": controller_key,
        "observation_space": CONTROLLERS[controller_key]["label"],
        "episodes": len(results),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "mean_final_position_error_m": float(np.mean([r["final_position_error"] for r in results])),
        "median_final_position_error_m": float(np.median([r["final_position_error"] for r in results])),
        "mean_final_velocity_error_mps": float(np.mean([r["final_velocity_error"] for r in results])),
        "mean_final_attitude_error_deg": float(np.mean([r["final_attitude_error_deg"] for r in results])),
        "mean_final_rate_error_deg_s": float(np.mean([r["final_rate_error_deg_s"] for r in results])),
        "mean_episode_return": float(np.mean([r["return"] for r in results])),
        "termination_reasons": dict(Counter(r["termination_reason"] for r in results)),
    }


def save_trajectory(path, result, p3_target, initial_state):
    np.savez_compressed(
        path,
        seed=result["seed"],
        times=result["times"],
        states=result["states"],
        target_errors=result["target_errors"],
        position_errors=result["position_errors"],
        axial_errors=result["axial_errors"],
        velocity_errors=result["velocity_errors"],
        p3_target=p3_target,
        initial_state=initial_state,
    )


def choose_representative_seed(results_by_controller):
    adopted = results_by_controller["adopted"]
    median_error = float(np.median([r["final_position_error"] for r in adopted]))
    selected = min(
        adopted,
        key=lambda result: abs(result["final_position_error"] - median_error),
    )
    return selected["seed"], median_error


def plot_position_error(results_by_controller, representative_seed, path):
    colors = {"docking_port_only": "#0072B2", "adopted": "#D55E00"}
    selected = {
        key: next(r for r in results if r["seed"] == representative_seed)
        for key, results in results_by_controller.items()
    }
    figure, axis = plt.subplots(figsize=(12, 6.5))
    for key, result in selected.items():
        axis.plot(
            result["times"],
            result["axial_errors"],
            color=colors[key],
            linewidth=2.0,
            label=CONTROLLERS[key]["label"],
        )

    position_tolerance = 20.0
    axis.axhline(
        position_tolerance,
        color="#56B4E9",
        linestyle=(0, (5, 2)),
        linewidth=1.2,
        label=r"Axial position tolerance $|e_{x,P3}| <= 20$ m",
    )
    axis.set_xlabel("Time, t [s]")
    axis.set_ylabel(r"Remaining axial position error to sampled $P_3$, $|e_{x,P3}|$ [m]")
    axis.set_title(
        f"Remaining axial position error to sampled $P_3$ during closing (testing seed {representative_seed})",
        pad=10,
        fontweight="normal",
    )
    axis.grid(True, color="0.82", linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)
    axis.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.12, 0.99),
        fontsize=9,
        borderaxespad=0.0,
    )

    max_time = max(float(result["times"][-1]) for result in selected.values())
    max_terminal_error = max(result["axial_errors"][-1] for result in selected.values())
    if max_terminal_error > position_tolerance:
        zoom = inset_axes(axis, width="34%", height="35%", loc="upper right", borderpad=2.5)
        for key, result in selected.items():
            zoom.plot(result["times"], result["axial_errors"], color=colors[key], linewidth=1.3)
        zoom.axhline(position_tolerance, color="#56B4E9", linestyle=(0, (5, 2)), linewidth=0.9)
        zoom.set_xlim(max(0.0, max_time - 1000.0), max_time + 50.0)
        zoom.set_ylim(0.0, max(position_tolerance * 1.4, max_terminal_error * 1.2))
        zoom.set_title("Terminal region", fontsize=8, pad=2)
        zoom.tick_params(labelsize=7, length=2)
        zoom.grid(True, color="0.85", linewidth=0.5)

    figure.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.12)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "trajectories").mkdir(parents=True, exist_ok=True)
    results_by_controller = {}
    raw_rows = []
    aggregate_rows = []
    sampled_conditions = {}

    loaded = {}
    for key, spec in CONTROLLERS.items():
        loaded[key] = load_controller(spec)

    try:
        for seed in SEEDS:
            reference_env = make_env(Docking3DoFGymEnv, seed=seed)
            _, reset_info = reference_env.reset(seed=seed)
            initial_state = np.asarray(reference_env.core_env.state, dtype=float).copy()
            p3_target = np.asarray(reference_env.core_env.p3_target, dtype=float).copy()
            reference_env.close()
            sampled_conditions[seed] = {"initial_state": initial_state, "p3_target": p3_target}

            for key, spec in CONTROLLERS.items():
                model, vecnorm, _ = loaded[key]
                result = rollout(
                    model,
                    vecnorm,
                    spec["env_class"],
                    seed,
                    initial_state,
                    p3_target,
                )
                results_by_controller.setdefault(key, []).append(result)
                raw_rows.append(scalar_row(key, result))
                save_trajectory(
                    OUTPUT_DIR / "trajectories" / f"{key}_seed{seed}.npz",
                    result,
                    p3_target,
                    initial_state,
                )
    finally:
        for _, _, dummy_env in loaded.values():
            dummy_env.close()

    write_csv(OUTPUT_DIR / "per_seed_results.csv", raw_rows)
    for key, results in results_by_controller.items():
        aggregate_rows.append(aggregate(key, results))
    aggregate_csv_rows = []
    for row in aggregate_rows:
        output = dict(row)
        output["termination_reasons"] = json.dumps(row["termination_reasons"], sort_keys=True)
        aggregate_csv_rows.append(output)
    write_csv(OUTPUT_DIR / "aggregate_results.csv", aggregate_csv_rows)

    representative_seed, adopted_median = choose_representative_seed(results_by_controller)
    representative_rows = []
    for key, results in results_by_controller.items():
        result = next(r for r in results if r["seed"] == representative_seed)
        condition = sampled_conditions[representative_seed]
        save_trajectory(OUTPUT_DIR / f"representative_{key}_seed{representative_seed}.npz", result, condition["p3_target"], condition["initial_state"])
        for time, axial_error in zip(result["times"], result["axial_errors"]):
            representative_rows.append({"controller": key, "seed": representative_seed, "time_s": time, "remaining_axial_position_error_m": axial_error})
    write_csv(OUTPUT_DIR / "representative_axial_position_error.csv", representative_rows)
    representative = {
        "seed": representative_seed,
        "selection_rule": "Adopted-controller final P3 position error nearest its 10-seed median.",
        "adopted_median_final_position_error_m": adopted_median,
    }
    (OUTPUT_DIR / "representative_selection.json").write_text(json.dumps(representative, indent=2), encoding="utf-8")
    plot_position_error(results_by_controller, representative_seed, OUTPUT_DIR / "closing_position_error_comparison.png")

    summary = {
        "models": {key: {"model": str(spec["model"]), "vecnormalize": str(spec["vecnorm"])} for key, spec in CONTROLLERS.items()},
        "testing_seeds": list(SEEDS),
        "aggregate": aggregate_rows,
        "representative": representative,
        "outputs": {
            "per_seed": str(OUTPUT_DIR / "per_seed_results.csv"),
            "aggregate": str(OUTPUT_DIR / "aggregate_results.csv"),
            "representative_data": str(OUTPUT_DIR / "representative_axial_position_error.csv"),
            "figure_png": str(OUTPUT_DIR / "closing_position_error_comparison.png"),
        },
    }
    (OUTPUT_DIR / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
