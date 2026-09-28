"""Plot local curriculum behavior against the normal full-P2 closing task."""

import contextlib
import io
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

SCRIPT_DIR = Path(__file__).resolve().parent
FAMILY_DIR = SCRIPT_DIR.parent
CLOSING_DIR = FAMILY_DIR.parent
VANILLA_DIR = CLOSING_DIR / "vanilla_sac"
for path in (FAMILY_DIR, CLOSING_DIR, VANILLA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_sac_closing_curriculum import (  # noqa: E402
    DEFAULT_CLOSING_CURRICULUM_STAGES,
    Docking3DoFGymEnv,
    make_curriculum_env,
)
from tune_sac_closing_optuna import normalize_observation, target_error  # noqa: E402


EXPERIMENT = "curriculum_v20_seed0"
MODEL_DIR = FAMILY_DIR / "models" / "curriculum" / EXPERIMENT
MODEL_PATH = MODEL_DIR / "stage_00_best_model.zip"
VECNORM_PATH = MODEL_DIR / "stage_00_best_vecnormalize.pkl"
OUTPUT_DIR = FAMILY_DIR / "results" / "curriculum" / EXPERIMENT / "figures"
PLOT_PATH = OUTPUT_DIR / "curriculum_v20_local_vs_p2_trajectory.png"
SUMMARY_PATH = OUTPUT_DIR / "curriculum_v20_local_vs_p2_trajectory_summary.txt"

NEAR_P3_SEEDS = tuple(range(1000, 1020))
P2_SEEDS = tuple(range(10))
NEAR_P3_STAGE = DEFAULT_CLOSING_CURRICULUM_STAGES[0]


def make_normal_p2_env(seed):
    return Docking3DoFGymEnv(
        phase="closing",
        mode="testing",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
        closing_reward_version="v20",
    )


def load_policy():
    dummy = DummyVecEnv([lambda: Monitor(make_normal_p2_env(999))])
    vecnorm = VecNormalize.load(str(VECNORM_PATH), dummy)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return SAC.load(str(MODEL_PATH)), vecnorm, dummy


def rollout(model, vecnorm, initialization, seed):
    if initialization == "near-P3 success_acquisition":
        env = make_curriculum_env(
            NEAR_P3_STAGE,
            seed=seed,
            reward_version="v20",
        )
        max_steps = 2000
    else:
        env = make_normal_p2_env(seed)
        max_steps = 12000

    with contextlib.redirect_stdout(io.StringIO()):
        observation, reset_info = env.reset(seed=seed)
        p3_target = env.core_env.p3_target.copy()
        final_info = {}
        for step in range(max_steps):
            normalized = normalize_observation(vecnorm, observation)
            action, _ = model.predict(normalized, deterministic=True)
            observation, _, terminated, truncated, final_info = env.step(action[0])
            if terminated or truncated:
                break
        states = np.asarray(env.get_history_arrays()[1], dtype=np.float64)
        final_error = target_error(env)
        tolerance = env.core_env.p3_target_tolerance.copy()
        env.close()

    errors = states[:, :2] - p3_target[:2]
    distances = np.linalg.norm(errors, axis=1)
    reason = final_info.get("termination_reason", "unknown")
    return {
        "initialization": initialization,
        "seed": seed,
        "steps": step + 1,
        "success": bool(final_info.get("success", False)),
        "failure": bool(final_info.get("failure", False)),
        "timeout": bool(final_info.get("time_limit", False)),
        "termination_reason": reason,
        "errors": errors,
        "closest_position_error": float(np.min(distances)),
        "final_position_error": float(distances[-1]),
        "final_max_target_ratio": float(
            np.max(np.abs(final_error) / np.maximum(tolerance, 1e-12))
        ),
        "curriculum_stage": reset_info.get("curriculum_stage"),
    }


def select_near_p3(results):
    successful = [result for result in results if result["success"]]
    if successful:
        # Prefer a nontrivial but clean successful local regulation trajectory.
        return max(successful, key=lambda result: result["steps"])
    return min(results, key=lambda result: result["final_max_target_ratio"])


def select_p2(results):
    failed = [result for result in results if not result["success"]]
    candidates = failed or results
    closest_median = np.median(
        [result["closest_position_error"] for result in candidates]
    )
    final_median = np.median(
        [result["final_position_error"] for result in candidates]
    )
    return min(
        candidates,
        key=lambda result: (
            abs(result["closest_position_error"] - closest_median)
            / max(closest_median, 1.0)
            + abs(result["final_position_error"] - final_median)
            / max(final_median, 1.0)
        ),
    )


def draw_panel(axis, result, panel_title):
    errors = result["errors"]
    axis.plot(errors[:, 0], errors[:, 1], linewidth=1.8, label="Trajectory")
    axis.scatter(
        errors[0, 0], errors[0, 1], s=65, marker="o", label="Start", zorder=3
    )
    axis.scatter(
        errors[-1, 0], errors[-1, 1], s=75, marker="x", label="End", zorder=3
    )
    axis.scatter(0.0, 0.0, s=65, marker="+", label="$P_3$ target", zorder=4)
    axis.set_title(panel_title, pad=9)
    axis.set_xlabel(r"Axial $P_3$-relative position error [m]", labelpad=5)
    axis.grid(True, alpha=0.3)
    axis.margins(x=0.08, y=0.15)
    axis.invert_xaxis()


def format_summary(result):
    outcome = (
        "success" if result["success"] else "unsafe"
        if result["failure"] else "timeout" if result["timeout"] else "other"
    )
    return (
        f"initialization: {result['initialization']}\n"
        f"evaluation seed: {result['seed']}\n"
        f"outcome: {outcome}\n"
        f"termination reason: {result['termination_reason']}\n"
        f"closest P3-position error [m]: "
        f"{result['closest_position_error']:.6f}\n"
        f"final P3-position error [m]: {result['final_position_error']:.6f}\n"
        f"final maximum target-tolerance ratio: "
        f"{result['final_max_target_ratio']:.6f}\n"
    )


def main():
    for path in (MODEL_PATH, VECNORM_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    model, vecnorm, dummy = load_policy()
    near_results = [
        rollout(model, vecnorm, "near-P3 success_acquisition", seed)
        for seed in NEAR_P3_SEEDS
    ]
    p2_results = [
        rollout(model, vecnorm, "normal P2 testing", seed) for seed in P2_SEEDS
    ]
    dummy.close()

    near = select_near_p3(near_results)
    p2 = select_p2(p2_results)

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    draw_panel(axes[0], near, r"(a) Near-$P_3$ initialization")
    draw_panel(axes[1], p2, r"(b) $P_2$ initialization")
    axes[0].set_ylabel(r"Lateral $P_3$-relative position error [m]")
    axes[1].set_ylabel(r"Lateral $P_3$-relative position error [m]")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        r"Curriculum policy behavior: local near-$P_3$ success versus "
        r"failure to transfer from $P_2$",
        y=0.97,
    )
    figure.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.23, wspace=0.24)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(PLOT_PATH, dpi=300)
    plt.close(figure)

    summary = (
        "Curriculum dead-end trajectory comparison\n"
        "These are deterministic rollouts of the retained stage-00 policy; "
        "they are not the historical 275k or 425k checkpoint trajectories.\n\n"
        "PANEL A\n"
        + format_summary(near)
        + "\nPANEL B\n"
        + format_summary(p2)
    )
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"figure: {PLOT_PATH}")
    print(f"summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
