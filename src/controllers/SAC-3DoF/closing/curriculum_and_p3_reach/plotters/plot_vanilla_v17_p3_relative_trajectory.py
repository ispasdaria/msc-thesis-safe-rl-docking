"""Evaluate one closing SAC rollout in sampled-P3 position-error coordinates."""

import argparse
import contextlib
import io
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FAMILY_DIR = os.path.dirname(SCRIPT_DIR)
CLOSING_DIR = os.path.dirname(FAMILY_DIR)
VANILLA_DIR = os.path.join(CLOSING_DIR, "vanilla_sac")
SRC_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..")
)

for path in (VANILLA_DIR, CLOSING_DIR, SRC_PATH):
    if path not in sys.path:
        sys.path.insert(0, path)

import evaluate_sac_closing as evaluator


MODEL_NAME = "sac_closing_p3augobs_rewardv17_300ksteps_seed0_buffer300k"
ARTIFACT_GROUP = "reward_shaping_trials"
EVALUATION_SEED = 1

MODEL_DIR = os.path.join(VANILLA_DIR, "models", ARTIFACT_GROUP)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME + ".zip")
VECNORM_PATH = os.path.join(MODEL_DIR, MODEL_NAME + "_vecnormalize.pkl")
RESULTS_DIR = os.path.join(
    VANILLA_DIR,
    "results",
    ARTIFACT_GROUP,
    "p3_relative_trajectory",
    MODEL_NAME,
)
PLOT_PATH = os.path.join(
    RESULTS_DIR,
    f"{MODEL_NAME}_p3_relative_trajectory_seed{EVALUATION_SEED}.png",
)


def evaluate_rollout():
    """Run deterministic evaluation and retain the sampled P3 target."""

    evaluator.MODEL_NAME = MODEL_NAME
    evaluator.MODEL_DIR = MODEL_DIR
    evaluator.MODEL_PATH = MODEL_PATH
    evaluator.VECNORM_PATH = VECNORM_PATH

    with contextlib.redirect_stdout(io.StringIO()):
        vecnorm, dummy_env = evaluator.load_vecnormalize_stats()
        model = evaluator.SAC.load(MODEL_PATH)
        env = evaluator.Docking3DoFGymEnv(
            phase="closing",
            mode="testing",
            dt=1.0,
            max_time=None,
            integrator="rk4",
            normalize_action=True,
            seed=EVALUATION_SEED,
            closing_reward_version="v17",
        )
        obs, _ = env.reset(seed=EVALUATION_SEED)

        p3_target = env.core_env.p3_target.copy()
        final_info = {}
        for step in range(12000):
            normalized_obs = evaluator.normalize_observation(vecnorm, obs)
            action, _ = model.predict(normalized_obs, deterministic=True)
            obs, _, terminated, truncated, final_info = env.step(action[0])
            if terminated or truncated:
                break

        history = env.get_history_arrays()
        times = np.asarray(history[0], dtype=float)
        states = np.asarray(history[1], dtype=float)
        tolerance = env.core_env.p3_target_tolerance.copy()
        env.close()
        dummy_env.close()

    position_errors = states[:, :2] - p3_target[:2]
    terminal_error = states[-1] - p3_target
    terminal_error[4] = np.arctan2(
        np.sin(terminal_error[4]), np.cos(terminal_error[4])
    )
    worst_ratio = float(
        np.max(np.abs(terminal_error) / np.maximum(tolerance, 1e-12))
    )
    return times, position_errors, final_info, p3_target, step + 1, worst_ratio


def plot_trajectory(position_errors, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(
        position_errors[:, 0],
        position_errors[:, 1],
        linewidth=1.8,
        label="Trajectory",
    )
    axis.scatter(
        position_errors[0, 0],
        position_errors[0, 1],
        s=65,
        marker="o",
        label="Start",
        zorder=3,
    )
    axis.scatter(
        position_errors[-1, 0],
        position_errors[-1, 1],
        s=75,
        marker="x",
        label="End",
        zorder=3,
    )
    axis.scatter(
        0.0,
        0.0,
        s=65,
        marker="+",
        label="$P_3$ target",
        zorder=4,
    )
    axis.set_xlabel(
        r"Axial $P_3$-relative position error, $e_{r_{p,x}}$ [m]",
        labelpad=3,
    )
    axis.set_ylabel(
        "Lateral $P_3$-relative position\nerror, $e_{r_{p,y}}$ [m]"
    )
    axis.set_title(
        "Representative trajectory of the closing policy trained with reward version 17\n"
        f"(testing seed {EVALUATION_SEED})",
        pad=12,
    )
    axis.margins(x=0.06, y=0.18)
    axis.grid(True, alpha=0.3)
    axis.invert_xaxis()
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=False,
    )
    figure.subplots_adjust(left=0.14, right=0.97, top=0.80, bottom=0.21)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=PLOT_PATH)
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    if not os.path.exists(VECNORM_PATH):
        raise FileNotFoundError(VECNORM_PATH)

    times, errors, final_info, p3_target, steps, worst_ratio = evaluate_rollout()
    distances = np.linalg.norm(errors, axis=1)
    closest_index = int(np.argmin(distances))
    minimum_distance = float(distances[closest_index])
    final_distance = float(distances[-1])
    error_increased_after_closest = bool(np.any(distances[closest_index:] > minimum_distance))

    plot_trajectory(errors, args.output)

    print(f"plot={args.output}")
    print(f"steps={steps}")
    print(f"p3_target={p3_target.tolist()}")
    print(f"minimum_position_error={minimum_distance:.9f}")
    print(f"minimum_position_error_time={times[closest_index]:.9f}")
    print(
        "closest_error=("
        f"{errors[closest_index, 0]:.9f}, {errors[closest_index, 1]:.9f})"
    )
    print(f"final_position_error={final_distance:.9f}")
    print(f"final_error=({errors[-1, 0]:.9f}, {errors[-1, 1]:.9f})")
    print(
        "worst_terminal_tolerance_ratio="
        f"{worst_ratio:.9f}"
    )
    print(f"error_increased_after_closest={error_increased_after_closest}")
    print(f"termination_reason={final_info.get('termination_reason')}")


if __name__ == "__main__":
    main()
