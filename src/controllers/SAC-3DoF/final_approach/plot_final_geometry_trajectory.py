"""Evaluate one final-approach SAC rollout and plot docking geometry."""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import evaluate_sac_final as evaluator
from envs.docking3dof.gym_env import Docking3DoFGymEnv


MODEL_NAME = "sac_final_approach_portobs_rewardv4_100ksteps_seed0"
ARTIFACT_GROUP = "reward_shaping_trials"
EVALUATION_SEED = 0

MODEL_DIR = os.path.join(SCRIPT_DIR, "models", ARTIFACT_GROUP)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME + ".zip")
VECNORM_PATH = os.path.join(MODEL_DIR, MODEL_NAME + "_vecnormalize.pkl")
RESULTS_DIR = os.path.join(
    SCRIPT_DIR,
    "results",
    ARTIFACT_GROUP,
    "geometry_trajectory",
    MODEL_NAME,
)
ROLLOUT_PATH = os.path.join(RESULTS_DIR, "rollout_seed_0.npz")
PLOT_PATH = os.path.join(
    RESULTS_DIR,
    f"{MODEL_NAME}_geometry_trajectory_seed{EVALUATION_SEED}.png",
)


def port_relative_position(states):
    """Use the repository's reduced 2D port geometry for logged states."""

    psi = states[:, 4]
    r_p_x = states[:, 0] - 4.1 * np.cos(psi) - 20.0
    r_p_y = states[:, 1] - 4.1 * np.sin(psi)
    return r_p_x, r_p_y


def get_geometry(seed):
    """Read cone and corridor parameters after the repository's reset."""

    env = Docking3DoFGymEnv(
        phase="final",
        mode="testing",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
    )
    env.reset(seed=seed)
    geometry = {
        "cone_slope": float(env.core_env.cone_slope),
        "corridor_radius": float(env.core_env.corridor_radius),
        "d_P3": float(env.core_env.d_P3),
    }
    env.close()
    return geometry


def evaluate_rollout():
    """Run deterministic evaluation using the existing evaluator workflow."""

    evaluator.MODEL_NAME = MODEL_NAME
    evaluator.MODEL_PATH = MODEL_PATH
    evaluator.VECNORM_PATH = VECNORM_PATH
    evaluator.RESULTS_DIR = RESULTS_DIR

    vecnorm, dummy_env = evaluator.load_vecnormalize_stats()
    model = evaluator.SAC.load(MODEL_PATH)
    result = evaluator.run_one_rollout(
        model=model,
        vecnorm=vecnorm,
        seed=EVALUATION_SEED,
        mode="testing",
        max_steps=5000,
    )
    dummy_env.close()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(
        ROLLOUT_PATH,
        seed=result["seed"],
        steps=result["steps"],
        success=result["success"],
        failure=result["failure"],
        termination_reason=str(result["termination_reason"]),
        states=result["states"],
    )
    return result


def plot_trajectory(result, geometry, output_path):
    r_p_x, r_p_y = port_relative_position(result["states"])

    finite_x = r_p_x[np.isfinite(r_p_x)]
    x_max = max(float(np.max(finite_x)), geometry["d_P3"], 0.0)
    x_min = min(float(np.min(finite_x)), 0.0)
    x_margin = max(0.05 * (x_max - x_min), 2.0)
    cone_x = np.linspace(0.0, x_max + x_margin, 500)
    corridor_x = np.linspace(0.0, geometry["d_P3"], 200)

    figure, axis = plt.subplots(figsize=(9, 7))
    axis.plot(r_p_x, r_p_y, linewidth=1.8, label="Policy trajectory")
    axis.scatter(r_p_x[0], r_p_y[0], s=65, marker="o", label="Start", zorder=3)
    axis.scatter(r_p_x[-1], r_p_y[-1], s=75, marker="x", label="End", zorder=3)
    axis.scatter(0.0, 0.0, s=65, marker="+", label="Docking point", zorder=4)

    axis.plot(
        cone_x,
        np.zeros_like(cone_x),
        "k:",
        linewidth=1.2,
        label="Docking axis",
    )
    axis.plot(
        cone_x,
        geometry["cone_slope"] * cone_x,
        ":",
        color="tab:orange",
        linewidth=1.2,
        label="Cone boundary",
    )
    axis.plot(
        cone_x,
        -geometry["cone_slope"] * cone_x,
        ":",
        color="tab:orange",
        linewidth=1.2,
        label="_nolegend_",
    )
    axis.plot(
        corridor_x,
        np.full_like(corridor_x, geometry["corridor_radius"]),
        ":",
        color="tab:purple",
        linewidth=1.2,
        label="Corridor boundary",
    )
    axis.plot(
        corridor_x,
        np.full_like(corridor_x, -geometry["corridor_radius"]),
        ":",
        color="tab:purple",
        linewidth=1.2,
        label="_nolegend_",
    )

    axis.set_xlim(x_min - x_margin, x_max + x_margin)
    axis.invert_xaxis()
    y_values = np.concatenate(
        (
            r_p_y,
            geometry["cone_slope"] * cone_x,
            -geometry["cone_slope"] * cone_x,
            np.array([geometry["corridor_radius"], -geometry["corridor_radius"]]),
        )
    )
    y_margin = max(0.05 * (float(np.max(y_values)) - float(np.min(y_values))), 2.0)
    axis.set_ylim(float(np.min(y_values)) - y_margin, float(np.max(y_values)) + y_margin)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"Axial docking-port-relative position, $r_{p,x}$ [m]")
    axis.set_ylabel("Lateral docking-port-relative\nposition, $r_{p,y}$ [m]")
    axis.set_title("Intermediate reward version (testing seed 0)")
    axis.grid(True, alpha=0.3)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    figure.subplots_adjust(bottom=0.37)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

    return r_p_x, r_p_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=PLOT_PATH)
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    if not os.path.exists(VECNORM_PATH):
        raise FileNotFoundError(VECNORM_PATH)

    result = evaluate_rollout()
    geometry = get_geometry(EVALUATION_SEED)
    r_p_x, r_p_y = plot_trajectory(result, geometry, args.output)

    print(f"plot={args.output}")
    print(f"steps={result['steps']}")
    print(f"success={result['success']}")
    print(f"termination_reason={result['termination_reason']}")
    print(f"violated_constraints={result['violated_constraints']}")
    print(f"d_P3={geometry['d_P3']:.6f}")
    print(f"start_r_p=({r_p_x[0]:.6f}, {r_p_y[0]:.6f})")
    print(f"end_r_p=({r_p_x[-1]:.6f}, {r_p_y[-1]:.6f})")


if __name__ == "__main__":
    main()