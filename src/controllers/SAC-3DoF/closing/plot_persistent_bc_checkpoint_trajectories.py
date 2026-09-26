"""Compare persistent-BC trajectories at the 50k and 600k checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from train_sac_closing_lqr_replay import make_closing_env


EXPERIMENT = "sac_closing_rewardv15_lqr_bc_constrained_600000steps"
MODEL_DIR = HERE / "models" / "lqr_bc_constrained" / EXPERIMENT / "seed_0"
OUTPUT_DIR = (
    HERE / "results" / "lqr_bc_constrained" / EXPERIMENT / "seed_0"
    / "publication_figures"
)
OUTPUT_STEM = OUTPUT_DIR / "persistent_bc_50k_vs_600k_trajectories"

CHECKPOINTS = (50_000, 600_000)
TESTING_SEED = 1
MAX_STEPS = 12_000
CONE_SLOPE = 0.1
CORRIDOR_HALF_WIDTH_M = 5.0
DPI = 400

BLUE = "#0072B2"
ORANGE = "#E66100"
GREEN = "#009E73"
PURPLE = "#9C6ADE"
GREY = "#555555"


def checkpoint_paths(step):
    stem = f"model_{step:07d}_steps"
    model_path = MODEL_DIR / f"{stem}.zip"
    vec_path = MODEL_DIR / f"{stem}_vecnormalize.pkl"
    for path in (model_path, vec_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return model_path, vec_path


def rollout(step):
    """Run one deterministic testing rollout and return P3-relative position."""
    model_path, vec_path = checkpoint_paths(step)
    dummy = DummyVecEnv([
        lambda: make_closing_env(
            mode="testing", seed=TESTING_SEED,
            reward_version="v15", monitor=False,
        )
    ])
    vec = VecNormalize.load(str(vec_path), dummy)
    vec.training = False
    vec.norm_reward = False
    model = SAC.load(str(model_path), device="cpu")
    env = make_closing_env(
        mode="testing", seed=TESTING_SEED,
        reward_version="v15", monitor=False,
    )
    try:
        observation, _ = env.reset(seed=TESTING_SEED)
        target = env.core_env.p3_target.copy()
        errors = [env.core_env.state[:2] - target[:2]]
        final_info = {}
        for _ in range(MAX_STEPS):
            normalized = vec.normalize_obs(
                np.asarray(observation, dtype=np.float32).reshape(1, -1)
            )
            action, _ = model.predict(normalized, deterministic=True)
            observation, _, terminated, truncated, final_info = env.step(action[0])
            errors.append(env.core_env.state[:2] - target[:2])
            if terminated or truncated:
                break
        return np.asarray(errors), final_info
    finally:
        env.close()
        vec.close()


def style_axis(axis):
    axis.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.75)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.75)
    axis.tick_params(width=0.7, length=3)


def main():
    trajectories = []
    outcomes = []
    for checkpoint in CHECKPOINTS:
        trajectory, info = rollout(checkpoint)
        trajectories.append(trajectory)
        outcomes.append(info.get("termination_reason", "max_steps"))

    all_x = np.concatenate([data[:, 0] for data in trajectories] + [np.array([0.0])])
    all_y = np.concatenate([data[:, 1] for data in trajectories] + [np.array([0.0])])
    x_data_min, x_data_max = float(all_x.min()), float(all_x.max())
    x_span = max(x_data_max - x_data_min, 1.0)
    x_left = x_data_max + 0.035 * x_span
    x_right = min(-25.0, x_data_min - 0.035 * x_span)
    cone_extent = CONE_SLOPE * max(x_left, 0.0)
    y_extent = 1.08 * max(abs(float(all_y.min())), abs(float(all_y.max())), cone_extent)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "axes.linewidth": 0.75,
        "legend.fontsize": 7.7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axis = plt.subplots(figsize=(8.0, 4.5))

    cone_x = np.linspace(0.0, x_left, 700)
    axis.plot(cone_x, CONE_SLOPE * cone_x, color="#FF9900", ls=":", lw=1.25)
    axis.plot(cone_x, -CONE_SLOPE * cone_x, color="#FF9900", ls=":", lw=1.25)
    # The final corridor begins at P3 and extends into the post-P3 region.
    axis.hlines((CORRIDOR_HALF_WIDTH_M, -CORRIDOR_HALF_WIDTH_M),
                x_right, 0.0, colors=PURPLE, linestyles=":", linewidth=1.25)
    axis.axhline(0.0, color=GREY, ls="-.", lw=0.75)

    for trajectory, color in zip(trajectories, (BLUE, ORANGE)):
        axis.plot(trajectory[:, 0], trajectory[:, 1], color=color, lw=1.65)
        axis.scatter(*trajectory[0], marker="D", s=38, facecolor="white",
                     edgecolor=color, linewidth=1.25, zorder=5)
        axis.scatter(*trajectory[-1], marker="X", s=48, color=color, zorder=6)
    axis.scatter(0.0, 0.0, marker="*", s=78, color=GREEN,
                 edgecolor="white", linewidth=0.45, zorder=7)
    axis.set_xlim(x_left, x_right)
    axis.set_ylim(-y_extent, y_extent)
    axis.set_xlabel(r"Axial position error relative to $P_3$ [m]")
    axis.set_ylabel(r"Lateral position error relative to $P_3$ [m]")
    style_axis(axis)
    axis.set_title(
        f"Representative persistent behavior-cloning SAC trajectories "
        f"(testing seed {TESTING_SEED})",
        pad=60, fontweight="normal", fontsize=10,
    )
    handles = [
        Line2D([], [], color=BLUE, lw=1.65, label="50k checkpoint"),
        Line2D([], [], color=ORANGE, lw=1.65, label="600k checkpoint"),
        Line2D([], [], marker="D", ls="none", mfc="white", mec=GREY,
               label="Initial state"),
        Line2D([], [], marker="X", ls="none", color=GREY,
               label="Terminal state"),
        Line2D([], [], marker="*", ls="none", color=GREEN, label=r"$P_3$ target"),
        Line2D([], [], color="#FF9900", ls=":", label="Docking cone"),
        Line2D([], [], color=PURPLE, ls=":", label="Docking corridor"),
    ]
    axis.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                ncol=4, frameon=False, handlelength=2.0, columnspacing=1.3)
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.14, top=0.73)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=DPI,
                   bbox_inches="tight", facecolor="white")
    plt.close(figure)

    for checkpoint, trajectory, outcome in zip(CHECKPOINTS, trajectories, outcomes):
        print(f"{checkpoint:,} steps: {outcome}; terminal position error "
              f"{np.linalg.norm(trajectory[-1]):.3f} m")
    print(f"PNG: {OUTPUT_STEM.with_suffix('.png').resolve()}")


if __name__ == "__main__":
    main()
