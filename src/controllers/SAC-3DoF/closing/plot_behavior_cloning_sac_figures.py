"""Generate the four thesis figures for the behaviour-cloning SAC run."""

from __future__ import annotations

import bisect
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# -----------------------------------------------------------------------------
# Paths and column/tag names: keep these together so the script is easy to edit.
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT = "sac_closing_rewardv15_lqr_bc_warmstart_expanded200_600000steps"
MODEL_DIR = SCRIPT_DIR / "models" / "lqr_bc_warmstart" / EXPERIMENT / "seed_0"
RESULT_DIR = SCRIPT_DIR / "results" / "lqr_bc_warmstart" / EXPERIMENT / "seed_0"
LOG_DIR = SCRIPT_DIR / "logs" / "lqr_bc_warmstart" / EXPERIMENT
OUTPUT_DIR = RESULT_DIR / "publication_figures"

LEARNING_CURVE_FILE = RESULT_DIR / "learning_curve.csv"
EPISODE_COLUMN = "episode"
EPISODE_STEP_COLUMN = "timestep"
EPISODE_RETURN_COLUMN = "episode_return"
ACTOR_LOSS_TAG = "train/actor_loss"
CRITIC_LOSS_TAG = "train/critic_loss"

CHECKPOINTS = (50_000, 100_000)
REPRESENTATIVE_SEED = 1
MAX_ROLLOUT_STEPS = 12_000
MOVING_AVERAGE_WINDOW = 10
CONE_SLOPE = 0.1
CORRIDOR_HALF_WIDTH_M = 5.0
DPI = 400

COLORS = {
    "blue": "#0072B2",
    "orange": "#E66100",
    "green": "#009E73",
    "purple": "#9C6ADE",
    "grey": "#6E6E6E",
}

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_sac_closing_lqr_replay import make_closing_env  # noqa: E402


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "axes.linewidth": 0.75,
        "legend.fontsize": 7.7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def require_columns(fieldnames, required, path):
    missing = [name for name in required if name not in (fieldnames or [])]
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in {path}; "
            f"available columns are {fieldnames}."
        )


def load_learning_curve():
    with LEARNING_CURVE_FILE.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        require_columns(
            reader.fieldnames,
            [EPISODE_COLUMN, EPISODE_STEP_COLUMN, EPISODE_RETURN_COLUMN],
            LEARNING_CURVE_FILE,
        )
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {LEARNING_CURVE_FILE}.")
    episodes = np.asarray([int(row[EPISODE_COLUMN]) for row in rows])
    steps = np.asarray([int(row[EPISODE_STEP_COLUMN]) for row in rows])
    returns = np.asarray([float(row[EPISODE_RETURN_COLUMN]) for row in rows])
    return episodes, steps, returns


def moving_average(x, y, window=MOVING_AVERAGE_WINDOW):
    if len(y) < window:
        return np.asarray([]), np.asarray([])
    kernel = np.ones(window, dtype=float) / window
    return x[window - 1 :], np.convolve(y, kernel, mode="valid")


def step_to_episode(step, episode_steps, episodes):
    index = int(np.searchsorted(episode_steps, step, side="left"))
    index = min(max(index, 0), len(episodes) - 1)
    return float(episodes[index])


def style_axis(axis):
    axis.grid(True, color="#D8D8D8", linewidth=0.55, alpha=0.75)


def add_checkpoint_lines(axis, episode_steps, episodes):
    labels = ("50k: actor updates begin", "100k checkpoint")
    colors = (COLORS["purple"], COLORS["orange"])
    for step, label, color in zip(CHECKPOINTS, labels, colors):
        episode = step_to_episode(step, episode_steps, episodes)
        axis.axvline(
            episode,
            color=color,
            linestyle=":",
            linewidth=1.6,
            label=label,
            zorder=2,
        )


def save_figure(figure, stem, layout_rect=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=layout_rect)
    figure.savefig(OUTPUT_DIR / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def checkpoint_paths(step):
    stem = f"model_{step:07d}_steps"
    model_path = MODEL_DIR / f"{stem}.zip"
    vec_path = MODEL_DIR / f"{stem}_vecnormalize.pkl"
    for path in (model_path, vec_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return model_path, vec_path


def rollout(step, seed=REPRESENTATIVE_SEED):
    """Run one deterministic policy and retain its P3-relative trajectory."""
    model_path, vec_path = checkpoint_paths(step)
    dummy = DummyVecEnv(
        [lambda: make_closing_env(mode="testing", seed=seed, reward_version="v15", monitor=False)]
    )
    vec = VecNormalize.load(str(vec_path), dummy)
    vec.training = False
    vec.norm_reward = False
    model = SAC.load(str(model_path), device="cpu")
    env = make_closing_env(mode="testing", seed=seed, reward_version="v15", monitor=False)
    try:
        observation, _ = env.reset(seed=seed)
        target = env.core_env.p3_target.copy()
        errors = [env.core_env.state[:2] - target[:2]]
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < MAX_ROLLOUT_STEPS:
            normalized = vec.normalize_obs(
                np.asarray(observation, dtype=np.float32).reshape(1, -1)
            )
            action, _ = model.predict(normalized, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action[0])
            errors.append(env.core_env.state[:2] - target[:2])
            steps += 1
        return np.asarray(errors)
    finally:
        env.close()
        vec.close()


def plot_trajectories():
    trajectory_50k = rollout(50_000)
    trajectory_600k = rollout(600_000)
    figure, axis = plt.subplots(figsize=(8.0, 4.5))

    all_x = np.concatenate((trajectory_50k[:, 0], trajectory_600k[:, 0], [0.0]))
    x_boundary = np.linspace(min(0.0, all_x.min()), max(0.0, all_x.max()), 600)
    cone = CONE_SLOPE * np.abs(x_boundary)
    axis.axhline(0, color="#555555", linewidth=0.7, zorder=0)
    axis.plot(x_boundary, cone, color="#FF9900", linestyle=":", linewidth=1.35, label="Docking cone")
    axis.plot(x_boundary, -cone, color="#FF9900", linestyle=":", linewidth=1.35)
    axis.axhline(CORRIDOR_HALF_WIDTH_M, color=COLORS["purple"], linestyle=":", linewidth=1.35, label="Docking corridor")
    axis.axhline(-CORRIDOR_HALF_WIDTH_M, color=COLORS["purple"], linestyle=":", linewidth=1.35)

    for data, color, label in (
        (trajectory_50k, COLORS["blue"], "50k checkpoint"),
        (trajectory_600k, COLORS["orange"], "600k checkpoint"),
    ):
        axis.plot(data[:, 0], data[:, 1], color=color, linewidth=1.8, label=label)
        axis.scatter(data[0, 0], data[0, 1], s=45, marker="D", facecolor="white", edgecolor=color, linewidth=1.4, zorder=5)
        axis.scatter(data[-1, 0], data[-1, 1], s=48, marker="X", color=color, zorder=5)

    axis.scatter(0, 0, s=82, marker="*", color=COLORS["green"], edgecolor="white", linewidth=0.5, label="$P_3$ target", zorder=6)
    # Proxy markers give the legend the same compact marker vocabulary as the
    # reference thesis figure without duplicating labels for both policies.
    axis.scatter([], [], s=38, marker="D", facecolor="white", edgecolor=COLORS["grey"], label="Initial state")
    axis.scatter([], [], s=40, marker="X", color=COLORS["grey"], label="Terminal state")
    axis.set_xlabel("Axial position error relative to $P_3$ [m]")
    axis.set_ylabel("Lateral position error relative to $P_3$ [m]")
    axis.set_title(
        f"Representative behavior-cloning SAC trajectories "
        f"(testing seed {REPRESENTATIVE_SEED})",
        pad=62,
    )
    # Independent axis scaling keeps the narrow terminal corridor legible while
    # still showing the complete 1.5 km closing trajectory.
    axis.invert_xaxis()
    style_axis(axis)
    handles, labels = axis.get_legend_handles_labels()
    order = [2, 3, 5, 6, 4, 1, 0]
    axis.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        ncol=4,
        columnspacing=1.5,
        handlelength=2.1,
    )
    save_figure(figure, "behavior_cloning_sac_trajectories")


def plot_learning_curve():
    episodes, steps, returns = load_learning_curve()
    # Online learning starts with the first recorded online episode.
    smooth_x, smooth_y = moving_average(episodes, returns)
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    axis.plot(episodes, returns, color="#8EC9ED", alpha=0.55, linewidth=0.7, label="Individual episode")
    axis.plot(smooth_x, smooth_y, color="#005B96", linewidth=1.7, label="10-episode moving average")
    add_checkpoint_lines(axis, steps, episodes)
    axis.set_xlabel("Episode number")
    axis.set_ylabel("Episode return [-]")
    axis.set_title("Behavior-cloning SAC learning curve")
    style_axis(axis)
    axis.legend(loc="lower right", frameon=False)
    save_figure(figure, "behavior_cloning_sac_learning_curve")


def find_event_file():
    files = sorted(LOG_DIR.rglob("events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event file found below {LOG_DIR}.")
    return files[-1]


def tensorboard_scalars(tag):
    accumulator = EventAccumulator(str(find_event_file()))
    accumulator.Reload()
    available = accumulator.Tags().get("scalars", [])
    if tag not in available:
        raise ValueError(f"Required TensorBoard tag {tag!r} is absent; available tags: {available}")
    events = accumulator.Scalars(tag)
    return np.asarray([event.step for event in events]), np.asarray([event.value for event in events])


def aggregate_loss_by_episode(loss_steps, losses, episode_steps, episodes):
    grouped = defaultdict(list)
    for step, loss in zip(loss_steps, losses):
        index = bisect.bisect_left(episode_steps.tolist(), int(step))
        if index < len(episodes):
            grouped[int(episodes[index])].append(float(loss))
    x = np.asarray(sorted(grouped))
    y = np.asarray([np.mean(grouped[episode]) for episode in x])
    if len(x) == 0:
        raise ValueError("No logged losses could be mapped to an episode.")
    return x, y


def draw_loss_axis(axis, tag, title, ylabel, color, episodes, episode_steps):
    loss_steps, losses = tensorboard_scalars(tag)
    x, y = aggregate_loss_by_episode(loss_steps, losses, episode_steps, episodes)
    smooth_x, smooth_y = moving_average(x, y)
    axis.plot(x, y, color=color, alpha=0.48, linewidth=0.7, label="Per-episode mean")
    if len(smooth_y):
        axis.plot(smooth_x, smooth_y, color=color, linewidth=1.55, label="10-episode moving average")
    add_checkpoint_lines(axis, episode_steps, episodes)
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontsize=8.5, pad=4)
    style_axis(axis)


def plot_training_losses():
    episodes, episode_steps, _ = load_learning_curve()
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.25), sharex=True)
    draw_loss_axis(axes[0], ACTOR_LOSS_TAG, "Actor loss", "Actor loss [-]", COLORS["blue"], episodes, episode_steps)
    draw_loss_axis(axes[1], CRITIC_LOSS_TAG, "Critic loss", "Critic loss [-]", COLORS["blue"], episodes, episode_steps)
    figure.suptitle("Behavior-cloning SAC training losses", fontsize=9.5, y=0.995)
    handles, labels = axes[1].get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        frameon=False,
        ncol=4,
        fontsize=7,
        columnspacing=1.2,
    )
    legend.set_in_layout(False)
    figure.supxlabel("Episode number", fontsize=9, y=0.015)
    # Keep the panels at their original size, reserving only a narrow gap
    # between the shared legend and the two subplot titles.
    save_figure(
        figure,
        "behavior_cloning_sac_training_losses",
        layout_rect=(0.0, 0.04, 1.0, 0.95),
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_trajectories()
    plot_learning_curve()
    plot_training_losses()
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
