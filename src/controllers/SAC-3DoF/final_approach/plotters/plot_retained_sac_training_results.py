"""Plot training metrics for the retained seed-12 final-approach SAC run.

The episode-return figure is generated from the committed learning-curve CSV.
The deterministic success figure is generated only when the original SB3
``evaluations.npz`` exists and contains per-rollout ``successes``. This script
never substitutes stochastic training outcomes for deterministic evaluation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
RUN_NAME = "sac_final_approach_portobs_rewardv9_100ksteps_seed12_optuna12"
RUN_DIR = (
    SCRIPT_DIR / "results" / "retained_policy_evaluations"
    / "optuna12_100ksteps_buffer100k"
    / "learning_curves" / RUN_NAME
)
CSV_PATH = RUN_DIR / "learning_curve.csv"
EVALUATION_PATH = RUN_DIR / "evaluations.npz"
OUTPUT_DIR = (
    SCRIPT_DIR / "results" / "retained_figures"
    / "training_curves" / RUN_NAME
)
LEARNING_STARTS = 1_000

BLUE = "#0072B2"
DARK_BLUE = "#003B5C"


def read_training_curve() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with CSV_PATH.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    episodes = np.asarray([int(row["episode"]) for row in rows])
    timesteps = np.asarray([int(row["timestep"]) for row in rows])
    returns = np.asarray([float(row["episode_return"]) for row in rows])
    return episodes, timesteps, returns


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(values, np.ones(window) / window, mode="valid")


def configure_style() -> None:
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10.5,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_returns(episodes: np.ndarray, returns: np.ndarray) -> None:
    window = 20
    smoothed = moving_average(returns, window)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(episodes, returns, color=BLUE, linewidth=0.8, alpha=0.28,
            label="Individual episode")
    ax.plot(episodes[window - 1 :], smoothed, color=DARK_BLUE, linewidth=2.0,
            label=f"{window}-episode moving average")
    ax.set_xlabel("Episode number")
    ax.set_ylabel("Episode return")
    ax.set_title("Training learning curve for the retained final-approach SAC policy",
                 loc="left", fontweight="bold")
    ax.grid(True, color="#D9D9D9", linewidth=0.65, alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="best")
    ax.margins(x=0.01)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.14, top=0.90)
    save_figure(fig, OUTPUT_DIR / "training_episode_return")


def plot_returns_from_learning_start(
    episodes: np.ndarray, timesteps: np.ndarray, returns: np.ndarray
) -> tuple[int, int]:
    """Plot completed episodes from the first one crossing learning_starts."""
    eligible = np.flatnonzero(timesteps >= LEARNING_STARTS)
    if eligible.size == 0:
        raise ValueError(
            f"No completed episode reaches learning_starts={LEARNING_STARTS}."
        )
    start = int(eligible[0])
    cropped_episodes = episodes[start:]
    cropped_returns = returns[start:]

    window = 10
    smoothed = moving_average(cropped_returns, window)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(
        cropped_episodes,
        cropped_returns,
        color=BLUE,
        linewidth=0.8,
        alpha=0.28,
        label="Individual episode",
    )
    ax.plot(
        cropped_episodes[window - 1 :],
        smoothed,
        color=DARK_BLUE,
        linewidth=2.0,
        label=f"{window}-episode moving average",
    )
    ax.set_xlabel("Episode number")
    ax.set_ylabel("Episode return")
    ax.set_title(
        "Training learning curve for the retained final-approach SAC policy",
        loc="left",
        fontweight="bold",
    )
    ax.grid(True, color="#D9D9D9", linewidth=0.65, alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left")
    ax.margins(x=0.01)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.14, top=0.90)
    save_figure(
        fig,
        OUTPUT_DIR / "training_episode_return_from_learning_start_10_episode_average",
    )
    return int(episodes[start]), int(timesteps[start])


def evaluation_episodes(eval_steps: np.ndarray, episodes: np.ndarray,
                        timesteps: np.ndarray) -> np.ndarray:
    mapped = []
    for step in eval_steps:
        index = int(np.searchsorted(timesteps, step, side="left"))
        # Evaluation can occur while the next training episode is unfinished.
        mapped.append(int(episodes[min(index, len(episodes) - 1)]))
    return np.asarray(mapped)


def plot_success_if_available(episodes: np.ndarray, timesteps: np.ndarray) -> bool:
    if not EVALUATION_PATH.exists():
        return False
    data = np.load(EVALUATION_PATH, allow_pickle=False)
    if "successes" not in data.files:
        return False
    rates = 100.0 * np.asarray(data["successes"], dtype=float).mean(axis=1)
    mapped_episodes = evaluation_episodes(data["timesteps"], episodes, timesteps)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(mapped_episodes, rates, color=DARK_BLUE, marker="o", linewidth=2.0)
    ax.set(xlabel="Episode number",
           ylabel="Deterministic evaluation success rate [%]",
           ylim=(-3.0, 103.0))
    ax.set_title(
        "Deterministic evaluation success rate during training for the retained "
        "final-approach SAC policy", loc="left", fontweight="bold"
    )
    ax.grid(True, color="#D9D9D9", linewidth=0.65, alpha=0.75)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.14, top=0.90)
    save_figure(fig, OUTPUT_DIR / "deterministic_evaluation_success")
    return True


def main() -> None:
    configure_style()
    episodes, timesteps, returns = read_training_curve()
    plot_returns(episodes, returns)
    learning_start_episode, first_logged_timestep = plot_returns_from_learning_start(
        episodes, timesteps, returns
    )
    success_written = plot_success_if_available(episodes, timesteps)
    print(f"episodes={len(episodes)}")
    print(f"last_completed_episode={episodes[-1]}")
    print(f"last_completed_episode_timestep={timesteps[-1]}")
    print(f"return_png={OUTPUT_DIR / 'training_episode_return.png'}")
    print(f"learning_starts={LEARNING_STARTS}")
    print(f"learning_start_episode={learning_start_episode}")
    print(f"first_logged_timestep_at_or_after_learning_start={first_logged_timestep}")
    print(
        "cropped_return_png="
        f"{OUTPUT_DIR / 'training_episode_return_from_learning_start_10_episode_average.png'}"
    )
    print(f"evaluation_archive={EVALUATION_PATH}")
    print(f"success_figure_written={success_written}")


if __name__ == "__main__":
    main()
