"""Plot total training reward per completed closing episode.

The input is the ``learning_curve.csv`` produced by ``train_sac_closing.py``.
By default, the plot is saved to
``closing/results/reward_shaping_trials/episode_return_learning_curves/<model_name>.png``.

Examples:
    python plot_episode_returns.py results/reward_shaping_trials/learning_curves/<model_name>/learning_curve.csv
    python plot_episode_returns.py path/to/learning_curve.csv --output custom.png
    python plot_episode_returns.py path/to/learning_curve.csv --window 20
    python plot_episode_returns.py path/to/learning_curve.csv --show-best-checkpoint false
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (
    SCRIPT_DIR
    / "results"
    / "reward_shaping_trials"
    / "episode_return_learning_curves"
)


def infer_model_name(csv_path):
    if csv_path.name == "learning_curve.csv":
        return csv_path.parent.name
    return csv_path.stem


def default_output_path(csv_path):
    return OUTPUT_DIR / f"{infer_model_name(csv_path)}.png"


def read_episode_returns(csv_path):
    episodes = []
    timesteps = []
    episode_returns = []

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"episode", "episode_return"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required column(s): {missing}")

        timestep_column = "timestep"
        if timestep_column not in (reader.fieldnames or []):
            timestep_column = "global_timestep"
        if timestep_column not in (reader.fieldnames or []):
            raise ValueError("Missing required column: timestep or global_timestep")

        for row in reader:
            episodes.append(int(row["episode"]))
            timesteps.append(int(row[timestep_column]))
            episode_returns.append(float(row["episode_return"]))

    if not episodes:
        raise ValueError(f"No episode rows found in {csv_path}")

    return episodes, timesteps, episode_returns


def read_best_eval_checkpoint(csv_path, episodes, timesteps, episode_returns):
    """Return the best evaluation checkpoint, or None if no eval archive exists."""
    eval_path = csv_path.with_name("evaluations.npz")
    if not eval_path.exists():
        return None

    with np.load(eval_path) as eval_data:
        eval_timesteps = eval_data["timesteps"]
        eval_results = eval_data["results"]

    mean_eval_returns = eval_results.mean(axis=1)
    best_eval_index = int(np.argmax(mean_eval_returns))
    best_timestep = int(eval_timesteps[best_eval_index])

    timestep_array = np.asarray(timesteps)
    candidate_indices = np.flatnonzero(timestep_array >= best_timestep)
    episode_index = (
        int(candidate_indices[0]) if len(candidate_indices) else len(episodes) - 1
    )

    return {
        "timestep": best_timestep,
        "mean_eval_return": float(mean_eval_returns[best_eval_index]),
        "episode": episodes[episode_index],
        "episode_return": episode_returns[episode_index],
    }


def moving_average(values, window):
    if window < 1:
        raise ValueError("Moving-average window must be at least 1")
    if len(values) < window:
        return np.array([])
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_episode_returns(
    csv_path,
    output_path,
    moving_average_window=10,
    show_best_checkpoint=True,
):
    episodes, timesteps, episode_returns = read_episode_returns(csv_path)
    averaged_returns = moving_average(episode_returns, moving_average_window)
    averaged_episodes = episodes[moving_average_window - 1 :]
    best_checkpoint = None
    if show_best_checkpoint:
        best_checkpoint = read_best_eval_checkpoint(
            csv_path, episodes, timesteps, episode_returns
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(
        episodes,
        episode_returns,
        color="tab:blue",
        linewidth=1.0,
        alpha=0.8,
        label="Episode return",
    )
    if len(averaged_returns) > 0:
        plt.plot(
            averaged_episodes,
            averaged_returns,
            color="tab:orange",
            linewidth=2.0,
            label=f"{moving_average_window}-episode moving average",
        )
    if best_checkpoint is not None:
        plt.scatter(
            best_checkpoint["episode"],
            best_checkpoint["episode_return"],
            color="tab:red",
            s=55,
            zorder=5,
            label=f"Best eval checkpoint (step {best_checkpoint['timestep']})",
        )
        plt.axvline(
            best_checkpoint["episode"],
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
        )

    model_name = infer_model_name(csv_path)
    plt.xlabel("Episode number")
    plt.ylabel("Total reward per episode")
    plt.title(f"Total Reward per Training Episode\n{model_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return best_checkpoint


def parse_bool(value):
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected a boolean value: true/false, yes/no, 1/0, or on/off."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot episode number versus total training-episode reward."
    )
    parser.add_argument("csv_path", type=Path, help="Path to learning_curve.csv")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=(
            "Output PNG path. By default, uses "
            "closing/results/reward_shaping_trials/"
            "episode_return_learning_curves/<model_name>.png."
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Moving-average window in episodes (default: 10).",
    )
    parser.add_argument(
        "--show-best-checkpoint",
        type=parse_bool,
        default=True,
        help=(
            "Mark the best checkpoint using evaluations.npz beside the CSV "
            "when available (default: true)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = args.csv_path.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    output_path = args.output
    if output_path is None:
        output_path = default_output_path(csv_path)
    else:
        output_path = output_path.resolve()

    best_checkpoint = plot_episode_returns(
        csv_path,
        output_path,
        args.window,
        args.show_best_checkpoint,
    )
    print(f"Saved episode-return plot to: {output_path}")
    if not args.show_best_checkpoint:
        print("Best-eval checkpoint marker disabled.")
    elif best_checkpoint is None:
        print("No evaluations.npz found beside the CSV; no checkpoint marked.")
    else:
        print(
            "Best eval checkpoint: "
            f"timestep {best_checkpoint['timestep']}, "
            f"mean eval return {best_checkpoint['mean_eval_return']:.10f}, "
            f"shown near episode {best_checkpoint['episode']}"
        )


if __name__ == "__main__":
    main()
