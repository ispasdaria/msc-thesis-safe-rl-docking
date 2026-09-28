"""
Plot total training reward per completed episode from a learning_curve.csv file.

By default, plots are saved to:
    final_approach/results/development_experiments/
    episode_return_learning_curves/<model_name>.png

The model name is inferred from the parent folder of the CSV path.

Usage:
    python plot_episode_returns.py path/to/<model_name>/learning_curve.csv
    python plot_episode_returns.py path/to/learning_curve.csv --output path/to/custom.png
    python plot_episode_returns.py path/to/learning_curve.csv --show-best-checkpoint false
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FINAL_APPROACH_DIR = Path(__file__).resolve().parents[1]
EPISODE_RETURN_LEARNING_CURVES_DIR = (
    FINAL_APPROACH_DIR / "results" / "development_experiments"
    / "episode_return_learning_curves"
)


def infer_model_name(csv_path):
    if csv_path.name == "learning_curve.csv":
        return csv_path.parent.name

    return csv_path.stem


def default_output_path(csv_path):
    model_name = infer_model_name(csv_path)
    return EPISODE_RETURN_LEARNING_CURVES_DIR / f"{model_name}.png"


def read_episode_returns(csv_path):
    episodes = []
    timesteps = []
    episode_returns = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_columns = {"episode", "episode_return"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required column(s): {missing}")

        timestep_column = "timestep"
        if timestep_column not in (reader.fieldnames or []):
            timestep_column = "global_timestep"

        for row in reader:
            episodes.append(int(row["episode"]))
            timesteps.append(int(row[timestep_column]))
            episode_returns.append(float(row["episode_return"]))

    if not episodes:
        raise ValueError(f"No episode rows found in {csv_path}")

    return episodes, timesteps, episode_returns


def find_episode_at_or_after_timestep(episodes, timesteps, target_timestep):
    timestep_array = np.asarray(timesteps)
    candidate_indices = np.flatnonzero(timestep_array >= target_timestep)

    if len(candidate_indices) == 0:
        return len(episodes) - 1

    return int(candidate_indices[0])


def read_best_eval_checkpoint(csv_path, episodes, timesteps, episode_returns):
    eval_path = csv_path.with_name("evaluations.npz")

    if not eval_path.exists():
        return None

    eval_data = np.load(eval_path)
    eval_timesteps = eval_data["timesteps"]
    eval_results = eval_data["results"]
    mean_eval_returns = eval_results.mean(axis=1)
    best_eval_idx = int(np.argmax(mean_eval_returns))
    best_timestep = int(eval_timesteps[best_eval_idx])
    episode_idx = find_episode_at_or_after_timestep(episodes, timesteps, best_timestep)

    return {
        "eval_path": eval_path,
        "timestep": best_timestep,
        "mean_eval_return": float(mean_eval_returns[best_eval_idx]),
        "episode": episodes[episode_idx],
        "episode_return": episode_returns[episode_idx],
    }


def moving_average(values, window):
    if len(values) < window:
        return np.array([])

    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")


def parse_bool(value):
    normalized = value.strip().lower()

    if normalized in {"true", "1", "yes", "y", "on"}:
        return True

    if normalized in {"false", "0", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(
        "Expected a boolean value: true/false, yes/no, 1/0, or on/off."
    )


def plot_episode_returns(
    csv_path,
    output_path,
    moving_average_window=10,
    show_best_checkpoint=True,
):
    episodes, timesteps, episode_returns = read_episode_returns(csv_path)
    model_name = infer_model_name(csv_path)
    best_eval_checkpoint = None
    if show_best_checkpoint:
        best_eval_checkpoint = read_best_eval_checkpoint(
            csv_path,
            episodes,
            timesteps,
            episode_returns,
        )
    averaged_returns = moving_average(episode_returns, moving_average_window)
    averaged_episodes = episodes[moving_average_window - 1 :]

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
    if best_eval_checkpoint is not None:
        plt.scatter(
            best_eval_checkpoint["episode"],
            best_eval_checkpoint["episode_return"],
            color="tab:red",
            s=55,
            zorder=5,
            label=(
                "Best eval checkpoint "
                f"(step {best_eval_checkpoint['timestep']})"
            ),
        )
        plt.axvline(
            best_eval_checkpoint["episode"],
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
        )
    plt.xlabel("Episode number")
    plt.ylabel("Total reward per episode")
    plt.title(f"Total Reward per Training Episode\n{model_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return best_eval_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot episode number vs total episode reward from a learning_curve.csv file."
        )
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to a learning_curve.csv file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Defaults to "
            "final_approach/results/development_experiments/"
            "episode_return_learning_curves/<model_name>.png."
        ),
    )
    parser.add_argument(
        "--show-best-checkpoint",
        type=parse_bool,
        default=True,
        help=(
            "Whether to mark the best eval checkpoint from evaluations.npz "
            "when available. Default: true."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = args.csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    output_path = args.output
    if output_path is None:
        output_path = default_output_path(csv_path)

    best_eval_checkpoint = plot_episode_returns(
        csv_path,
        output_path,
        show_best_checkpoint=args.show_best_checkpoint,
    )
    print(f"Saved episode-return plot to: {output_path}")
    if not args.show_best_checkpoint:
        print("Best-eval checkpoint marker disabled.")
    elif best_eval_checkpoint is None:
        print("No evaluations.npz found beside the CSV; no best-eval checkpoint marked.")
    else:
        print(
            "Best eval checkpoint: "
            f"timestep {best_eval_checkpoint['timestep']}, "
            f"mean eval return {best_eval_checkpoint['mean_eval_return']:.10f}, "
            f"shown near episode {best_eval_checkpoint['episode']}"
        )


if __name__ == "__main__":
    main()
