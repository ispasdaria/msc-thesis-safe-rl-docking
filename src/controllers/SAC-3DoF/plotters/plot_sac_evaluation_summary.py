"""Generate compact SAC evaluation summary figures.

The script consumes only committed evaluation CSV and JSON artifacts; it does
not load a policy or rerun the environment. Outputs are written to
``SAC-3DoF/figures/evaluation_summary`` by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAC_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def resolve_column(columns, candidates: tuple[str, ...]) -> str:
    """Return the first supported column name present in the input data."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Missing required column; expected one of {candidates}")


def final_approach_distribution(output_dir: Path) -> None:
    base = (
        SAC_ROOT
        / "final_approach/results/retained_policy_evaluations/eval_many"
        / "sac_final_approach_portobs_rewardv9_100ksteps_seed12_optuna12"
    )
    frames = []
    for campaign in ("testing_seeds_0_199", "testing_seeds_1000_1199"):
        frame = pd.read_csv(base / campaign / "evaluation_many.csv")
        frame["campaign"] = campaign
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)

    distance_column = resolve_column(
        data.columns,
        ("final_port_distance", "final_port_distance_m"),
    )
    values = np.sort(data[distance_column].to_numpy(dtype=float))
    ecdf = np.arange(1, len(values) + 1) / len(values)

    fig, ax = plt.subplots(figsize=(3.45, 2.45), constrained_layout=True)
    ax.step(values, ecdf, where="post", color="#0072B2", linewidth=1.6)
    ax.axvline(0.1, color="black", linestyle="--", linewidth=1.0,
               label="docking threshold")
    ax.set(xlabel="Final docking-port distance [m]",
           ylabel="Empirical cumulative probability", xlim=(0.0, 0.105),
           ylim=(0.0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(output_dir / "final_approach_terminal_distance_ecdf.png", dpi=300)
    plt.close(fig)


def closing_comparison(output_dir: Path) -> None:
    vanilla_path = (
        SAC_ROOT
        / "closing/vanilla_sac/results/multiseed_evaluation"
        / "sac_closing_rewardv15_trial16_multiseed_600ksteps"
        / "test_seeds_1000_1099__final-ckpt_300k-400k-500k-600k"
        / "checkpoint_0300000_steps/seed_0/summary.json"
    )
    balanced_path = (
        SAC_ROOT
        / "closing/phase_balanced_replay/results/multiseed_evaluation"
        / "sac_closing_v15_trial16_terminal_phase_balanced_guarded600k_seed0"
        / "best_and_100k_checkpoint_seeds3000_3049"
        / "checkpoint_0100000_steps/seed_0/summary.json"
    )
    rows = [load_json(vanilla_path), load_json(balanced_path)]
    labels = ["Vanilla SAC\n(300k)", "Phase-balanced SAC\n(+100k)"]
    metrics = [
        ("mean_final_position_ratio", "Position"),
        ("mean_final_velocity_ratio", "Velocity"),
        ("mean_final_alignment_ratio", "Attitude"),
        ("mean_final_max_target_ratio", "Worst component"),
    ]
    for path, row in zip((vanilla_path, balanced_path), rows):
        missing = [key for key, _ in metrics if key not in row]
        if missing:
            raise ValueError(f"{path} is missing metrics: {missing}")

    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 2.8), constrained_layout=True)
    for idx, (row, label) in enumerate(zip(rows, labels)):
        ax.bar(x + (idx - 0.5) * width, [row[key] for key, _ in metrics],
               width, label=label)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0,
               label="P3 tolerance")
    ax.set_ylabel("Mean normalized terminal error")
    ax.set_xticks(x, [label for _, label in metrics])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.savefig(output_dir / "closing_terminal_ratio_comparison.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SAC_ROOT / "figures" / "evaluation_summary",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_approach_distribution(args.output_dir)
    closing_comparison(args.output_dir)


if __name__ == "__main__":
    main()
