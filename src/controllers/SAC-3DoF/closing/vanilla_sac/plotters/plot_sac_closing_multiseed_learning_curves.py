"""Generate learning-curve plots for an existing multiseed closing run."""

import argparse
import json
import os
import sys
from pathlib import Path

CLOSING_DIR = Path(__file__).resolve().parents[2]
if str(CLOSING_DIR) not in sys.path:
    sys.path.insert(0, str(CLOSING_DIR))

from train_sac_closing_multiseed import (
    RESULTS_ROOT,
    write_learning_curve_plots,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot training-return and checkpoint-evaluation learning curves "
            "for a multiseed closing experiment."
        )
    )
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Training seeds to plot; defaults to experiment_config.json.",
    )
    parser.add_argument(
        "--highlight-checkpoint",
        type=int,
        default=None,
        help=(
            "Training timestep to mark on both plots, for example 400000."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result_root = os.path.join(RESULTS_ROOT, args.experiment_name)
    config_path = os.path.join(result_root, "experiment_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Experiment configuration not found: {config_path}")

    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    seeds = args.seeds or config["training_seeds"]
    write_learning_curve_plots(
        args.experiment_name,
        seeds,
        highlight_checkpoint=args.highlight_checkpoint,
    )

    plot_dir = os.path.join(result_root, "learning_curves")
    print(f"Learning-curve plots written to: {plot_dir}")


if __name__ == "__main__":
    main()
