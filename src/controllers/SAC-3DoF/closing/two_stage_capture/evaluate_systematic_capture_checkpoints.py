"""Evaluate every saved checkpoint from one systematic capture experiment."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from common import TRIAL_DIR


CHECKPOINT_PATTERN = re.compile(r"systematic_capture_sac_(\d+)_steps\.zip$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--distances", default="150")
    parser.add_argument("--episodes-per-distance", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=90_000)
    parser.add_argument("--max-time", type=float, default=2_000.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--overwrite-incomplete",
        action="store_true",
        help="Replace non-empty checkpoint directories lacking complete results.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def checkpoint_pairs(model_dir):
    pairs = []
    for model_path in model_dir.glob("systematic_capture_sac_*_steps.zip"):
        match = CHECKPOINT_PATTERN.fullmatch(model_path.name)
        if match is None:
            continue
        steps = int(match.group(1))
        vecnormalize_path = model_dir / (
            f"systematic_capture_sac_vecnormalize_{steps}_steps.pkl"
        )
        if not vecnormalize_path.is_file():
            raise FileNotFoundError(vecnormalize_path)
        pairs.append((steps, model_path, vecnormalize_path))
    if not pairs:
        raise FileNotFoundError(f"No systematic capture checkpoints in {model_dir}")
    return sorted(pairs)


def prepare_checkpoint_output(output_dir, output_root, overwrite_incomplete, dry_run):
    summary_path = output_dir / "summary.json"
    rollouts_path = output_dir / "rollouts.csv"
    if summary_path.is_file() and rollouts_path.is_file():
        print(f"Skipping completed evaluation: {output_dir}")
        return False
    if not output_dir.exists():
        return True

    contents = list(output_dir.iterdir())
    if contents and not overwrite_incomplete:
        raise FileExistsError(
            f"Incomplete evaluation directory is not empty: {output_dir}. "
            "Pass --overwrite-incomplete to replace it."
        )

    resolved_root = output_root.resolve()
    resolved_output = output_dir.resolve()
    if not resolved_output.is_relative_to(resolved_root):
        raise ValueError(f"Refusing to replace output outside {resolved_root}")
    if dry_run:
        print(f"Would replace incomplete evaluation: {output_dir}")
    else:
        shutil.rmtree(output_dir)
    return True


def main():
    args = parse_args()
    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        raise NotADirectoryError(model_dir)
    output_root = args.output_root or (
        TRIAL_DIR
        / "artifacts"
        / "systematic"
        / "results"
        / f"checkpoint_evaluations_{model_dir.name}"
    )
    evaluator = Path(__file__).with_name("evaluate_systematic_capture.py")

    for steps, model_path, vecnormalize_path in checkpoint_pairs(model_dir):
        output_dir = output_root / f"{steps}_steps"
        if not prepare_checkpoint_output(
            output_dir,
            output_root,
            args.overwrite_incomplete,
            args.dry_run,
        ):
            continue
        command = [
            sys.executable,
            str(evaluator),
            "--model",
            str(model_path),
            "--vecnormalize",
            str(vecnormalize_path),
            "--distances",
            args.distances,
            "--episodes-per-distance",
            str(args.episodes_per_distance),
            "--seed-start",
            str(args.seed_start),
            "--max-time",
            str(args.max_time),
            "--device",
            args.device,
            "--output-dir",
            str(output_dir),
        ]
        print(f"Evaluating {steps} steps -> {output_dir}")
        if args.dry_run:
            print(subprocess.list2cmdline(command))
        else:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
