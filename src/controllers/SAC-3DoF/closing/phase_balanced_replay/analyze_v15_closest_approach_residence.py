"""Measure closest-approach residence for vanilla V15 SAC, seeds 0-49.

No training is performed.  Each policy rollout is deterministic and uses the
retained 400k Optuna-trial-16 model and its matching VecNormalize statistics.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CLOSING_DIR = HERE.parent
VANILLA_DIR = CLOSING_DIR / "vanilla_sac"
SRC = HERE.parents[3]
for path in (HERE, CLOSING_DIR, VANILLA_DIR, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_sac_closing as evaluator


MODEL_NAME = "sac_closing_p3augobs_rewardv15_optuna_trial16_400ksteps_seed0_buffer300k"
MODEL_PATH = VANILLA_DIR / "models" / "reward_shaping_trials" / f"{MODEL_NAME}.zip"
VECNORM_PATH = (
    VANILLA_DIR / "models" / "reward_shaping_trials" / f"{MODEL_NAME}_vecnormalize.pkl"
)
PHASE_MODEL_PATH = (
    HERE / "models" / "multiseed"
    / "sac_closing_v15_trial16_terminal_phase_balanced_guarded600k_seed0"
    / "seed_0" / "model_0100000_steps.zip"
)
PHASE_VECNORM_PATH = PHASE_MODEL_PATH.with_name(
    "model_0100000_steps_vecnormalize.pkl"
)
DEFAULT_OUTPUT = (
    VANILLA_DIR / "results" / "retained_figures" / "horizon_comparison"
    / "closest_approach_residence_seeds0_49.csv"
)
HORIZON = 20_000
_MODEL = None
_VECNORM = None
_DUMMY = None


def initialize_worker(model_path: str, vecnorm_path: str):
    global _MODEL, _VECNORM, _DUMMY
    evaluator.MODEL_PATH = model_path
    evaluator.VECNORM_PATH = vecnorm_path
    with contextlib.redirect_stdout(io.StringIO()):
        _VECNORM, _DUMMY = evaluator.load_vecnormalize_stats()
        _MODEL = evaluator.SAC.load(model_path)


def evaluate_seed(seed: int) -> dict[str, float | int | bool | str]:
    env = evaluator.Docking3DoFGymEnv(
        phase="closing", mode="testing", dt=1.0, max_time=float(HORIZON),
        integrator="rk4", normalize_action=True, seed=seed,
        closing_reward_version="v15",
    )
    obs, _ = env.reset(seed=seed)
    target = np.asarray(env.core_env.p3_target, dtype=float)
    info = {}
    for _ in range(HORIZON):
        normalized = evaluator.normalize_observation(_VECNORM, obs)
        action, _ = _MODEL.predict(normalized, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action[0])
        if terminated or truncated:
            break
    times, states, *_ = env.get_history_arrays()
    env.close()
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)
    distance = np.linalg.norm(states[:, :2] - target[:2], axis=1)
    imin = int(np.argmin(distance))
    threshold = 1.1 * float(distance[imin])
    inside = distance <= threshold
    left = right = imin
    while left > 0 and inside[left - 1]:
        left -= 1
    while right + 1 < len(inside) and inside[right + 1]:
        right += 1
    later_left_band = bool(np.any(~inside[right + 1 :]))
    return {
        "seed": seed,
        "closest_time_s": float(times[imin]),
        "minimum_position_error_m": float(distance[imin]),
        "residence_threshold_m": threshold,
        "residence_start_s": float(times[left]),
        "residence_end_s": float(times[right]),
        "contiguous_residence_s": float(times[right] - times[left]),
        "terminal_position_error_m": float(distance[-1]),
        "terminal_to_minimum_ratio": float(distance[-1] / distance[imin]),
        "left_10pct_band_after_closest": later_left_band,
        "closest_at_terminal": bool(imin == len(distance) - 1),
        "termination_reason": str(info.get("termination_reason", "unknown")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--policy", choices=("vanilla", "phase-balanced"), default="vanilla")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.policy == "phase-balanced":
        model_path, vecnorm_path = PHASE_MODEL_PATH, PHASE_VECNORM_PATH
        default_output = (
            HERE / "results" / "retained_figures"
            / "retained_checkpoint_diagnostics"
            / "closest_approach_residence_seeds0_49.csv"
        )
    else:
        model_path, vecnorm_path = MODEL_PATH, VECNORM_PATH
        default_output = DEFAULT_OUTPUT
    output = args.output or default_output
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize_worker,
        initargs=(str(model_path), str(vecnorm_path)),
    ) as executor:
        rows = list(executor.map(evaluate_seed, range(50)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    durations = np.asarray([float(row["contiguous_residence_s"]) for row in rows])
    departed = sum(bool(row["left_10pct_band_after_closest"]) for row in rows)
    print(f"policy={args.policy}")
    print(f"model={model_path.resolve()}")
    print(f"vecnormalize={vecnorm_path.resolve()}")
    print(f"output={output.resolve()}")
    print(f"residence_s mean={durations.mean():.3f} median={np.median(durations):.3f} "
          f"std={durations.std(ddof=1):.3f} min={durations.min():.3f} "
          f"q1={np.percentile(durations, 25):.3f} q3={np.percentile(durations, 75):.3f} "
          f"max={durations.max():.3f}")
    print(f"left_10pct_band_after_closest={departed}/50")
    print(f"closest_at_terminal={sum(bool(r['closest_at_terminal']) for r in rows)}/50")


if __name__ == "__main__":
    main()
