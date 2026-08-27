"""Evaluate systematic capture checkpoints from controlled X_H distances."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from common import SRC_ROOT, TRIAL_DIR, write_json

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from systematic_capture_env import SystematicCaptureEnv  # noqa: E402
from systematic_capture_action import LowPassCaptureActionWrapper  # noqa: E402
from systematic_capture_observation import (  # noqa: E402
    FixedCaptureObservationScale,
    MarkovCaptureObservationScale,
)
from systematic_capture_reset_env import SystematicCaptureResetWrapper  # noqa: E402
from systematic_interface import DEFAULT_CONTRACT  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, default=None)
    parser.add_argument("--distances", default="30,60,100,150")
    parser.add_argument("--episodes-per-distance", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=20_000)
    parser.add_argument("--max-time", type=float, default=2_000.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--action-filter-alpha", type=float, default=None)
    return parser.parse_args()


def parse_distances(value):
    distances = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not distances:
        raise ValueError("At least one evaluation distance is required.")
    for distance in distances:
        if not 0.0 < distance <= DEFAULT_CONTRACT.handover_distance_m:
            raise ValueError(f"Invalid capture distance: {distance:g} m.")
    return distances


def make_env(max_time, seed, observation_mode="raw", action_filter_alpha=None):
    env = SystematicCaptureResetWrapper(
        SystematicCaptureEnv(mode="testing", max_time=max_time, seed=seed),
        capture_distance_m=DEFAULT_CONTRACT.handover_distance_m,
        enable_lateral_divergence_termination=observation_mode != "raw",
    )
    if observation_mode == "fixed":
        env = FixedCaptureObservationScale(env)
    elif observation_mode == "markov":
        env = MarkovCaptureObservationScale(env)
    elif observation_mode != "raw":
        raise ValueError(f"Unknown observation mode: {observation_mode}")
    if action_filter_alpha is not None:
        if observation_mode != "markov":
            raise ValueError("Action filtering requires the Markov observation.")
        env = LowPassCaptureActionWrapper(env, alpha=action_filter_alpha)
    return Monitor(env)


def configured_action_filter_alpha(model_path):
    config_path = (
        TRIAL_DIR
        / "artifacts"
        / "systematic"
        / "results"
        / model_path.parent.name
        / "run_config.json"
    )
    if not config_path.is_file():
        return None
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    return config.get("action_filter", {}).get("alpha")


def main():
    args = parse_args()
    distances = parse_distances(args.distances)
    if args.episodes_per_distance <= 0:
        raise ValueError("--episodes-per-distance must be positive.")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    vecnormalize_path = args.vecnormalize or args.model.with_name(
        "final_vecnormalize.pkl"
    )
    if not vecnormalize_path.is_file():
        raise FileNotFoundError(vecnormalize_path)
    action_filter_alpha = args.action_filter_alpha
    if action_filter_alpha is None:
        action_filter_alpha = configured_action_filter_alpha(args.model)

    output_dir = args.output_dir or (
        TRIAL_DIR
        / "artifacts"
        / "systematic"
        / "results"
        / f"eval_{args.model.parent.name}_{args.seed_start}"
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse evaluation directory: {output_dir}")
    output_dir.mkdir(parents=True)

    model = SAC.load(str(args.model), device=args.device)
    observation_size = int(np.prod(model.observation_space.shape))
    if observation_size == 15:
        observation_mode = "markov"
        raw_vec = DummyVecEnv(
            [
                lambda: make_env(
                    args.max_time,
                    args.seed_start,
                    observation_mode,
                    action_filter_alpha,
                )
            ]
        )
        vec_env = VecNormalize.load(str(vecnormalize_path), raw_vec)
    elif observation_size == 12:
        observation_mode = "raw"
        raw_vec = DummyVecEnv(
            [lambda: make_env(args.max_time, args.seed_start, observation_mode)]
        )
        vec_env = VecNormalize.load(str(vecnormalize_path), raw_vec)
        if not bool(vec_env.norm_obs):
            vec_env.close()
            observation_mode = "fixed"
            raw_vec = DummyVecEnv(
                [
                    lambda: make_env(
                        args.max_time,
                        args.seed_start,
                        observation_mode,
                    )
                ]
            )
            vec_env = VecNormalize.load(str(vecnormalize_path), raw_vec)
    else:
        raise ValueError(
            f"Unsupported model observation size: {observation_size}."
        )
    vec_env.training = False
    vec_env.norm_reward = False
    model.set_env(vec_env)

    rows = []
    summaries = []
    try:
        for distance_index, distance in enumerate(distances):
            vec_env.env_method("set_capture_distance", distance)
            reason_counts = Counter()
            successes = 0
            steps_to_success = []
            for episode in range(args.episodes_per_distance):
                seed = args.seed_start + distance_index * args.episodes_per_distance + episode
                vec_env.seed(seed)
                observation = vec_env.reset()
                done = np.array([False])
                steps = 0
                episode_reward = 0.0
                final_info = {}
                while not bool(done[0]):
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, done, infos = vec_env.step(action)
                    steps += 1
                    episode_reward += float(reward[0])
                    final_info = infos[0]
                success = bool(final_info.get("success", False))
                reason = final_info.get("termination_reason") or "unknown"
                reason_counts[reason] += 1
                successes += int(success)
                if success:
                    steps_to_success.append(steps)
                if "terminal_physical_state" not in final_info:
                    raise RuntimeError(
                        "Terminal physical state is missing from episode info; "
                        "refusing to record the auto-reset state."
                    )
                final_state = np.asarray(
                    final_info["terminal_physical_state"], dtype=np.float64
                )
                final_target = np.asarray(
                    final_info["terminal_p3_target"], dtype=np.float64
                )
                geometry = final_info.get("geometry", {})
                velocity = final_info.get("velocity", {})
                alignment = final_info.get("alignment", {})
                rows.append(
                    {
                        "distance_m": distance,
                        "seed": seed,
                        "success": success,
                        "termination_reason": reason,
                        "failure_reason": final_info.get("failure_reason") or "",
                        "violated_constraints": ";".join(
                            final_info.get("violated_constraints", [])
                        ),
                        "alignment_warning_reasons": ";".join(
                            final_info.get("alignment_warning_reasons", [])
                        ),
                        "alignment_unsafe_reasons": ";".join(
                            final_info.get("alignment_unsafe_reasons", [])
                        ),
                        "geometry_warn": bool(geometry.get("warn", False)),
                        "geometry_unsafe": bool(geometry.get("unsafe", False)),
                        "velocity_warn": bool(velocity.get("warn", False)),
                        "velocity_unsafe": bool(velocity.get("unsafe", False)),
                        "alignment_warn": bool(alignment.get("warn", False)),
                        "alignment_unsafe": bool(alignment.get("unsafe", False)),
                        "steps": steps,
                        "episode_reward": episode_reward,
                        **{
                            f"final_{label}": float(value)
                            for label, value in zip(
                                ("rx", "ry", "vx", "vy", "psi", "omega"),
                                final_state,
                            )
                        },
                        **{
                            f"p3_target_{label}": float(value)
                            for label, value in zip(
                                ("rx", "ry", "vx", "vy", "psi", "omega"),
                                final_target,
                            )
                        },
                    }
                )
            summaries.append(
                {
                    "distance_m": distance,
                    "episodes": args.episodes_per_distance,
                    "successes": successes,
                    "success_rate": successes / args.episodes_per_distance,
                    "termination_reasons": dict(reason_counts),
                    "mean_steps_to_success": (
                        float(np.mean(steps_to_success)) if steps_to_success else None
                    ),
                }
            )
    finally:
        vec_env.close()

    with (output_dir / "rollouts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output_dir / "summary.json",
        {
            "schema": "systematic_capture_evaluation_v2_terminal_state_safe",
            "model": str(args.model.resolve()),
            "vecnormalize": str(vecnormalize_path.resolve()),
            "deterministic": True,
            "success_set": "validated_X_P3_testing",
            "fixed_observation_scaling": observation_mode in {"fixed", "markov"},
            "observation_mode": observation_mode,
            "action_filter_alpha": action_filter_alpha,
            "by_distance": summaries,
        },
    )
    print(json.dumps(summaries, indent=2))
    print(f"Evaluation: {output_dir}")


if __name__ == "__main__":
    main()
