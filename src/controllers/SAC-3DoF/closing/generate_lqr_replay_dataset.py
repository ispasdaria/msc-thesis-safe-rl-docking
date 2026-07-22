"""Generate full-episode LQR demonstrations for SAC replay preloading.

The LQR continues to act directly on :class:`Docking3DoFEnv`.  A separate
Gymnasium wrapper is used only to expose the observation, normalized action,
reward, and termination semantics consumed by closing-phase SAC.

Only complete, successful P2-to-P3 episodes are written to the dataset.  No
existing simulator, controller, or training module is modified by this script.

Example
-------
python generate_lqr_replay_dataset.py --num-successes 30 --reward-version v20
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
SRC_PATH = REPO_ROOT / "src"
LQR_PATH = SRC_PATH / "controllers" / "LQR-3DoF"

for import_path in (SRC_PATH, LQR_PATH):
    import_path_string = str(import_path)
    if import_path_string not in sys.path:
        sys.path.insert(0, import_path_string)

from envs.docking3dof.gym_env import Docking3DoFGymEnv  # noqa: E402
from lqr_3dof_closing import (  # noqa: E402
    LQRClosingController,
    build_Q_closing,
    build_R_closing,
    build_linearized_3dof_matrices,
    compute_lqr_gain,
)


DATASET_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "demonstrations"
REQUIRED_TRANSITION_KEYS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminated",
    "truncated",
    "episode_ids",
    "episode_offsets",
    "episode_seeds",
)


def available_closing_reward_versions():
    """Return selectable rewards for either repository Gym API revision."""

    return tuple(
        getattr(Docking3DoFGymEnv, "closing_reward_versions", ("v15",))
    )


def make_demonstration_env(mode, seed, reward_version):
    """Construct the Gym adapter without assuming a versioned reward API."""

    kwargs = {
        "phase": "closing",
        "mode": mode,
        "dt": 1.0,
        "max_time": None,
        "integrator": "rk4",
        "normalize_action": True,
        "seed": seed,
    }
    parameters = inspect.signature(Docking3DoFGymEnv.__init__).parameters
    if "closing_reward_version" in parameters:
        kwargs["closing_reward_version"] = reward_version
    elif reward_version != "v15":
        raise ValueError(
            "The current Gym environment exposes one closing reward only; "
            "use reward version 'v15'."
        )
    return Docking3DoFGymEnv(**kwargs)


def build_lqr_controller(core_env):
    """Build the repository's tuned closing LQR for one core environment."""

    Q = build_Q_closing()
    R = build_R_closing(core_env)
    A, B = build_linearized_3dof_matrices(core_env)
    K, _ = compute_lqr_gain(A=A, B=B, Q=Q, R=R)
    return LQRClosingController(K)


def collect_lqr_episode(env, controller, seed):
    """Run one complete closing episode and return SAC-format transitions."""

    observation, _ = env.reset(seed=int(seed))
    core_env = env.core_env
    initial_state = core_env.state.copy()
    p3_target = core_env.p3_target.copy()
    action_limits = np.array(
        [core_env.ax_max, core_env.ay_max, core_env.alphaz_max],
        dtype=np.float64,
    )

    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminated_values = []
    truncated_values = []
    saturation_values = []

    terminated = False
    truncated = False
    final_info = {}

    while not (terminated or truncated):
        physical_action_raw = controller.compute_action(
            core_env,
            core_env.state,
        )
        normalized_action_raw = physical_action_raw / action_limits
        normalized_action = np.clip(normalized_action_raw, -1.0, 1.0)
        saturated = bool(
            np.any(np.abs(normalized_action_raw - normalized_action) > 1e-12)
        )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            final_info,
        ) = env.step(normalized_action)

        observations.append(np.asarray(observation, dtype=np.float32).copy())
        actions.append(np.asarray(normalized_action, dtype=np.float32).copy())
        rewards.append(float(reward))
        next_observations.append(
            np.asarray(next_observation, dtype=np.float32).copy()
        )
        terminated_values.append(bool(terminated))
        truncated_values.append(bool(truncated))
        saturation_values.append(saturated)

        observation = next_observation

    transition_count = len(rewards)
    return {
        "seed": int(seed),
        "success": bool(final_info.get("success", False)),
        "termination_reason": final_info.get("termination_reason"),
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_observations": np.asarray(
            next_observations,
            dtype=np.float32,
        ),
        "terminated": np.asarray(terminated_values, dtype=np.bool_),
        "truncated": np.asarray(truncated_values, dtype=np.bool_),
        "initial_state": np.asarray(initial_state, dtype=np.float64),
        "p3_target": np.asarray(p3_target, dtype=np.float64),
        "final_state": core_env.state.copy(),
        "episode_return": float(np.sum(rewards, dtype=np.float64)),
        "saturation_fraction": (
            float(np.mean(saturation_values)) if transition_count else 0.0
        ),
    }


def combine_successful_episodes(episodes, reward_version, mode):
    """Combine complete successful episodes without transition subsampling."""

    if not episodes:
        raise ValueError("At least one successful episode is required.")
    if not all(episode["success"] for episode in episodes):
        raise ValueError("Demonstration datasets may contain only successes.")

    episode_lengths = np.asarray(
        [len(episode["rewards"]) for episode in episodes],
        dtype=np.int64,
    )
    episode_offsets = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(episode_lengths))
    )
    episode_ids = np.concatenate(
        [
            np.full(length, episode_id, dtype=np.int32)
            for episode_id, length in enumerate(episode_lengths)
        ]
    )

    metadata = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "phase": "closing",
        "mode": str(mode),
        "reward_version": str(reward_version),
        "normalize_action": True,
        "observation_size": 12,
        "action_size": 3,
        "successful_episodes": len(episodes),
        "transition_count": int(episode_offsets[-1]),
        "only_complete_successes": True,
        "controller": "LQRClosingController",
    }

    dataset = {
        "observations": np.concatenate(
            [episode["observations"] for episode in episodes]
        ),
        "actions": np.concatenate([episode["actions"] for episode in episodes]),
        "rewards": np.concatenate([episode["rewards"] for episode in episodes]),
        "next_observations": np.concatenate(
            [episode["next_observations"] for episode in episodes]
        ),
        "terminated": np.concatenate(
            [episode["terminated"] for episode in episodes]
        ),
        "truncated": np.concatenate(
            [episode["truncated"] for episode in episodes]
        ),
        "episode_ids": episode_ids,
        "episode_offsets": episode_offsets,
        "episode_seeds": np.asarray(
            [episode["seed"] for episode in episodes],
            dtype=np.int64,
        ),
        "episode_returns": np.asarray(
            [episode["episode_return"] for episode in episodes],
            dtype=np.float64,
        ),
        "saturation_fractions": np.asarray(
            [episode["saturation_fraction"] for episode in episodes],
            dtype=np.float64,
        ),
        "initial_states": np.stack(
            [episode["initial_state"] for episode in episodes]
        ),
        "p3_targets": np.stack([episode["p3_target"] for episode in episodes]),
        "final_states": np.stack(
            [episode["final_state"] for episode in episodes]
        ),
        "termination_reasons": np.asarray(
            [episode["termination_reason"] for episode in episodes],
            dtype=np.str_,
        ),
        "metadata_json": np.asarray(json.dumps(metadata, indent=2)),
    }
    validate_demonstration_dataset(dataset)
    return dataset


def _metadata_from_dataset(dataset):
    value = np.asarray(dataset["metadata_json"])
    if value.shape != ():
        raise ValueError("metadata_json must be a scalar string.")
    return json.loads(str(value.item()))


def validate_demonstration_dataset(dataset, expected_reward_version=None):
    """Validate replay compatibility and full successful episode boundaries."""

    missing = [key for key in REQUIRED_TRANSITION_KEYS if key not in dataset]
    if "metadata_json" not in dataset:
        missing.append("metadata_json")
    if missing:
        raise ValueError(f"Demonstration dataset is missing keys: {missing}")

    observations = np.asarray(dataset["observations"])
    actions = np.asarray(dataset["actions"])
    rewards = np.asarray(dataset["rewards"])
    next_observations = np.asarray(dataset["next_observations"])
    terminated = np.asarray(dataset["terminated"], dtype=np.bool_)
    truncated = np.asarray(dataset["truncated"], dtype=np.bool_)
    episode_ids = np.asarray(dataset["episode_ids"])
    episode_offsets = np.asarray(dataset["episode_offsets"], dtype=np.int64)
    episode_seeds = np.asarray(dataset["episode_seeds"])
    metadata = _metadata_from_dataset(dataset)

    transition_count = len(rewards)
    expected_lengths = {
        "observations": len(observations),
        "actions": len(actions),
        "next_observations": len(next_observations),
        "terminated": len(terminated),
        "truncated": len(truncated),
        "episode_ids": len(episode_ids),
    }
    inconsistent = {
        key: value
        for key, value in expected_lengths.items()
        if value != transition_count
    }
    if inconsistent:
        raise ValueError(
            "Transition arrays have inconsistent lengths: "
            f"rewards={transition_count}, others={inconsistent}."
        )
    if transition_count <= 0:
        raise ValueError("Demonstration dataset contains no transitions.")
    if observations.shape != (transition_count, 12):
        raise ValueError(
            "observations must have shape (N, 12), got "
            f"{observations.shape}."
        )
    if next_observations.shape != (transition_count, 12):
        raise ValueError(
            "next_observations must have shape (N, 12), got "
            f"{next_observations.shape}."
        )
    if actions.shape != (transition_count, 3):
        raise ValueError(
            f"actions must have shape (N, 3), got {actions.shape}."
        )
    for name, values in (
        ("observations", observations),
        ("actions", actions),
        ("rewards", rewards),
        ("next_observations", next_observations),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values.")
    if np.any(actions < -1.000001) or np.any(actions > 1.000001):
        raise ValueError("Normalized demonstration actions must lie in [-1, 1].")

    episode_count = len(episode_seeds)
    if episode_offsets.shape != (episode_count + 1,):
        raise ValueError(
            "episode_offsets must have one more entry than episode_seeds."
        )
    if episode_offsets[0] != 0 or episode_offsets[-1] != transition_count:
        raise ValueError("episode_offsets must span all transitions from 0 to N.")
    if np.any(np.diff(episode_offsets) <= 0):
        raise ValueError("Every stored demonstration episode must be non-empty.")

    expected_ids = np.concatenate(
        [
            np.full(
                episode_offsets[index + 1] - episode_offsets[index],
                index,
                dtype=np.int32,
            )
            for index in range(episode_count)
        ]
    )
    if not np.array_equal(episode_ids, expected_ids):
        raise ValueError("episode_ids do not agree with episode_offsets.")

    terminal_indices = episode_offsets[1:] - 1
    if not np.all(terminated[terminal_indices]):
        raise ValueError("Every demonstration must end in true termination.")
    if np.any(truncated):
        raise ValueError("Successful demonstration datasets may not contain timeouts.")
    nonterminal_mask = np.ones(transition_count, dtype=np.bool_)
    nonterminal_mask[terminal_indices] = False
    if np.any(terminated[nonterminal_mask]):
        raise ValueError("A demonstration contains an early terminal transition.")

    if metadata.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported demonstration schema version: "
            f"{metadata.get('schema_version')}."
        )
    if metadata.get("phase") != "closing":
        raise ValueError("Demonstration dataset phase must be 'closing'.")
    if metadata.get("transition_count") != transition_count:
        raise ValueError("metadata transition_count does not match the arrays.")
    if metadata.get("successful_episodes") != episode_count:
        raise ValueError("metadata successful_episodes does not match the arrays.")
    if expected_reward_version is not None:
        reward_version = metadata.get("reward_version")
        if reward_version != expected_reward_version:
            raise ValueError(
                "Demonstration reward version does not match training: "
                f"{reward_version!r} != {expected_reward_version!r}."
            )
    return metadata


def save_demonstration_dataset(path, dataset, overwrite=False):
    """Write a validated demonstration dataset to a compressed NPZ file."""

    validate_demonstration_dataset(dataset)
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Dataset already exists: {path}. Pass --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)
    return path


def load_demonstration_dataset(path, expected_reward_version=None):
    """Load a demonstration NPZ without enabling pickle deserialization."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Demonstration dataset not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        dataset = {key: archive[key] for key in archive.files}
    metadata = validate_demonstration_dataset(
        dataset,
        expected_reward_version=expected_reward_version,
    )
    return dataset, metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate complete successful closing LQR trajectories in SAC "
            "replay-buffer format."
        )
    )
    parser.add_argument("--num-successes", type=int, default=30)
    parser.add_argument(
        "--seed-start",
        type=int,
        default=20_000,
        help="First environment seed attempted (default: 20000).",
    )
    parser.add_argument(
        "--candidate-seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit candidate seeds to run in the given order. This is useful "
            "for reproducing a previously audited successful demonstration set."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=300,
        help="Maximum full LQR episodes attempted (default: 300).",
    )
    parser.add_argument(
        "--mode",
        choices=["training", "testing", "nominal"],
        default="training",
    )
    parser.add_argument(
        "--reward-version",
        choices=list(available_closing_reward_versions()),
        default="v15",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NPZ path. A descriptive path is generated by default.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_successes <= 0:
        parser.error("--num-successes must be positive.")
    if args.max_attempts < args.num_successes:
        parser.error("--max-attempts must be at least --num-successes.")
    if args.candidate_seeds is not None:
        if len(set(args.candidate_seeds)) != len(args.candidate_seeds):
            parser.error("--candidate-seeds must be unique.")
        if len(args.candidate_seeds) < args.num_successes:
            parser.error(
                "--candidate-seeds must contain at least --num-successes values."
            )
    return args


def main():
    args = parse_args()
    output_path = args.output
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / (
            f"lqr_closing_{args.mode}_reward{args.reward_version}_"
            f"{args.num_successes}successes.npz"
        )

    env = make_demonstration_env(
        mode=args.mode,
        seed=args.seed_start,
        reward_version=args.reward_version,
    )
    controller = build_lqr_controller(env.core_env)
    successful_episodes = []

    try:
        if args.candidate_seeds is None:
            candidate_seeds = range(
                args.seed_start,
                args.seed_start + args.max_attempts,
            )
        else:
            candidate_seeds = args.candidate_seeds[: args.max_attempts]

        for attempt_index, seed in enumerate(candidate_seeds):
            episode = collect_lqr_episode(env, controller, seed)
            print(
                f"attempt={attempt_index + 1:03d} | seed={seed} | "
                f"success={episode['success']} | "
                f"reason={episode['termination_reason']} | "
                f"steps={len(episode['rewards'])} | "
                f"return={episode['episode_return']:.3f} | "
                f"saturation={episode['saturation_fraction']:.3f}",
                flush=True,
            )
            if episode["success"]:
                successful_episodes.append(episode)
                print(
                    "  retained successful demonstration "
                    f"{len(successful_episodes)}/{args.num_successes}",
                    flush=True,
                )
            if len(successful_episodes) == args.num_successes:
                break
    finally:
        env.close()

    if len(successful_episodes) != args.num_successes:
        partial_path = Path(output_path).with_name(
            f"{Path(output_path).stem}.partial.npz"
        )
        if successful_episodes:
            partial_dataset = combine_successful_episodes(
                successful_episodes,
                reward_version=args.reward_version,
                mode=args.mode,
            )
            save_demonstration_dataset(
                partial_path,
                partial_dataset,
                overwrite=args.overwrite,
            )
        raise RuntimeError(
            "Could not collect the requested number of successful episodes: "
            f"collected {len(successful_episodes)} after "
            f"{args.max_attempts} attempts. Partial successes were saved to "
            f"{partial_path}."
        )

    dataset = combine_successful_episodes(
        successful_episodes,
        reward_version=args.reward_version,
        mode=args.mode,
    )
    saved_path = save_demonstration_dataset(
        output_path,
        dataset,
        overwrite=args.overwrite,
    )
    metadata = _metadata_from_dataset(dataset)

    print("=" * 80)
    print("LQR replay dataset complete")
    print("=" * 80)
    print(f"Path:                {saved_path}")
    print(f"Successful episodes: {metadata['successful_episodes']}")
    print(f"Transitions:         {metadata['transition_count']}")
    print(f"Reward version:      {metadata['reward_version']}")
    print("All stored episodes are complete P2-to-P3 successes.")


if __name__ == "__main__":
    main()
