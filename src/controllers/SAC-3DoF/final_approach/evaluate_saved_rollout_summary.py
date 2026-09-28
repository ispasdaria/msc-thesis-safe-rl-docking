"""Print summary statistics for a saved final-approach rollout."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROLLOUT = (
    SCRIPT_DIR
    / "results"
    / "development_experiments"
    / "reward_shaping_trials"
    / "eval_rollouts"
    / "sac_final_approach_portobs_rewardv9_100ksteps_seed0"
    / "rollout_001_seed_1.npz"
)


def inspect_rollout(path: Path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Rollout file not found: {path}\n"
            "Pass a valid .npz path as the first command-line argument."
        )

    print("Reading:", path)

    with np.load(path, allow_pickle=True) as data:
        required_keys = {"states", "actions_clipped"}
        missing_keys = sorted(required_keys.difference(data.files))
        if missing_keys:
            raise KeyError(
                f"Rollout {path} is missing required keys: {', '.join(missing_keys)}"
            )

        states = data["states"]
        actions = data["actions_clipped"]

        if "sac_rewards" in data.files:
            rewards = data["sac_rewards"]
        elif "rewards" in data.files:
            rewards = data["rewards"]
        else:
            raise KeyError(
                f"Rollout {path} contains neither 'sac_rewards' nor 'rewards'."
            )

        print("available keys:", data.files)
        print("states shape:", states.shape)
        print("rx min/max:", states[:, 0].min(), states[:, 0].max())
        print("ry min/max:", states[:, 1].min(), states[:, 1].max())
        print("vx min/max:", states[:, 2].min(), states[:, 2].max())
        print("vy min/max:", states[:, 3].min(), states[:, 3].max())
        print("psi min/max:", states[:, 4].min(), states[:, 4].max())
        print("omega min/max:", states[:, 5].min(), states[:, 5].max())
        print("actions min/max:", actions.min(axis=0), actions.max(axis=0))
        print("rewards min/max:", rewards.min(), rewards.max())

        if "port_distances" in data.files:
            print(
                "port distance min/max:",
                data["port_distances"].min(),
                data["port_distances"].max(),
            )
        if "port_speeds" in data.files:
            print(
                "port speed min/max:",
                data["port_speeds"].min(),
                data["port_speeds"].max(),
            )
        if "s_dot_values" in data.files:
            print(
                "s_dot min/max:",
                data["s_dot_values"].min(),
                data["s_dot_values"].max(),
            )


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "rollout",
        nargs="?",
        type=Path,
        default=DEFAULT_ROLLOUT,
        help=f"rollout .npz file (default: {DEFAULT_ROLLOUT})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    inspect_rollout(args.rollout)


if __name__ == "__main__":
    main()
