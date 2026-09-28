"""Train the 10 Hz equivalent of the 91%-success BC-constrained SAC.

All physical-time schedules are derived by the shared trainer from the 1 Hz
reference.  Explicit command-line overrides remain available for screened
10 Hz optimizer variants, while the timing and reward contract is fixed here.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_sac_closing_lqr_bc_constrained import main as train_bc_constrained

DEFAULT_BC_DEMONSTRATIONS = (
    SCRIPT_DIR
    / "demonstrations"
    / "lqr_closing_training_rewardv15_dt_dt0p1_bc_stride10_78episodes.npz"
)
DEFAULT_REPLAY_DEMONSTRATIONS = (
    SCRIPT_DIR
    / "demonstrations"
    / "lqr_closing_training_rewardv15_dt_dt0p1_replay_20episodes.npz"
)


def build_argv(argv=None):
    user_args = list(sys.argv[1:] if argv is None else argv)
    fixed = [
        "--dt",
        "0.1",
        "--max-time",
        "12000",
        "--action-hold-steps",
        "1",
        "--reward-version",
        "v15_dt",
        "--bc-action-loss-scaling",
        "mean_abs",
    ]
    has_shared = "--demonstrations" in user_args
    if not has_shared and "--bc-demonstrations" not in user_args:
        fixed.extend(("--bc-demonstrations", str(DEFAULT_BC_DEMONSTRATIONS)))
    if not has_shared and "--replay-demonstrations" not in user_args:
        fixed.extend(
            ("--replay-demonstrations", str(DEFAULT_REPLAY_DEMONSTRATIONS))
        )
    return [*fixed, *user_args]


def main(argv=None):
    return train_bc_constrained(build_argv(argv))


if __name__ == "__main__":
    main()
