"""Pilot I: Pilot H with terminal residual-authority fading."""

from __future__ import annotations

import sys

from train_sac_closing_reference_residual_lqr_pilot_h_dt01_lateral_curriculum import (
    DEFAULT_ARGUMENTS as PILOT_H_ARGUMENTS,
)
from train_sac_closing_reference_residual_lqr_terminal import main


# This is a controlled follow-up to Pilot H. Repeated scalar arguments are
# intentional: argparse uses the last value, retaining every other Pilot H
# hyperparameter exactly while changing the label/name and enabling the gate.
DEFAULT_ARGUMENTS = [
    *PILOT_H_ARGUMENTS,
    "--experiment-name",
    "sac_closing_lqr_reference_residual_pilot_i_dt0p1_terminal_fade_500k_seed0",
    "--pilot-label",
    "I",
    "--terminal-residual-fade-full-ratio",
    "1.2",
    "--terminal-residual-fade-zero-ratio",
    "1.0",
]


if __name__ == "__main__":
    main([*DEFAULT_ARGUMENTS, *sys.argv[1:]])
