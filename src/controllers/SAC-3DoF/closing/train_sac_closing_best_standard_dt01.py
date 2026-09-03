"""Train the full-budget 10 Hz equivalent of the best standard closing SAC.

The underlying trainer applies the timestep-consistent V15 reward and the
Trial-16 physical-time conversions.  This preset supplies the full 4M-step
budget corresponding to the original 400k seconds of interaction.
"""

from __future__ import annotations

import sys

from train_sac_closing_dt01 import main as train_standard_dt01


DEFAULT_OUTPUT_NAME = "sac_closing_best_standard_v15dt_dt0p1_4m_seed0"


def build_argv(argv=None):
    user_args = list(sys.argv[1:] if argv is None else argv)
    defaults = [
        "--dt",
        "0.1",
        "--reward-version",
        "v15_dt",
        "--integrator",
        "rk4",
        "--action-hold-steps",
        "1",
        "--total-timesteps",
        "4000000",
        "--buffer-size",
        "3000000",
        "--learning-starts",
        "100000",
        "--train-freq",
        "160",
        "--gradient-steps",
        "8",
        "--eval-freq",
        "500000",
        "--max-time",
        "12000",
    ]
    if "--output-name" not in user_args:
        defaults.extend(("--output-name", DEFAULT_OUTPUT_NAME))
    return [*defaults, *user_args]


def main(argv=None):
    return train_standard_dt01(build_argv(argv))


if __name__ == "__main__":
    main()
