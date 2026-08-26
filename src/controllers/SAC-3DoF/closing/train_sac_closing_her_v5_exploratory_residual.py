"""Run the V5 exploratory-residual SAC+HER pilot.

V5 retains the exact frozen V15 controller in the control loop while adding
small, temporally correlated residual probes to the HER prefill.  Complete
safety-terminal episodes are retained so the critic observes both useful
action variation and the edge of the safe set.
"""

from __future__ import annotations

import sys

from train_sac_closing_her_v4_residual_v15 import main


V5_DEFAULT_ARGS = [
    "--experiment-name",
    "sac_closing_goal_her_v5_exploratory_residual_pilot50k_seed0",
    "--total-timesteps",
    "50000",
    "--checkpoint-steps",
    "10000",
    "25000",
    "50000",
    "--source-transition-count",
    "60000",
    "--source-max-episodes",
    "20",
    "--residual-scale",
    "0.10",
    "--prefill-residual-std",
    "0.35",
    "--prefill-residual-clip",
    "0.75",
    "--prefill-residual-hold-steps",
    "20",
    "--prefill-zero-episode-fraction",
    "0.25",
    "--prefill-random-seed",
    "5000",
    "--retain-failure-prefill",
    "--initial-log-std",
    "-2.0",
    "--batch-size",
    "256",
    "--train-freq",
    "4",
    "--gradient-steps",
    "2",
    "--ent-coef",
    "auto_0.05",
]


if __name__ == "__main__":
    # User arguments come last, so ordinary value options can override the
    # pilot defaults without duplicating this trainer.
    main([*V5_DEFAULT_ARGS, *sys.argv[1:]])
