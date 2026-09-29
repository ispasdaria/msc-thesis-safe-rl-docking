"""Train 200k-step direct residual SAC with balanced lateral prefill."""

from __future__ import annotations

import sys

from train_sac_closing_her_residual_lqr import main


DEFAULT_ARGUMENTS = [
    "--experiment-name",
    "sac_closing_goal_her_direct_residual_dt0p1_lateral_balanced_200k_seed0",
    "--reward-version",
    "lqr_residual_lateral",
    "--total-timesteps",
    "200000",
    "--checkpoint-steps",
    "120000",
    "200000",
    "--source-transition-count",
    "200000",
    "--source-seed-start",
    "10",
    "--source-max-episodes",
    "12",
    "--prefill-min-zero-episodes",
    "1",
    "--prefill-min-exploratory-episodes",
    "1",
    "--prefill-zero-episode-fraction",
    "0.5",
    "--prefill-residual-std",
    "0.25",
    "--prefill-residual-clip",
    "0.75",
    "--prefill-residual-hold-steps",
    "10",
    "--buffer-size",
    "500000",
    "--critic-warmup-steps",
    "30000",
    "--zero-residual-actor-weight",
    "1.0",
    "--zero-residual-actor-final-weight",
    "0.1",
    "--zero-residual-actor-anneal-end-step",
    "150000",
    "--actor-learning-rate-scale",
    "0.25",
    "--goal-component-weights",
    "1.0",
    "4.0",
    "1.0",
    "2.0",
    "1.0",
    "1.0",
    "--worst-component-weight",
    "0.75",
    "--progress-weight",
    "2.0",
    "--lateral-constraint-progress-weight",
    "4.0",
    "--success-bonus",
    "5.0",
    "--failure-penalty",
    "25.0",
    "--timeout-penalty",
    "2.0",
    "--timeout-violation-weight",
    "2.0",
]


if __name__ == "__main__":
    main([*DEFAULT_ARGUMENTS, *sys.argv[1:]])
