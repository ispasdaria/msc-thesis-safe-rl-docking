"""Physically matched 10 Hz replication of residual-SAC Pilot E."""

from __future__ import annotations

import sys

from train_sac_closing_reference_residual_lqr_stratified import main


# Pilot E had the highest single held-out result among the completed residual
# studies (76/100 at dt=1 s). Counts representing physical interaction time
# are multiplied by ten here. train_freq is also multiplied by ten, preserving
# Pilot E's 12,500 optimizer updates over 50,000 simulated seconds.
DEFAULT_ARGUMENTS = [
    "--experiment-name",
    "sac_closing_lqr_reference_residual_pilot_e_dt0p1_500k_seed0",
    "--pilot-label",
    "E",
    "--dt",
    "0.1",
    "--total-timesteps",
    "500000",
    "--checkpoint-steps",
    "100000",
    "250000",
    "500000",
    "--source-transition-count",
    "600000",
    "--source-max-episodes",
    "12",
    "--prefill-residual-hold-steps",
    "10",
    "--max-eval-steps",
    "120000",
    "--buffer-size",
    "3000000",
    "--outer-replay-size",
    "1000000",
    "--inner-replay-size",
    "1000000",
    "--gamma",
    "0.9999899995499715",
    "--train-freq",
    "40",
    "--critic-warmup-steps",
    "100000",
    "--zero-residual-actor-weight",
    "1.0",
    "--zero-residual-actor-final-weight",
    "0.1",
    "--zero-residual-actor-anneal-end-step",
    "300000",
    "--time-scale-persistent-reward",
]


if __name__ == "__main__":
    main([*DEFAULT_ARGUMENTS, *sys.argv[1:]])
