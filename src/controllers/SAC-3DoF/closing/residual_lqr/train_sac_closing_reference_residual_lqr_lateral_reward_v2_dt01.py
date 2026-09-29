"""Train 10 Hz reference-residual SAC with the lateral-progress reward V2."""

from __future__ import annotations

import sys

from train_sac_closing_reference_residual_lqr_stratified import main


# This reward ablation retains the completed 0.1 s reference configuration.
# The retuned 10 Hz LQR, SAC optimizer, replay distribution, interaction time,
# validation seeds, and actor-prior schedule remain unchanged.  The reward:
#
# * discounts and integrates magnitude costs in physical time;
# * treats smoothness as an action-rate cost (scale = 1 s / dt);
# * gives extra potential-based progress for lateral P3 violation;
# * differentiates timeouts by their remaining worst constraint violation.
DEFAULT_ARGUMENTS = [
    "--experiment-name",
    "sac_closing_lqr_reference_residual_pilot_g_dt0p1_reward_v2_500k_seed0",
    "--pilot-label",
    "G",
    "--dt",
    "0.1",
    "--total-timesteps",
    "500000",
    "--checkpoint-steps",
    "100000",
    "250000",
    "500000",
    "--source-transition-count",
    "100000",
    "--source-max-episodes",
    "12",
    "--prefill-residual-hold-steps",
    "10",
    "--max-eval-steps",
    "120000",
    "--buffer-size",
    "650000",
    "--outer-replay-size",
    "150000",
    "--inner-replay-size",
    "50000",
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
    "--reward-design",
    "dt01_lateral_v2",
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
    "--reference-magnitude-weight",
    "0.002",
    "--reference-smoothness-weight",
    "0.001",
    "--time-scale-persistent-reward",
    "--time-consistent-smoothness-reward",
]


if __name__ == "__main__":
    main([*DEFAULT_ARGUMENTS, *sys.argv[1:]])
