# Retained Optuna Trial 12 Campaign - 600k Steps

This campaign uses docking-port observations, reward version 9, the selected Optuna trial-12 hyperparameters, a 600,000-step training budget, and a 600,000-transition replay buffer.

Training seeds 0 through 19 are included. Each seed has a final checkpoint and an evaluation-selected best checkpoint, with matching VecNormalize files. The retained round-5 controller is the final seed-11 checkpoint:

`sac_final_approach_portobs_rewardv9_600ksteps_seed11_optuna12_buffer600k`

The two seed-18 files ending in `_previous_grouped_copy.zip` are archival duplicate copies and do not have same-stem VecNormalize files. The corresponding standard seed-18 checkpoints and normalization files remain present. Other campaign checkpoints support policy-selection comparisons.
