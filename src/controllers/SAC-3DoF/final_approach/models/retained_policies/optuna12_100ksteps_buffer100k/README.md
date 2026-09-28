# Retained Optuna Trial 12 Campaign - 100k Steps

This campaign uses docking-port observations, reward version 9, the selected Optuna trial-12 hyperparameters, a 100,000-step training budget, and a 100,000-transition replay buffer.

Training seeds 0 through 19 are included. Each seed has a final checkpoint and an evaluation-selected best checkpoint, with matching VecNormalize files. The retained round-1 controller is the final seed-12 checkpoint:

`sac_final_approach_portobs_rewardv9_100ksteps_seed12_optuna12`

Other checkpoints remain available for training-seed comparison and policy-selection traceability.
