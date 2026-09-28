# Optuna Trial 12 - 300k Steps, 300k Replay Buffer

This campaign applies the selected Optuna trial-12 SAC hyperparameters to reward version 9 with a 300,000-step training budget and a 300,000-transition replay buffer.

Training seeds 0 through 19 are included. Each seed has a final checkpoint and an evaluation-selected best checkpoint, and every model has a matching VecNormalize file. The campaign was used to compare replay-buffer scaling and training-seed robustness. It was not retained as the final controller family.

The round-3 seed-19 failure audit evaluates one checkpoint from this campaign.
