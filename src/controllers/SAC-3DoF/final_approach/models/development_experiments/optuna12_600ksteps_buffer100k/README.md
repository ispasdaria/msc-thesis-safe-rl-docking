# Optuna Trial 12 - 600k Steps, 100k Replay Buffer

This campaign applies the selected Optuna trial-12 SAC hyperparameters to reward version 9 with a 600,000-step training budget while retaining a 100,000-transition replay buffer.

Training seeds 0 through 19 are included. Each seed has a final checkpoint and an evaluation-selected best checkpoint, and every model has a matching VecNormalize file. The campaign isolates the effect of extending training without increasing replay-buffer capacity. It was not retained as the final controller family.
