# Optuna Trial 12 - 300k Steps, 100k Replay Buffer

This campaign applies the selected Optuna trial-12 SAC hyperparameters to reward version 9 with a 300,000-step training budget and a 100,000-transition replay buffer.

Training seeds 0 through 19 are included. Each seed has a final checkpoint and an evaluation-selected best checkpoint, and every model has a matching VecNormalize file. The campaign supports training-seed and replay-buffer comparisons but was not retained as the final controller family.

Corresponding logs and evaluation results use the same experiment-family name.
