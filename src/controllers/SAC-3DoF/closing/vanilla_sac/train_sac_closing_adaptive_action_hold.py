"""Train vanilla V15 SAC with adaptive far/near control cadence.

The policy holds actions for at most two one-second simulator steps while it
is more than 400 m from P3. Control returns after one second near P3, while in
a safety warning, or immediately after crossing into either condition.

This remains a from-scratch vanilla SAC experiment. It uses the same Optuna
trial-16 hyperparameters, seed, budget, and replay-buffer size as the strongest
original V15 vanilla baseline. Checkpoints are selected by task outcome and
safety before target quality and shaped return.
"""

import sys
from pathlib import Path


CLOSING_DIR = Path(__file__).resolve().parent.parent
if str(CLOSING_DIR) not in sys.path:
    sys.path.insert(0, str(CLOSING_DIR))

import train_sac_closing_multiseed as training


training.DEFAULT_TRAINING_SEEDS = [0]
training.DEFAULT_TOTAL_TIMESTEPS = 400_000
training.DEFAULT_REWARD_VERSION = "v15"
training.DEFAULT_BUFFER_SIZE = 300_000
training.DEFAULT_ACTION_HOLD_SECONDS = 2
training.DEFAULT_ADAPTIVE_ACTION_HOLD_DISTANCE_M = 400.0
training.DEFAULT_CHECKPOINT_SELECTION = "task_outcome"


if __name__ == "__main__":
    training.main()
