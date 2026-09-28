"""Retrain the best vanilla closing SAC with a two-second action hold.

The reference is the reward-V15, Optuna trial-16, seed-0 policy trained for
400,000 decisions with a 300,000-transition replay buffer. No LQR controller,
demonstrations, behavior cloning, safety filter, or pretrained weights are
used. The only intended change is the two-second zero-order action hold.
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


if __name__ == "__main__":
    training.main()
