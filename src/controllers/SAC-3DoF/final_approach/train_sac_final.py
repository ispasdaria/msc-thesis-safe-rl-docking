"""
Train SAC for the 3-DoF docking final-approach phase.

This script only trains the SAC agent.
"""

import os
import sys

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from envs.docking3dof.gym_env import Docking3DoFGymEnv


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------

def make_training_env(seed=0):
    """
    Creates one monitored final-approach training environment.

    The environment returns raw physical observations.
    VecNormalize normalizes observations and rewards externally.
    """

    env = Docking3DoFGymEnv(
        phase="final",
        mode="training",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
    )

    env = Monitor(env)
    return env


def main():
    # -----------------------------------------------------------------
    # Training configuration
    # -----------------------------------------------------------------

    seed = 0
    total_timesteps = 50_000

    model_name = "sac_final_approach_rewardv5_50ksteps_seed0"

    model_path = os.path.join(MODEL_DIR, model_name)
    vecnorm_path = os.path.join(MODEL_DIR, f"{model_name}_vecnormalize.pkl")

    print("=" * 80)
    print("SAC final-approach training")
    print("=" * 80)
    print(f"Repository root: {REPO_ROOT}")
    print(f"Model path:      {model_path}")
    print(f"VecNorm path:    {vecnorm_path}")
    print(f"Total steps:     {total_timesteps}")
    print("=" * 80)

    # -----------------------------------------------------------------
    # Create training environment
    # -----------------------------------------------------------------

    train_env = DummyVecEnv([
        lambda: make_training_env(seed=seed)
    ])

    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    # -----------------------------------------------------------------
    # SAC model
    # -----------------------------------------------------------------

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        verbose=1,
        seed=seed,
        tensorboard_log=LOG_DIR,
        device="auto",
    )

    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=model_name,
        progress_bar=True,
    )

    # -----------------------------------------------------------------
    # Save final model and normalization statistics
    # -----------------------------------------------------------------

    model.save(model_path)
    train_env.save(vecnorm_path)

    train_env.close()

    print("=" * 80)
    print("Training completed.")
    print(f"Saved model to:   {model_path}.zip")
    print(f"Saved VecNorm to: {vecnorm_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()