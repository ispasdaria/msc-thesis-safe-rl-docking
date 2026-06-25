"""
Train SAC for the 3-DoF docking final-approach phase.

This script only trains the SAC agent.
"""

import os
import sys
import csv
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
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
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "learning_curves")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------

def make_training_env(seed=0):
    """
    Creates one monitored final-approach training environment.

    The environment returns raw port-relative physical observations.
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


# ---------------------------------------------------------------------
# Learning-curve callback
# ---------------------------------------------------------------------

class LearningCurveCallback(BaseCallback):
    """
    Records one point per completed episode during training.

    x-axis: total training timesteps when the episode ended.
    y-axis: raw episode return reported by Monitor.
    """

    def __init__(self, csv_path, plot_path, rolling_window=20):
        super().__init__()

        self.csv_path = csv_path
        self.plot_path = plot_path
        self.rolling_window = rolling_window

        self.timesteps = []
        self.episode_returns = []
        self.rolling_returns = []
        self.termination_reasons = []
        self._recent_returns = deque(maxlen=rolling_window)

    def _on_step(self):
        for info in self.locals.get("infos", []):
            episode_info = info.get("episode")

            if episode_info is None:
                continue

            episode_return = float(episode_info["r"])
            self._recent_returns.append(episode_return)

            self.timesteps.append(self.num_timesteps)
            self.episode_returns.append(episode_return)
            self.rolling_returns.append(float(np.mean(self._recent_returns)))
            self.termination_reasons.append(
                info.get("termination_reason", episode_info.get("termination_reason", ""))
            )

        return True

    def _on_training_end(self):
        self._save_csv()
        self._save_plot()

    def _save_csv(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "timestep",
                    "episode_return",
                    "rolling_mean_return",
                    "termination_reason",
                ]
            )

            for episode, timestep, episode_return, rolling_return, termination_reason in zip(
                np.arange(1, len(self.episode_returns) + 1),
                self.timesteps,
                self.episode_returns,
                self.rolling_returns,
                self.termination_reasons,
            ):
                writer.writerow(
                    [
                        int(episode),
                        int(timestep),
                        f"{episode_return:.10f}",
                        f"{rolling_return:.10f}",
                        termination_reason,
                    ]
                )

    def _save_plot(self):
        if not self.episode_returns:
            return

        plt.figure(figsize=(10, 6))
        plt.plot(
            self.timesteps,
            self.episode_returns,
            color="tab:blue",
            alpha=0.35,
            linewidth=1.0,
            label="Episode return",
        )
        plt.plot(
            self.timesteps,
            self.rolling_returns,
            color="tab:orange",
            linewidth=2.0,
            label=f"Rolling mean ({self.rolling_window} episodes)",
        )
        plt.xlabel("Training timesteps")
        plt.ylabel("Episode return")
        plt.title("SAC Final-Approach Learning Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=200)
        plt.close()


def main():
    # -----------------------------------------------------------------
    # Training configuration
    # -----------------------------------------------------------------

    seed = 0
    total_timesteps = 100_000

    model_name = "sac_final_approach_portobs_rewardv9_100ksteps_seed0"

    model_path = os.path.join(MODEL_DIR, model_name)
    vecnorm_path = os.path.join(MODEL_DIR, f"{model_name}_vecnormalize.pkl")
    learning_curve_dir = os.path.join(RESULTS_DIR, model_name)
    learning_curve_csv = os.path.join(learning_curve_dir, "learning_curve.csv")
    learning_curve_plot = os.path.join(learning_curve_dir, "learning_curve.png")

    os.makedirs(learning_curve_dir, exist_ok=True)

    print("=" * 80)
    print("SAC final-approach training")
    print("=" * 80)
    print(f"Repository root: {REPO_ROOT}")
    print(f"Model path:      {model_path}")
    print(f"VecNorm path:    {vecnorm_path}")
    print(f"Learning curve:  {learning_curve_plot}")
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

    learning_curve_callback = LearningCurveCallback(
        csv_path=learning_curve_csv,
        plot_path=learning_curve_plot,
        rolling_window=20,
    )

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=model_name,
        progress_bar=True,
        callback=learning_curve_callback,
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
    print(f"Saved curve CSV:  {learning_curve_csv}")
    print(f"Saved curve plot: {learning_curve_plot}")
    print("=" * 80)


if __name__ == "__main__":
    main()
