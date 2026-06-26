"""
Train multiple SAC final-approach agents with the Optuna trial-12 hyperparameters.

Each training seed is saved as a separate model/VecNormalize pair so training
seed robustness can be evaluated afterward.
"""

import argparse
import csv
import os
import sys
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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "learning_curves")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_training_env(seed):
    env = Docking3DoFGymEnv(
        phase="final",
        mode="training",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
    )
    return Monitor(env)


class LearningCurveCallback(BaseCallback):
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

            for episode, timestep, episode_return, rolling_return, reason in zip(
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
                        reason,
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


def model_name_for_seed(seed, total_timesteps):
    steps_label = f"{total_timesteps // 1000}ksteps"
    return f"sac_final_approach_portobs_rewardv9_{steps_label}_seed{seed}_optuna12"


def train_one_seed(seed, total_timesteps, device, progress_bar, overwrite):
    model_name = model_name_for_seed(seed, total_timesteps)

    model_path = os.path.join(MODEL_DIR, model_name)
    model_zip_path = model_path + ".zip"
    vecnorm_path = os.path.join(MODEL_DIR, f"{model_name}_vecnormalize.pkl")
    learning_curve_dir = os.path.join(RESULTS_DIR, model_name)
    learning_curve_csv = os.path.join(learning_curve_dir, "learning_curve.csv")
    learning_curve_plot = os.path.join(learning_curve_dir, "learning_curve.png")

    if not overwrite and os.path.exists(model_zip_path) and os.path.exists(vecnorm_path):
        print("=" * 80)
        print(f"Skipping seed {seed}: existing model and VecNormalize found.")
        print(f"Model:   {model_zip_path}")
        print(f"VecNorm: {vecnorm_path}")
        print("Use --overwrite to retrain this seed.")
        return

    os.makedirs(learning_curve_dir, exist_ok=True)

    print("=" * 80)
    print("SAC final-approach multi-seed training")
    print("=" * 80)
    print(f"Training seed: {seed}")
    print(f"Model path:    {model_path}")
    print(f"VecNorm path:  {vecnorm_path}")
    print(f"Total steps:   {total_timesteps}")
    print("=" * 80)

    train_env = DummyVecEnv([lambda: make_training_env(seed=seed)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.98,
    )

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=0.0003395686575052769,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        train_freq=8,
        gradient_steps=8,
        ent_coef="auto_0.1",
        target_update_interval=1,
        verbose=1,
        seed=seed,
        tensorboard_log=LOG_DIR,
        device=device,
    )

    callback = LearningCurveCallback(
        csv_path=learning_curve_csv,
        plot_path=learning_curve_plot,
        rolling_window=20,
    )

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=model_name,
        progress_bar=progress_bar,
        callback=callback,
    )

    model.save(model_path)
    train_env.save(vecnorm_path)
    train_env.close()

    print("=" * 80)
    print(f"Training completed for seed {seed}.")
    print(f"Saved model:   {model_zip_path}")
    print(f"Saved VecNorm: {vecnorm_path}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SAC final-approach agents over multiple training seeds."
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=9)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain seeds even when model files already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be greater than or equal to --seed-start.")

    for seed in range(args.seed_start, args.seed_end + 1):
        train_one_seed(
            seed=seed,
            total_timesteps=args.total_timesteps,
            device=args.device,
            progress_bar=args.progress_bar,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
