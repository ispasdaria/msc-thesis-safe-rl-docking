"""
Train SAC for the 3-DoF docking closing phase.

This script only trains the SAC agent.
"""

import argparse
import csv
import json
import os
import sys
from collections import deque

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
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

MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "reward_shaping_trials")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "reward_shaping_trials")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "reward_shaping_trials")
EPISODE_RETURN_LEARNING_CURVES_DIR = os.path.join(
    RESULTS_DIR,
    "episode_return_learning_curves",
)
EVALUATION_LEARNING_CURVES_DIR = os.path.join(
    RESULTS_DIR,
    "evaluation_learning_curves",
)

CLOSING_REWARD_VERSION = "v15"

# Best configuration from trial 16 of the first 20-trial Optuna study,
# ``sac_closing_p3augobs_rewardv15_optuna_v2``.
TRIAL_16_PARAMS = {
    "learning_rate": 0.00026276728418436003,
    "batch_size": 512,
    "tau": 0.01,
    "gamma": 0.99,
    "learning_starts": 10_000,
    "train_freq": 16,
    "gradient_steps": 8,
    "ent_coef": "auto_0.05",
}

# Configuration used by the Reward-v15 run immediately before Optuna.  Keep
# this explicit so a matched-budget rerun changes only the training budget.
PRE_OPTUNA_PARAMS = {
    "learning_rate": 0.0003395686575052769,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.98,
    "learning_starts": 1_000,
    "train_freq": 8,
    "gradient_steps": 8,
    "ent_coef": "auto_0.1",
}

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a SAC closing-phase agent."
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used by Stable-Baselines3/PyTorch: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--configuration",
        choices=["pre-optuna", "post-optuna"],
        default="post-optuna",
        help="SAC hyperparameter configuration to train.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=400_000,
        help="Training budget in environment steps (default: 400000).",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="Episode horizon in seconds (closing default when omitted).",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Unique artifact name. Existing outputs are never overwritten.",
    )
    parser.add_argument("--initialize-model", default=None)
    parser.add_argument("--initialize-vecnormalize", default=None)
    return parser.parse_args()


def validate_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested with --device cuda, but "
            "torch.cuda.is_available() is False."
        )


def print_device_info(device):
    print(f"Requested device: {device}")
    print(f"CUDA available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device:      {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------

def make_training_env(seed=12, max_time=None):
    """
    Creates one monitored closing training environment.

    The environment returns the augmented docking-port safety state and the
    physical state error relative to the sampled P3 target. VecNormalize
    normalizes observations and rewards externally.
    """

    env = Docking3DoFGymEnv(
        phase="closing",
        mode="training",
        dt=1.0,
        max_time=max_time,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
        closing_reward_version=CLOSING_REWARD_VERSION,
    )

    env = Monitor(env)
    return env


def make_eval_env(seed=0, max_time=None):
    """
    Creates one raw closing evaluation environment.

    The policy is evaluated deterministically on the testing distribution. The
    observations are normalized manually with the current VecNormalize stats
    from the training environment.
    """

    env = Docking3DoFGymEnv(
        phase="closing",
        mode="testing",
        dt=1.0,
        max_time=max_time,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
        closing_reward_version=CLOSING_REWARD_VERSION,
    )

    return env


def evaluate_deterministic_policy(
    model,
    eval_seeds,
    max_steps=12000,
    eval_env_factory=None,
):
    """
    Evaluates the current deterministic policy with the live VecNormalize stats.
    """

    if eval_env_factory is None:
        eval_env_factory = make_eval_env

    vecnorm = model.get_vec_normalize_env()
    if vecnorm is None:
        raise RuntimeError("VecNormalize environment is required for evaluation.")

    old_training = vecnorm.training
    old_norm_reward = vecnorm.norm_reward

    vecnorm.training = False
    vecnorm.norm_reward = False

    returns = []
    lengths = []

    try:
        for seed in eval_seeds:
            env = eval_env_factory(seed=seed)
            obs, _ = env.reset(seed=seed)

            total_reward = 0.0
            episode_length = 0

            for _ in range(max_steps):
                norm_obs = vecnorm.normalize_obs(obs.reshape(1, -1))
                action, _ = model.predict(norm_obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action[0])

                total_reward += float(reward)
                episode_length += 1

                if terminated or truncated:
                    break

            env.close()
            returns.append(total_reward)
            lengths.append(episode_length)
    finally:
        vecnorm.training = old_training
        vecnorm.norm_reward = old_norm_reward

    return returns, lengths


# ---------------------------------------------------------------------
# Learning-curve helpers and callback
# ---------------------------------------------------------------------

def moving_average(values, window):
    if len(values) < window:
        return np.array([])

    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")


class LearningCurveCallback(BaseCallback):
    """
    Records one point per completed episode during training.

    x-axis: completed training episode number.
    y-axis: raw episode return reported by Monitor.
    """

    def __init__(self, csv_path, plot_path, model_name, moving_average_window=10):
        super().__init__()

        self.csv_path = csv_path
        self.plot_path = plot_path
        self.model_name = model_name
        self.moving_average_window = moving_average_window

        self.episodes = []
        self.timesteps = []
        self.episode_returns = []
        self.rolling_returns = []
        self.termination_reasons = []
        self.best_checkpoint_episode = None
        self.best_checkpoint_timestep = None
        self.best_checkpoint_score = None

    def set_best_checkpoint(self, episode, timestep, score):
        self.best_checkpoint_episode = int(episode)
        self.best_checkpoint_timestep = int(timestep)
        self.best_checkpoint_score = float(score)

    def _on_step(self):
        for info in self.locals.get("infos", []):
            episode_info = info.get("episode")

            if episode_info is None:
                continue

            episode_return = float(episode_info["r"])

            self.episodes.append(len(self.episodes) + 1)
            self.timesteps.append(self.num_timesteps)
            self.episode_returns.append(episode_return)
            self.rolling_returns.append(
                float(np.mean(self.episode_returns[-self.moving_average_window :]))
            )
            self.termination_reasons.append(
                info.get("termination_reason", episode_info.get("termination_reason", ""))
            )

        return True

    def _on_training_end(self):
        self.save_outputs()

    def save_outputs(self):
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
                    "is_best_checkpoint",
                ]
            )

            for episode, timestep, episode_return, rolling_return, termination_reason in zip(
                self.episodes,
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
                        int(episode == self.best_checkpoint_episode),
                    ]
                )

    def _save_plot(self):
        if not self.episode_returns:
            return

        averaged_returns = moving_average(
            self.episode_returns,
            self.moving_average_window,
        )
        averaged_episodes = self.episodes[self.moving_average_window - 1 :]

        plt.figure(figsize=(10, 6))
        plt.plot(
            self.episodes,
            self.episode_returns,
            color="tab:blue",
            linewidth=1.0,
            alpha=0.8,
            label="Episode return",
        )
        if len(averaged_returns) > 0:
            plt.plot(
                averaged_episodes,
                averaged_returns,
                color="tab:orange",
                linewidth=2.0,
                label=f"{self.moving_average_window}-episode moving average",
            )
        if self.best_checkpoint_episode is not None:
            plt.axvline(
                self.best_checkpoint_episode,
                color="tab:red",
                linestyle="--",
                linewidth=1.6,
                label="Best deterministic checkpoint",
            )
            if self.best_checkpoint_episode in self.episodes:
                best_index = self.episodes.index(self.best_checkpoint_episode)
                plt.scatter(
                    [self.best_checkpoint_episode],
                    [self.episode_returns[best_index]],
                    color="tab:red",
                    s=45,
                    zorder=5,
                )
        plt.xlabel("Episode number")
        plt.ylabel("Total reward per episode")
        plt.title(f"Total Reward per Training Episode\n{self.model_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=200)
        plt.close()


class BestModelCallback(BaseCallback):
    """
    Saves the best deterministic-evaluation checkpoint seen so far.

    The score is the mean return over fixed deterministic evaluation episodes.
    The checkpoint is marked on the training learning curve at the number of
    completed training episodes when the evaluation was run.
    """

    def __init__(
        self,
        best_model_path,
        best_vecnorm_path,
        eval_seeds,
        eval_freq=50_000,
        max_steps=12000,
        eval_env_factory=None,
    ):
        super().__init__()
        self.best_model_path = best_model_path
        self.best_vecnorm_path = best_vecnorm_path
        self.eval_seeds = list(eval_seeds)
        self.eval_freq = eval_freq
        self.max_steps = max_steps
        self.eval_env_factory = eval_env_factory

        self.best_mean_return = -np.inf
        self.completed_episodes = 0
        self.best_episode = None
        self.best_timestep = None
        self.last_eval_timestep = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            episode_info = info.get("episode")

            if episode_info is None:
                continue

            self.completed_episodes += 1

        if self.num_timesteps - self.last_eval_timestep < self.eval_freq:
            return True

        self.last_eval_timestep = self.num_timesteps
        returns, lengths = evaluate_deterministic_policy(
            self.model,
            eval_seeds=self.eval_seeds,
            max_steps=self.max_steps,
            eval_env_factory=self.eval_env_factory,
        )
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        mean_length = float(np.mean(lengths))

        print(
            "Best-checkpoint evaluation "
            f"episode={self.completed_episodes}, "
            f"timestep={self.num_timesteps}, "
            f"mean_return={mean_return:.6f}, "
            f"std_return={std_return:.6f}, "
            f"mean_length={mean_length:.2f}"
        )

        if mean_return <= self.best_mean_return:
            return True

        self.best_mean_return = mean_return
        self.best_episode = self.completed_episodes
        self.best_timestep = self.num_timesteps
        self.model.save(self.best_model_path)
        self.training_env.save(self.best_vecnorm_path)

        print(
            "Saved new best deterministic-eval checkpoint "
            f"to {self.best_model_path}.zip"
        )

        return True

    def get_best_checkpoint(self):
        if self.best_episode is None:
            return None

        return {
            "episode": self.best_episode,
            "timestep": self.best_timestep,
            "score": self.best_mean_return,
        }


class PeriodicEvalCallback(BaseCallback):
    """
    Evaluates the current deterministic policy every N completed episodes.

    The callback saves:
      - a CSV with mean/std return and per-seed returns,
      - an SB3-style evaluations.npz file,
      - a mean-return plot with a standard-deviation band.
    """

    def __init__(
        self,
        csv_path,
        npz_path,
        plot_path,
        model_name,
        eval_seeds,
        eval_every_episodes=10,
        max_steps=12000,
        eval_env_factory=None,
    ):
        super().__init__()

        self.csv_path = csv_path
        self.npz_path = npz_path
        self.plot_path = plot_path
        self.model_name = model_name
        self.eval_seeds = list(eval_seeds)
        self.eval_every_episodes = eval_every_episodes
        self.max_steps = max_steps
        self.eval_env_factory = eval_env_factory

        self.completed_episodes = 0

        self.eval_episodes = []
        self.eval_timesteps = []
        self.eval_results = []
        self.eval_lengths = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            episode_info = info.get("episode")

            if episode_info is None:
                continue

            self.completed_episodes += 1

            if self.completed_episodes % self.eval_every_episodes != 0:
                continue

            returns, lengths = self._evaluate_current_policy()
            mean_return = float(np.mean(returns))
            std_return = float(np.std(returns))

            self.eval_episodes.append(self.completed_episodes)
            self.eval_timesteps.append(self.num_timesteps)
            self.eval_results.append(returns)
            self.eval_lengths.append(lengths)

            print(
                "Periodic evaluation "
                f"episode={self.completed_episodes}, "
                f"timestep={self.num_timesteps}, "
                f"mean_return={mean_return:.6f}, "
                f"std_return={std_return:.6f}"
            )

            self._save_csv()
            self._save_npz()
            self._save_plot()

        return True

    def _on_training_end(self):
        self._save_csv()
        self._save_npz()
        self._save_plot()

    def _evaluate_current_policy(self):
        return evaluate_deterministic_policy(
            self.model,
            eval_seeds=self.eval_seeds,
            max_steps=self.max_steps,
            eval_env_factory=self.eval_env_factory,
        )

    def _save_csv(self):
        if not self.eval_results:
            return

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = [
                "episode",
                "timestep",
                "mean_return",
                "std_return",
                "mean_length",
            ]
            header.extend(f"seed_{seed}_return" for seed in self.eval_seeds)
            header.extend(f"seed_{seed}_length" for seed in self.eval_seeds)
            writer.writerow(header)

            for episode, timestep, returns, lengths in zip(
                self.eval_episodes,
                self.eval_timesteps,
                self.eval_results,
                self.eval_lengths,
            ):
                row = [
                    int(episode),
                    int(timestep),
                    f"{np.mean(returns):.10f}",
                    f"{np.std(returns):.10f}",
                    f"{np.mean(lengths):.10f}",
                ]
                row.extend(f"{value:.10f}" for value in returns)
                row.extend(int(value) for value in lengths)
                writer.writerow(row)

    def _save_npz(self):
        if not self.eval_results:
            return

        results = np.array(self.eval_results, dtype=np.float64)
        mean_returns = np.mean(results, axis=1)
        best_eval_index = int(np.argmax(mean_returns))

        np.savez(
            self.npz_path,
            timesteps=np.array(self.eval_timesteps, dtype=np.int64),
            episodes=np.array(self.eval_episodes, dtype=np.int64),
            results=results,
            ep_lengths=np.array(self.eval_lengths, dtype=np.int64),
            eval_seeds=np.array(self.eval_seeds, dtype=np.int64),
            best_eval_index=np.array(best_eval_index, dtype=np.int64),
            best_episode=np.array(
                self.eval_episodes[best_eval_index],
                dtype=np.int64,
            ),
            best_timestep=np.array(
                self.eval_timesteps[best_eval_index],
                dtype=np.int64,
            ),
            best_mean_return=np.array(
                mean_returns[best_eval_index],
                dtype=np.float64,
            ),
        )

    def get_best_checkpoint(self):
        if not self.eval_results:
            return None

        results = np.array(self.eval_results, dtype=np.float64)
        mean_returns = np.mean(results, axis=1)
        best_eval_index = int(np.argmax(mean_returns))

        return {
            "episode": self.eval_episodes[best_eval_index],
            "timestep": self.eval_timesteps[best_eval_index],
            "mean_return": mean_returns[best_eval_index],
        }

    def _save_plot(self):
        if not self.eval_results:
            return

        episodes = np.array(self.eval_episodes)
        results = np.array(self.eval_results, dtype=np.float64)
        mean_returns = np.mean(results, axis=1)
        std_returns = np.std(results, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(
            episodes,
            mean_returns,
            color="tab:green",
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="Mean evaluation return",
        )
        plt.fill_between(
            episodes,
            mean_returns - std_returns,
            mean_returns + std_returns,
            color="tab:green",
            alpha=0.2,
            label="1 std across seeds",
        )
        plt.xlabel("Completed training episodes")
        plt.ylabel("Mean total evaluation reward")
        plt.title(f"Deterministic Evaluation Learning Curve\n{self.model_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=200)
        plt.close()


def main():
    args = parse_args()
    validate_device(args.device)
    if args.max_time is not None and args.max_time <= 0:
        raise ValueError("--max-time must be positive")
    initialized = args.initialize_model is not None or args.initialize_vecnormalize is not None
    if initialized and not (args.initialize_model and args.initialize_vecnormalize):
        raise ValueError(
            "--initialize-model and --initialize-vecnormalize must be supplied together"
        )
    for source in (args.initialize_model, args.initialize_vecnormalize):
        if source is not None and not os.path.isfile(source):
            raise FileNotFoundError(source)

    # -----------------------------------------------------------------
    # Training configuration
    # -----------------------------------------------------------------

    seed = 0
    total_timesteps = args.total_timesteps
    if total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    params = PRE_OPTUNA_PARAMS if args.configuration == "pre-optuna" else TRIAL_16_PARAMS
    steps_label = f"{total_timesteps // 1000}ksteps"
    if args.experiment_name is not None:
        model_name = args.experiment_name
    elif args.configuration == "pre-optuna":
        model_name = (
            f"sac_closing_p3augobs_rewardv15_preoptuna_{steps_label}_"
            "seed0_buffer300k"
        )
    else:
        model_name = (
            f"sac_closing_p3augobs_rewardv15_optuna_trial16_{steps_label}_"
            "seed0_buffer300k"
        )

    model_path = os.path.join(MODEL_DIR, model_name)
    best_model_path = os.path.join(MODEL_DIR, f"{model_name}_best")
    vecnorm_path = os.path.join(MODEL_DIR, f"{model_name}_vecnormalize.pkl")
    best_vecnorm_path = os.path.join(
        MODEL_DIR, f"{model_name}_best_vecnormalize.pkl"
    )
    learning_curve_dir = os.path.join(EPISODE_RETURN_LEARNING_CURVES_DIR, model_name)
    learning_curve_csv = os.path.join(learning_curve_dir, "learning_curve.csv")
    learning_curve_plot = os.path.join(
        EPISODE_RETURN_LEARNING_CURVES_DIR,
        f"{model_name}.png",
    )
    evaluation_curve_dir = os.path.join(EVALUATION_LEARNING_CURVES_DIR, model_name)
    evaluation_curve_csv = os.path.join(
        evaluation_curve_dir,
        "evaluation_learning_curve.csv",
    )
    evaluations_npz = os.path.join(evaluation_curve_dir, "evaluations.npz")
    evaluation_curve_plot = os.path.join(
        EVALUATION_LEARNING_CURVES_DIR,
        f"{model_name}.png",
    )

    protected_outputs = [
        model_path + ".zip", best_model_path + ".zip", vecnorm_path,
        best_vecnorm_path, learning_curve_csv, learning_curve_plot,
        evaluation_curve_csv, evaluations_npz, evaluation_curve_plot,
    ]
    if any(os.path.exists(path) for path in protected_outputs):
        raise FileExistsError(
            "Refusing to overwrite an existing experiment artifact. "
            f"Choose a new --experiment-name: {model_name}"
        )
    os.makedirs(learning_curve_dir, exist_ok=False)
    os.makedirs(evaluation_curve_dir, exist_ok=False)

    print("=" * 80)
    print("SAC closing training")
    print("=" * 80)
    print(f"Repository root: {REPO_ROOT}")
    print(f"Model path:      {model_path}")
    print(f"Best model:      {best_model_path}")
    print(f"VecNorm path:    {vecnorm_path}")
    print(f"Best VecNorm:    {best_vecnorm_path}")
    print(f"Learning curve:  {learning_curve_plot}")
    print(f"Eval curve:      {evaluation_curve_plot}")
    print(f"Evaluations NPZ: {evaluations_npz}")
    print(f"Total steps:     {total_timesteps}")
    print(f"Episode horizon: {args.max_time}")
    print(f"Initialization:  {'checkpoint' if initialized else 'from scratch'}")
    print(f"Configuration:   {args.configuration}")
    print(f"SAC parameters:  {params}")
    print_device_info(args.device)
    print("=" * 80)

    # -----------------------------------------------------------------
    # Create training environment
    # -----------------------------------------------------------------

    train_env = DummyVecEnv([
        lambda: make_training_env(seed=seed, max_time=args.max_time)
    ])

    if initialized:
        train_env = VecNormalize.load(args.initialize_vecnormalize, train_env)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=params["gamma"],
        )

    # -----------------------------------------------------------------
    # SAC model
    # -----------------------------------------------------------------

    if initialized:
        model = SAC.load(args.initialize_model, env=train_env, device=args.device)
        # SAC.load restores weights and optimizer state but not replay data.
        # With reset_num_timesteps=True below, refill the new replay buffer
        # for the original trial-16 warm-up before allowing gradient updates.
        model.learning_starts = params["learning_starts"]
        model.tensorboard_log = LOG_DIR
    else:
        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=params["learning_rate"],
            buffer_size=300_000,
            learning_starts=params["learning_starts"],
            batch_size=params["batch_size"],
            tau=params["tau"],
            gamma=params["gamma"],
            train_freq=params["train_freq"],
            gradient_steps=params["gradient_steps"],
            ent_coef=params["ent_coef"],
            target_update_interval=1,
            verbose=1,
            seed=seed,
            tensorboard_log=LOG_DIR,
            device=args.device,
        )

    config_path = os.path.join(learning_curve_dir, "experiment_config.json")
    with open(config_path, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "experiment_name": model_name,
                "training_seed": seed,
                "new_environment_steps": total_timesteps,
                "episode_horizon_seconds": args.max_time,
                "configuration": args.configuration,
                "sac_parameters": params,
                "buffer_size": 300_000,
                "initialization": "checkpoint" if initialized else "from_scratch",
                "source_model": os.path.abspath(args.initialize_model) if initialized else None,
                "source_vecnormalize": os.path.abspath(args.initialize_vecnormalize) if initialized else None,
                "existing_artifacts_overwritten": False,
            },
            stream,
            indent=2,
        )

    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------

    learning_curve_callback = LearningCurveCallback(
        csv_path=learning_curve_csv,
        plot_path=learning_curve_plot,
        model_name=model_name,
        moving_average_window=10,
    )
    best_model_callback = BestModelCallback(
        best_model_path=best_model_path,
        best_vecnorm_path=best_vecnorm_path,
        eval_seeds=range(10),
        eval_freq=50_000,
        max_steps=int(args.max_time or 12000),
        eval_env_factory=lambda seed: make_eval_env(seed, max_time=args.max_time),
    )
    periodic_eval_callback = PeriodicEvalCallback(
        csv_path=evaluation_curve_csv,
        npz_path=evaluations_npz,
        plot_path=evaluation_curve_plot,
        model_name=model_name,
        eval_seeds=[0, 1, 2, 3, 4],
        eval_every_episodes=10,
        max_steps=int(args.max_time or 12000),
        eval_env_factory=lambda seed: make_eval_env(seed, max_time=args.max_time),
    )

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=model_name,
        progress_bar=True,
        callback=[learning_curve_callback, best_model_callback, periodic_eval_callback],
        reset_num_timesteps=True,
    )

    best_checkpoint = best_model_callback.get_best_checkpoint()
    if best_checkpoint is not None:
        learning_curve_callback.set_best_checkpoint(**best_checkpoint)
        learning_curve_callback.save_outputs()
        print(
            "Marked best deterministic checkpoint on training learning curve "
            f"at episode {best_checkpoint['episode']} "
            f"(timestep {best_checkpoint['timestep']}, "
            f"mean_eval_return={best_checkpoint['score']:.6f})"
        )

    # -----------------------------------------------------------------
    # Save closing model and normalization statistics
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
    print(f"Saved eval CSV:   {evaluation_curve_csv}")
    print(f"Saved eval NPZ:   {evaluations_npz}")
    print(f"Saved eval plot:  {evaluation_curve_plot}")
    print(f"Saved best to:    {best_model_path}.zip")
    print("=" * 80)


if __name__ == "__main__":
    main()
