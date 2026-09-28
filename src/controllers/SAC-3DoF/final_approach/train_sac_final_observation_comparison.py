"""Train final-approach SAC with one of two additional observation spaces.

The existing docking-port controller is not loaded, modified, or retrained.
This script keeps its reward, dynamics, reset distribution, SAC settings, and
VecNormalize procedure, while giving each new observation contract its own
model and normalization-statistics filenames.
"""

import argparse
import os
import sys

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from observation_space_env import FinalApproachObservationEnv, OBSERVATION_MODES
from train_sac_final_single_run import (
    LearningCurveCallback,
    SaveBestModelAndVecNormalizeCallback,
)


MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "development_experiments", "observation_comparison")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "development_experiments", "observation_comparison")
RESULTS_DIR = os.path.join(
    SCRIPT_DIR, "results", "development_experiments", "observation_comparison", "learning_curves"
)

SEED = 12
TOTAL_TIMESTEPS = 100_000
BUFFER_SIZE = 100_000
REWARD_VERSION = "v9"


def make_env(observation_mode, seed, mode):
    env = FinalApproachObservationEnv(
        observation_mode=observation_mode,
        mode=mode,
        dt=1.0,
        max_time=5_000.0,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
    )
    return Monitor(env)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one isolated final-approach observation-space variant."
    )
    parser.add_argument(
        "--observation-mode",
        choices=("com", "combined"),
        required=True,
        help="Additional observation space to train.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")


def train(observation_mode, device, progress_bar, overwrite):
    if observation_mode not in OBSERVATION_MODES:
        raise ValueError(f"Unsupported observation mode: {observation_mode}")

    model_name = {
        "com": "final_sac_com_observation",
        "combined": "final_sac_combined_observation",
    }[observation_mode]
    model_path = os.path.join(MODEL_DIR, model_name)
    vecnorm_path = os.path.join(MODEL_DIR, f"{model_name}_vecnormalize.pkl")
    best_model_path = os.path.join(MODEL_DIR, f"{model_name}_best")
    best_vecnorm_path = os.path.join(
        MODEL_DIR, f"{model_name}_best_vecnormalize.pkl"
    )
    curve_dir = os.path.join(RESULTS_DIR, model_name)
    curve_csv = os.path.join(curve_dir, "learning_curve.csv")
    curve_plot = os.path.join(curve_dir, "learning_curve.png")

    if not overwrite and os.path.exists(model_path + ".zip"):
        raise FileExistsError(
            f"{model_path}.zip already exists; use --overwrite to retrain."
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(curve_dir, exist_ok=True)

    train_env = VecNormalize(
        DummyVecEnv([lambda: make_env(observation_mode, SEED, "training")]),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.98,
    )
    eval_env = VecNormalize(
        DummyVecEnv([
            lambda: make_env(observation_mode, SEED + 10_000, "testing")
        ]),
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.98,
    )
    eval_env.training = False
    eval_env.norm_reward = False

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=0.0003395686575052769,
        buffer_size=BUFFER_SIZE,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        train_freq=8,
        gradient_steps=8,
        ent_coef="auto_0.1",
        target_update_interval=1,
        verbose=1,
        seed=SEED,
        tensorboard_log=LOG_DIR,
        device=device,
    )

    learning_curve_callback = LearningCurveCallback(
        csv_path=curve_csv,
        plot_path=curve_plot,
        rolling_window=20,
    )
    save_best_callback = SaveBestModelAndVecNormalizeCallback(
        best_model_path=best_model_path,
        best_vecnorm_path=best_vecnorm_path,
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=save_best_callback,
        best_model_save_path=None,
        log_path=curve_dir,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )

    print(f"Observation mode: {observation_mode}")
    print(f"Reward definition: {REWARD_VERSION} final-approach path")
    print(f"Model: {model_path}.zip")
    print(f"VecNormalize: {vecnorm_path}")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            tb_log_name=model_name,
            progress_bar=progress_bar,
            callback=[learning_curve_callback, eval_callback],
        )
        model.save(model_path)
        train_env.save(vecnorm_path)
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    args = parse_args()
    validate_device(args.device)
    train(args.observation_mode, args.device, args.progress_bar, args.overwrite)
