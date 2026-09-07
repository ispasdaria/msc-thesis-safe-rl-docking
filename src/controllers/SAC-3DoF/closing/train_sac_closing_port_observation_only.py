"""Train the vanilla Trial-16 closing SAC with six port-relative observations.

Only the observation construction differs from the original Trial-16 run.
Reward v15, dynamics, reset distribution, termination, SAC hyperparameters,
training budget, replay buffer, and normalization settings are retained.
"""

import argparse
import os
import sys

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import train_sac_closing as original
from closing_port_observation_env import ClosingDockingPortObservationEnv


MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "reward_shaping_trials")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "reward_shaping_trials")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "reward_shaping_trials")

SEED = 0
TOTAL_TIMESTEPS = 400_000
BUFFER_SIZE = 300_000
MODEL_NAME = "closing_sac_trial16_docking_port_obs_only"


def make_training_env(seed=SEED):
    return ClosingDockingPortObservationEnv(
        mode="training",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
        closing_reward_version=original.CLOSING_REWARD_VERSION,
    )


def make_eval_env(seed=0):
    return ClosingDockingPortObservationEnv(
        mode="testing",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
        closing_reward_version=original.CLOSING_REWARD_VERSION,
    )


def evaluate_policy(model, eval_seeds, max_steps=12000):
    return original.evaluate_deterministic_policy(
        model,
        eval_seeds=eval_seeds,
        max_steps=max_steps,
        eval_env_factory=make_eval_env,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def validate_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")


def main():
    args = parse_args()
    validate_device(args.device)

    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    best_model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_best")
    vecnorm_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_vecnormalize.pkl")
    best_vecnorm_path = os.path.join(
        MODEL_DIR, f"{MODEL_NAME}_best_vecnormalize.pkl"
    )
    learning_curve_dir = os.path.join(
        RESULTS_DIR, "episode_return_learning_curves", MODEL_NAME
    )
    evaluation_curve_dir = os.path.join(
        RESULTS_DIR, "evaluation_learning_curves", MODEL_NAME
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(learning_curve_dir, exist_ok=True)
    os.makedirs(evaluation_curve_dir, exist_ok=True)

    train_env = VecNormalize(
        DummyVecEnv([lambda: original.Monitor(make_training_env(SEED))]),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=original.TRIAL_16_PARAMS["gamma"],
    )

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=original.TRIAL_16_PARAMS["learning_rate"],
        buffer_size=BUFFER_SIZE,
        learning_starts=original.TRIAL_16_PARAMS["learning_starts"],
        batch_size=original.TRIAL_16_PARAMS["batch_size"],
        tau=original.TRIAL_16_PARAMS["tau"],
        gamma=original.TRIAL_16_PARAMS["gamma"],
        train_freq=original.TRIAL_16_PARAMS["train_freq"],
        gradient_steps=original.TRIAL_16_PARAMS["gradient_steps"],
        ent_coef=original.TRIAL_16_PARAMS["ent_coef"],
        target_update_interval=1,
        verbose=1,
        seed=SEED,
        tensorboard_log=LOG_DIR,
        device=args.device,
    )

    learning_curve_callback = original.LearningCurveCallback(
        csv_path=os.path.join(learning_curve_dir, "learning_curve.csv"),
        plot_path=os.path.join(learning_curve_dir, "learning_curve.png"),
        model_name=MODEL_NAME,
        moving_average_window=10,
    )
    best_model_callback = original.BestModelCallback(
        best_model_path=best_model_path,
        best_vecnorm_path=best_vecnorm_path,
        eval_seeds=range(10),
        eval_freq=50_000,
        max_steps=12_000,
        eval_env_factory=make_eval_env,
    )
    periodic_eval_callback = original.PeriodicEvalCallback(
        csv_path=os.path.join(evaluation_curve_dir, "evaluation_learning_curve.csv"),
        npz_path=os.path.join(evaluation_curve_dir, "evaluations.npz"),
        plot_path=os.path.join(evaluation_curve_dir, "evaluation_learning_curve.png"),
        model_name=MODEL_NAME,
        eval_seeds=[0, 1, 2, 3, 4],
        eval_every_episodes=10,
        max_steps=12_000,
        eval_env_factory=make_eval_env,
    )

    print(f"Observation shape: {train_env.observation_space.shape}")
    print(f"Reward version: {original.CLOSING_REWARD_VERSION}")
    print(f"Model path: {model_path}.zip")
    print(f"VecNormalize path: {vecnorm_path}")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            tb_log_name=MODEL_NAME,
            progress_bar=True,
            callback=[
                learning_curve_callback,
                best_model_callback,
                periodic_eval_callback,
            ],
        )
        model.save(model_path)
        train_env.save(vecnorm_path)
    finally:
        train_env.close()


if __name__ == "__main__":
    main()
