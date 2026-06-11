import os
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


ENV_ID = "MountainCarContinuous-v0"

RUN_NAME = "sac_mountain_car_100k_lr3e4_bs256"

LOG_DIR = f"mountain_car_exercise/sac_mountain_car_logs/{RUN_NAME}"
MODEL_DIR = "mountain_car_exercise/models"
MODEL_NAME = RUN_NAME


def make_env():
    env = gym.make(ENV_ID)
    env = Monitor(env)
    return env


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_env = make_env()
    eval_env = make_env()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, MODEL_NAME + "_best"),
        log_path=LOG_DIR,
        eval_freq=5_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        tensorboard_log=LOG_DIR,
        seed=0,
        device="auto",
    )

    model.learn(
        total_timesteps=100_000,
        callback=eval_callback,
        tb_log_name=RUN_NAME,
    )

    model.save(os.path.join(MODEL_DIR, MODEL_NAME))

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()