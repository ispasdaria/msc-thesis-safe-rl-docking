import os
import numpy as np 
import gymnasium as gym
import winsound 

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise 


ENV_ID = "MountainCarContinuous-v0"

RUN_NAME = "sac_mountain_car_zoo_params"
LOG_DIR = f"mountain_car_exercise/sac_mountain_car_logs/{RUN_NAME}"
MODEL_DIR = "mountain_car_exercise/models"
MODEL_NAME = RUN_NAME


def make_env():
    env = gym.make(ENV_ID)
    env = Monitor(env)
    return env


def notify_done():
    try:
        import win32com.client
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.Speak("Training completed!")
    except Exception:
        print("Training completed!")


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

    n_actions = train_env.action_space.shape[-1]

    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.5 *np.ones(n_actions)
    )

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=0,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        tensorboard_log=LOG_DIR,
        seed=0,
        device="auto",
        action_noise=action_noise, 
    )

    model.learn(
        total_timesteps=60_000,
        callback=eval_callback,
        tb_log_name=RUN_NAME,
    )

    model.save(os.path.join(MODEL_DIR, MODEL_NAME))

    train_env.close()
    eval_env.close()

    #winsound.Beep(1000, 700)
    notify_done()

if __name__ == "__main__":
    main()