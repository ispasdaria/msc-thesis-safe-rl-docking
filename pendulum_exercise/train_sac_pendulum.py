import gymnasium as gym
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

BASE_DIR = Path(__file__).resolve().parent

env = gym.make("Pendulum-v1")
env = Monitor(env)

model = SAC(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=20_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",
    tensorboard_log=str(BASE_DIR / "sac_pendulum_logs"),
)

model.learn(
    total_timesteps=20_000,
    tb_log_name="SAC_Pendulum"
)

model.save(BASE_DIR / "models" / "sac_pendulum_20k_1e-3_model")
env.close()
