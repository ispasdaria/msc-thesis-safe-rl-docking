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

    # Main SAC defaults from Stable-Baselines3
    learning_rate=3e-4,
    buffer_size=1_000_000,
    learning_starts=100,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    action_noise=None,
    replay_buffer_class=None,
    replay_buffer_kwargs=None,
    optimize_memory_usage=False,
    ent_coef="auto",
    target_update_interval=1,
    target_entropy="auto",
    use_sde=False,
    sde_sample_freq=-1,
    use_sde_at_warmup=False,
    stats_window_size=100,
    policy_kwargs=None,
    seed=None,
    device="auto",

    # Your logging / display choices
    verbose=1,
    tensorboard_log=str(BASE_DIR / "sac_pendulum_logs"),
)

model.learn(
    total_timesteps=20_000,
    tb_log_name="SAC_Pendulum_Default_3e-4"
)

model.save(BASE_DIR / "models" / "sac_pendulum_default_3e-4")
env.close()
