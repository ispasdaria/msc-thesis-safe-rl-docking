import csv
from pathlib import Path

import numpy as np
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise


ENV_ID = "MountainCarContinuous-v0"
BASE_DIR = Path(__file__).resolve().parent

SEEDS = [0, 1, 2, 3, 4]
N_TIMESTEPS = 60_000
N_EVAL_EPISODES = 20

LOG_BASE_DIR = BASE_DIR / "sac_mountain_car_logs"
MODEL_BASE_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RUN_GROUP_NAME = "sac_mountain_car_zoo_params_60k_5seeds"

GOAL_POSITION = 0.45


def make_env(seed=None):
    env = gym.make(ENV_ID)
    env = Monitor(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    return env


def train_one_seed(seed):
    run_name = f"{RUN_GROUP_NAME}_seed{seed}"

    log_dir = LOG_BASE_DIR / run_name
    model_dir = MODEL_BASE_DIR / run_name
    best_model_dir = model_dir / "best_model"

    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_env(seed=seed)
    eval_env = make_env(seed=seed + 1000)
    n_actions = train_env.action_space.shape[-1]

    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.5 * np.ones(n_actions),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(log_dir),
        eval_freq=5_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    model = SAC(
        policy="MlpPolicy",
        env=train_env,

        # RL-Baselines3-Zoo parameters for MountainCarContinuous-v0
        learning_rate=3e-4,
        buffer_size=50_000,
        batch_size=64,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=1,
        learning_starts=0,

        verbose=1,
        tensorboard_log=str(log_dir),
        seed=seed,
        device="auto",
        action_noise=action_noise,
    )

    print("\n" + "=" * 70)
    print(f"Training seed {seed}")
    print("=" * 70)

    model.learn(
        total_timesteps=N_TIMESTEPS,
        callback=eval_callback,
        tb_log_name=run_name,
    )

    final_model_path = model_dir / "final_model"
    model.save(final_model_path)

    train_env.close()
    eval_env.close()

    return final_model_path, best_model_dir / "best_model"


def evaluate_model(model_path, seed, n_eval_episodes=N_EVAL_EPISODES):
    model = SAC.load(model_path)

    episode_results = []

    for ep in range(n_eval_episodes):
        eval_seed = seed * 1000 + ep

        env = gym.make(ENV_ID)
        obs, _ = env.reset(seed=eval_seed)
        max_position = float(obs[0])

        position_history = []
        velocity_history = []
        action_history = []
        reward_history = []

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)
            position_history.append(obs[0])
            velocity_history.append(obs[1])
            action_history.append(action[0])
            reward_history.append(reward)
            max_position = max(max_position, float(obs[0]))

        env.close()

        position_history = np.array(position_history)
        velocity_history = np.array(velocity_history)
        action_history = np.array(action_history)
        reward_history = np.array(reward_history)

        final_position = float(position_history[-1])
        final_velocity = float(velocity_history[-1])
        total_reward = float(np.sum(reward_history))
        episode_length = int(len(reward_history))
        mean_abs_action = float(np.mean(np.abs(action_history)))
        success = bool(max_position >= GOAL_POSITION)

        episode_results.append(
            {
                "training_seed": seed,
                "eval_episode": ep,
                "eval_seed": eval_seed,
                "success": success,
                "total_reward": total_reward,
                "episode_length": episode_length,
                "max_position": max_position,
                "final_position": final_position,
                "final_velocity": final_velocity,
                "mean_abs_action": mean_abs_action,
            }
        )

    return episode_results


def summarize_seed_results(seed, episode_results):
    success_rate = np.mean([r["success"] for r in episode_results])
    mean_reward = np.mean([r["total_reward"] for r in episode_results])
    std_reward = np.std([r["total_reward"] for r in episode_results])
    mean_episode_length = np.mean([r["episode_length"] for r in episode_results])
    mean_max_position = np.mean([r["max_position"] for r in episode_results])
    mean_abs_action = np.mean([r["mean_abs_action"] for r in episode_results])

    return {
        "training_seed": seed,
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_episode_length": mean_episode_length,
        "mean_max_position": mean_max_position,
        "mean_abs_action": mean_abs_action,
    }


def save_csv(path, rows):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with path.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_episode_results = []
    summary_results = []

    for seed in SEEDS:
        final_model_path, best_model_path = train_one_seed(seed)

        # Evaluate the best checkpoint when available; otherwise evaluate the final checkpoint.
        if best_model_path.with_suffix(".zip").exists():
            model_to_evaluate = best_model_path
            model_type = "best"
        else:
            model_to_evaluate = final_model_path
            model_type = "final"

        print("\n" + "-" * 70)
        print(f"Evaluating seed {seed} using {model_type} model:")
        print(model_to_evaluate)
        print("-" * 70)

        episode_results = evaluate_model(
            model_path=model_to_evaluate,
            seed=seed,
            n_eval_episodes=N_EVAL_EPISODES,
        )

        for result in episode_results:
            result["model_type"] = model_type

        seed_summary = summarize_seed_results(seed, episode_results)
        seed_summary["model_type"] = model_type

        all_episode_results.extend(episode_results)
        summary_results.append(seed_summary)

    episode_csv_path = RESULTS_DIR / f"{RUN_GROUP_NAME}_episode_results.csv"

    summary_csv_path = RESULTS_DIR / f"{RUN_GROUP_NAME}_summary.csv"

    save_csv(episode_csv_path, all_episode_results)
    save_csv(summary_csv_path, summary_results)

    print("\n" + "=" * 90)
    print("SUMMARY OVER TRAINING SEEDS")
    print("=" * 90)
    print(
        f"{'Seed':>6} | {'Model':>6} | {'Success':>8} | {'Mean reward':>12} | "
        f"{'Std reward':>10} | {'Mean length':>12} | {'Mean max pos':>12} | {'Mean |a|':>9}"
    )
    print("-" * 90)

    for row in summary_results:
        print(
            f"{row['training_seed']:>6} | "
            f"{row['model_type']:>6} | "
            f"{row['success_rate']:>8.2f} | "
            f"{row['mean_reward']:>12.2f} | "
            f"{row['std_reward']:>10.2f} | "
            f"{row['mean_episode_length']:>12.1f} | "
            f"{row['mean_max_position']:>12.3f} | "
            f"{row['mean_abs_action']:>9.3f}"
        )

    print("\nSaved episode results to:")
    print(episode_csv_path)

    print("\nSaved summary results to:")
    print(summary_csv_path)

if __name__ == "__main__":
    main()
