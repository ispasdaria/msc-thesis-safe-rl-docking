"""
Tune SAC hyperparameters for the 3-DoF docking final-approach phase.

The objective is safety-first: unsafe trials are heavily penalized, and
safe trials are ranked by final port distance and lateral offset.
"""

import argparse
import csv
import os
import sys

import numpy as np
import optuna
from stable_baselines3 import SAC
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
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "optuna")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_training_env(seed=0):
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


def make_eval_env(seed=0):
    return Docking3DoFGymEnv(
        phase="final",
        mode="testing",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
    )


def normalize_observation(vecnorm, obs):
    return vecnorm.normalize_obs(obs.reshape(1, -1))


def evaluate_model(model, vecnorm, seeds, max_steps):
    results = []

    vecnorm.training = False
    vecnorm.norm_reward = False

    for seed in seeds:
        env = make_eval_env(seed=seed)
        obs, info = env.reset(seed=seed)
        final_info = info

        port_distances = []
        s_values = []
        rho_values = []

        for step in range(max_steps):
            norm_obs = normalize_observation(vecnorm, obs)
            action, _ = model.predict(norm_obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action[0])

            final_info = info
            port_distances.append(info.get("port_distance", np.nan))
            s_values.append(info.get("s", np.nan))
            rho_values.append(info.get("rho", np.nan))

            if terminated or truncated:
                break

        env.close()

        port_distances = np.array(port_distances, dtype=np.float64)
        s_values = np.array(s_values, dtype=np.float64)
        rho_values = np.array(rho_values, dtype=np.float64)

        results.append(
            {
                "seed": seed,
                "steps": step + 1,
                "success": bool(final_info.get("success", False)),
                "failure": bool(final_info.get("failure", False)),
                "time_limit": bool(final_info.get("time_limit", False)),
                "termination_reason": final_info.get("termination_reason"),
                "violated_constraints": final_info.get("violated_constraints", []),
                "final_port_distance": float(port_distances[-1]),
                "min_port_distance": float(np.nanmin(port_distances)),
                "final_s": float(s_values[-1]),
                "min_s": float(np.nanmin(s_values)),
                "final_rho": float(rho_values[-1]),
                "max_rho": float(np.nanmax(rho_values)),
            }
        )

    return results


def summarize_results(results):
    failures = np.array([r["failure"] for r in results], dtype=np.float64)
    successes = np.array([r["success"] for r in results], dtype=np.float64)

    return {
        "success_rate": float(np.mean(successes)),
        "failure_rate": float(np.mean(failures)),
        "time_limit_rate": float(np.mean([r["time_limit"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "mean_final_port_distance": float(
            np.nanmean([r["final_port_distance"] for r in results])
        ),
        "mean_min_port_distance": float(
            np.nanmean([r["min_port_distance"] for r in results])
        ),
        "mean_final_s": float(np.nanmean([r["final_s"] for r in results])),
        "mean_min_s": float(np.nanmean([r["min_s"] for r in results])),
        "mean_final_rho": float(np.nanmean([r["final_rho"] for r in results])),
        "mean_max_rho": float(np.nanmean([r["max_rho"] for r in results])),
    }


def score_summary(summary):
    failure_rate = summary["failure_rate"]

    if failure_rate > 0.0:
        return (
            -1000.0
            - 200.0 * failure_rate
            - summary["mean_final_port_distance"]
            - 10.0 * summary["mean_final_rho"]
        )

    return (
        100.0 * summary["success_rate"]
        - summary["mean_final_port_distance"]
        - 5.0 * summary["mean_final_rho"]
    )


def suggest_sac_params(trial):
    train_update = trial.suggest_categorical(
        "train_update",
        ["1_1", "4_4", "8_8"],
    )
    train_freq, gradient_steps = [int(value) for value in train_update.split("_")]

    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-4,
            7e-4,
            log=True,
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size",
            [128, 256, 512],
        ),
        "tau": trial.suggest_categorical(
            "tau",
            [0.002, 0.005, 0.01],
        ),
        "gamma": trial.suggest_categorical(
            "gamma",
            [0.98, 0.99, 0.995],
        ),
        "learning_starts": trial.suggest_categorical(
            "learning_starts",
            [1000, 5000, 10000],
        ),
        "ent_coef": trial.suggest_categorical(
            "ent_coef",
            ["auto", "auto_0.1", "auto_0.5"],
        ),
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
    }


def append_trial_csv(csv_path, trial, score, summary):
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "trial",
                    "score",
                    "success_rate",
                    "failure_rate",
                    "time_limit_rate",
                    "mean_steps",
                    "mean_final_port_distance",
                    "mean_min_port_distance",
                    "mean_final_s",
                    "mean_min_s",
                    "mean_final_rho",
                    "mean_max_rho",
                    "params",
                ]
            )

        writer.writerow(
            [
                trial.number,
                f"{score:.10f}",
                f"{summary['success_rate']:.10f}",
                f"{summary['failure_rate']:.10f}",
                f"{summary['time_limit_rate']:.10f}",
                f"{summary['mean_steps']:.10f}",
                f"{summary['mean_final_port_distance']:.10f}",
                f"{summary['mean_min_port_distance']:.10f}",
                f"{summary['mean_final_s']:.10f}",
                f"{summary['mean_min_s']:.10f}",
                f"{summary['mean_final_rho']:.10f}",
                f"{summary['mean_max_rho']:.10f}",
                repr(trial.params),
            ]
        )


def build_objective(args, metrics_csv):
    eval_seeds = list(range(args.eval_seeds))

    def objective(trial):
        seed = args.seed + trial.number
        params = suggest_sac_params(trial)

        train_env = DummyVecEnv([lambda: make_training_env(seed=seed)])
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=params["gamma"],
        )

        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=params["learning_rate"],
            buffer_size=args.buffer_size,
            learning_starts=params["learning_starts"],
            batch_size=params["batch_size"],
            tau=params["tau"],
            gamma=params["gamma"],
            train_freq=params["train_freq"],
            gradient_steps=params["gradient_steps"],
            ent_coef=params["ent_coef"],
            target_update_interval=1,
            verbose=0,
            seed=seed,
            tensorboard_log=LOG_DIR,
            device=args.device,
        )

        model.learn(
            total_timesteps=args.train_steps,
            tb_log_name=f"{args.study_name}_trial_{trial.number:03d}",
            progress_bar=args.progress_bar,
        )

        results = evaluate_model(
            model=model,
            vecnorm=train_env,
            seeds=eval_seeds,
            max_steps=args.max_eval_steps,
        )
        summary = summarize_results(results)
        score = score_summary(summary)

        for key, value in summary.items():
            trial.set_user_attr(key, value)

        append_trial_csv(metrics_csv, trial, score, summary)

        if args.save_models:
            trial_dir = os.path.join(RESULTS_DIR, args.study_name, f"trial_{trial.number:03d}")
            os.makedirs(trial_dir, exist_ok=True)
            model.save(os.path.join(trial_dir, "model"))
            train_env.save(os.path.join(trial_dir, "vecnormalize.pkl"))

        train_env.close()

        print(
            f"Trial {trial.number:03d} | "
            f"score={score:.3f} | "
            f"failure_rate={summary['failure_rate']:.2f} | "
            f"final_port={summary['mean_final_port_distance']:.2f} m | "
            f"final_rho={summary['mean_final_rho']:.2f} m | "
            f"params={trial.params}"
        )

        return score

    return objective


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for SAC final approach."
    )
    parser.add_argument("--study-name", default="sac_final_rewardv9_optuna")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=100_000)
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--max-eval-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL. Defaults to a SQLite DB in results/optuna.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete an existing study with the same name before starting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    study_dir = os.path.join(RESULTS_DIR, args.study_name)
    os.makedirs(study_dir, exist_ok=True)

    storage = args.storage
    if storage is None:
        storage = "sqlite:///" + os.path.join(study_dir, "study.db")

    metrics_csv = os.path.join(study_dir, "trial_metrics.csv")

    if args.fresh:
        try:
            optuna.delete_study(study_name=args.study_name, storage=storage)
        except KeyError:
            pass
        if os.path.exists(metrics_csv):
            os.remove(metrics_csv)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=0,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=not args.fresh,
    )

    print("=" * 80)
    print("Optuna SAC final-approach tuning")
    print("=" * 80)
    print(f"Study name:   {args.study_name}")
    print(f"Storage:      {storage}")
    print(f"Metrics CSV:  {metrics_csv}")
    print(f"Trials:       {args.n_trials}")
    print(f"Train steps:  {args.train_steps}")
    print(f"Eval seeds:   {list(range(args.eval_seeds))}")
    print("=" * 80)

    study.optimize(
        build_objective(args, metrics_csv),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    best = study.best_trial

    print("=" * 80)
    print("Best trial")
    print("=" * 80)
    print(f"Trial:        {best.number}")
    print(f"Score:        {best.value:.6f}")
    print(f"Params:       {best.params}")
    for key, value in best.user_attrs.items():
        print(f"{key}: {value}")
    print("=" * 80)


if __name__ == "__main__":
    main()
