"""Run the expanded-v2 SAC search for the 3-DoF closing phase.

The objective is safety-first and is based on the sampled six-state P3 target.
Port distance is recorded only as a diagnostic because the closing phase should
end near P3 (roughly 300 m from the target), not at the docking port.

Example:
    python tune_sac_closing_optuna.py --n-trials 40 --train-steps 300000 --fresh
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

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
OPTUNA_VERSION = "v2"
LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "optuna", OPTUNA_VERSION)
RESULTS_DIR = os.path.join(
    SCRIPT_DIR,
    "results",
    "optuna",
    OPTUNA_VERSION,
)

SEARCH_SPACE = {
    "learning_rate": {
        "low": 1.8e-4,
        "high": 3.5e-4,
        "log": True,
    },
    "batch_size": [256, 512],
    "tau": [0.005, 0.01, 0.015, 0.02],
    "gamma": [0.99, 0.9925, 0.995, 0.997],
    "learning_starts": [5_000, 10_000, 20_000, 30_000],
    "ent_coef": [
        "auto_0.01",
        "auto_0.02",
        "auto_0.05",
        "auto_0.1",
    ],
    "train_update": ["8_8", "16_8", "16_4", "32_8", "32_16"],
}

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_training_env(seed=0, closing_reward_version="v15"):
    env = Docking3DoFGymEnv(
        phase="closing",
        mode="training",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
        closing_reward_version=closing_reward_version,
    )
    return Monitor(env)


def make_eval_env(seed=0, closing_reward_version="v15"):
    return Docking3DoFGymEnv(
        phase="closing",
        mode="testing",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
        closing_reward_version=closing_reward_version,
    )


def normalize_observation(vecnorm, obs):
    return vecnorm.normalize_obs(obs.reshape(1, -1))


def target_error(env):
    """Return the current P3 error with a wrapped attitude component."""
    if env.core_env.p3_target is None:
        raise RuntimeError("Closing evaluation environment has no P3 target.")

    error = env.core_env.state - env.core_env.p3_target
    error = np.asarray(error, dtype=np.float64)
    error[4] = env.core_env.wrap_to_pi(error[4])
    return error


def target_error_metrics(error, tolerance):
    ratios = np.abs(error) / np.maximum(tolerance, 1e-12)
    return {
        "position_error": float(np.linalg.norm(error[0:2])),
        "lateral_error": float(abs(error[1])),
        "velocity_error": float(np.linalg.norm(error[2:4])),
        "attitude_error_deg": float(np.rad2deg(abs(error[4]))),
        "rate_error_degs": float(np.rad2deg(abs(error[5]))),
        "position_ratio": float(np.linalg.norm(ratios[0:2])),
        "velocity_ratio": float(np.max(ratios[2:4])),
        "alignment_ratio": float(np.max(ratios[4:6])),
        "max_target_ratio": float(np.max(ratios)),
    }


def evaluate_model(
    model,
    vecnorm,
    seeds,
    max_steps,
    closing_reward_version="v15",
):
    results = []

    old_training = vecnorm.training
    old_norm_reward = vecnorm.norm_reward
    vecnorm.training = False
    vecnorm.norm_reward = False

    try:
        for seed in seeds:
            env = make_eval_env(
                seed=seed,
                closing_reward_version=closing_reward_version,
            )

            try:
                obs, info = env.reset(seed=seed)
                final_info = info
                tolerance = np.asarray(
                    env.core_env.p3_target_tolerance,
                    dtype=np.float64,
                )
                initial_error = target_error(env)
                initial_metrics = target_error_metrics(initial_error, tolerance)

                min_position_error = initial_metrics["position_error"]
                port_distances = []
                rho_values = []
                port_speeds = []
                abs_s_dot_values = []
                abs_psi_values = []
                abs_omega_values = []
                warning_steps = 0
                episode_return = 0.0

                for step in range(max_steps):
                    norm_obs = normalize_observation(vecnorm, obs)
                    action, _ = model.predict(norm_obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(
                        action[0]
                    )
                    final_info = info
                    episode_return += float(reward)

                    current_error = target_error(env)
                    min_position_error = min(
                        min_position_error,
                        float(np.linalg.norm(current_error[0:2])),
                    )
                    port_distances.append(info.get("port_distance", np.nan))
                    rho_values.append(info.get("rho", np.nan))
                    port_speeds.append(info.get("port_speed", np.nan))
                    abs_s_dot_values.append(abs(info.get("s_dot", np.nan)))
                    abs_psi_values.append(abs(info.get("psi_e", np.nan)))
                    abs_omega_values.append(abs(info.get("omega", np.nan)))
                    warning_steps += int(bool(info.get("warning", False)))

                    if terminated or truncated:
                        break

                final_error = target_error(env)
                final_metrics = target_error_metrics(final_error, tolerance)
                initial_position_error = initial_metrics["position_error"]
                progress_fraction = (
                    initial_position_error - final_metrics["position_error"]
                ) / max(initial_position_error, 1e-12)

                results.append(
                    {
                        "seed": seed,
                        "steps": step + 1,
                        "episode_return": episode_return,
                        "success": bool(final_info.get("success", False)),
                        "failure": bool(final_info.get("failure", False)),
                        "time_limit": bool(final_info.get("time_limit", False)),
                        "termination_reason": final_info.get(
                            "termination_reason"
                        ),
                        "violated_constraints": final_info.get(
                            "violated_constraints",
                            [],
                        ),
                        "initial_position_error": initial_position_error,
                        "final_position_error": final_metrics["position_error"],
                        "min_position_error": min_position_error,
                        "progress_fraction": float(
                            np.clip(progress_fraction, -1.0, 1.0)
                        ),
                        "final_lateral_error": final_metrics["lateral_error"],
                        "final_velocity_error": final_metrics["velocity_error"],
                        "final_attitude_error_deg": final_metrics[
                            "attitude_error_deg"
                        ],
                        "final_rate_error_degs": final_metrics[
                            "rate_error_degs"
                        ],
                        "final_position_ratio": final_metrics[
                            "position_ratio"
                        ],
                        "final_velocity_ratio": final_metrics[
                            "velocity_ratio"
                        ],
                        "final_alignment_ratio": final_metrics[
                            "alignment_ratio"
                        ],
                        "final_max_target_ratio": final_metrics[
                            "max_target_ratio"
                        ],
                        "final_port_distance": float(port_distances[-1]),
                        "min_port_distance": float(np.nanmin(port_distances)),
                        "final_rho": float(rho_values[-1]),
                        "max_rho": float(np.nanmax(rho_values)),
                        "max_port_speed": float(np.nanmax(port_speeds)),
                        "max_abs_s_dot": float(np.nanmax(abs_s_dot_values)),
                        "max_abs_psi_deg": float(
                            np.rad2deg(np.nanmax(abs_psi_values))
                        ),
                        "max_abs_omega_degs": float(
                            np.rad2deg(np.nanmax(abs_omega_values))
                        ),
                        "warning_fraction": warning_steps / (step + 1),
                    }
                )
            finally:
                env.close()
    finally:
        vecnorm.training = old_training
        vecnorm.norm_reward = old_norm_reward

    return results


def summarize_results(results):
    def mean(key):
        return float(np.nanmean([result[key] for result in results]))

    return {
        "success_rate": mean("success"),
        "failure_rate": mean("failure"),
        "time_limit_rate": mean("time_limit"),
        "mean_steps": mean("steps"),
        "mean_initial_position_error": mean("initial_position_error"),
        "mean_final_position_error": mean("final_position_error"),
        "mean_min_position_error": mean("min_position_error"),
        "mean_progress_fraction": mean("progress_fraction"),
        "mean_final_lateral_error": mean("final_lateral_error"),
        "mean_final_velocity_error": mean("final_velocity_error"),
        "mean_final_attitude_error_deg": mean("final_attitude_error_deg"),
        "mean_final_rate_error_degs": mean("final_rate_error_degs"),
        "mean_final_position_ratio": mean("final_position_ratio"),
        "mean_final_velocity_ratio": mean("final_velocity_ratio"),
        "mean_final_alignment_ratio": mean("final_alignment_ratio"),
        "mean_final_max_target_ratio": mean("final_max_target_ratio"),
        "mean_final_port_distance": mean("final_port_distance"),
        "mean_min_port_distance": mean("min_port_distance"),
        "mean_final_rho": mean("final_rho"),
        "mean_max_rho": mean("max_rho"),
        "mean_max_port_speed": mean("max_port_speed"),
        "mean_max_abs_s_dot": mean("max_abs_s_dot"),
        "mean_max_abs_psi_deg": mean("max_abs_psi_deg"),
        "mean_max_abs_omega_degs": mean("max_abs_omega_degs"),
        "mean_warning_fraction": mean("warning_fraction"),
    }


def score_summary(summary):
    """Return a scalar score with a strict safe-over-unsafe separation."""
    target_quality = (
        1000.0 * summary["mean_progress_fraction"]
        - 20.0 * summary["mean_final_position_ratio"]
        - 10.0 * summary["mean_final_velocity_ratio"]
        - 10.0 * summary["mean_final_alignment_ratio"]
        - 25.0 * summary["mean_warning_fraction"]
    )

    if summary["failure_rate"] > 0.0:
        return (
            -1_000_000.0
            - 100_000.0 * summary["failure_rate"]
            + target_quality
        )

    return 100_000.0 * summary["success_rate"] + target_quality


def suggest_sac_params(trial):
    train_update = trial.suggest_categorical(
        "train_update",
        SEARCH_SPACE["train_update"],
    )
    train_freq, gradient_steps = [
        int(value) for value in train_update.split("_")
    ]

    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            SEARCH_SPACE["learning_rate"]["low"],
            SEARCH_SPACE["learning_rate"]["high"],
            log=SEARCH_SPACE["learning_rate"]["log"],
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size",
            SEARCH_SPACE["batch_size"],
        ),
        "tau": trial.suggest_categorical(
            "tau",
            SEARCH_SPACE["tau"],
        ),
        "gamma": trial.suggest_categorical(
            "gamma",
            SEARCH_SPACE["gamma"],
        ),
        "learning_starts": trial.suggest_categorical(
            "learning_starts",
            SEARCH_SPACE["learning_starts"],
        ),
        "ent_coef": trial.suggest_categorical(
            "ent_coef",
            SEARCH_SPACE["ent_coef"],
        ),
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
    }


def append_trial_csv(csv_path, trial, score, summary):
    file_exists = os.path.exists(csv_path)
    metric_names = list(summary)

    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["trial", "score", *metric_names, "params"])

        writer.writerow(
            [
                trial.number,
                f"{score:.10f}",
                *(f"{summary[name]:.10f}" for name in metric_names),
                repr(trial.params),
            ]
        )


def build_objective(args, metrics_csv):
    eval_seeds = list(range(args.eval_seeds))

    def objective(trial):
        # Keep the training seed fixed so trial-to-trial differences reflect
        # hyperparameters rather than a different stochastic training run.
        seed = args.seed
        params = suggest_sac_params(trial)
        train_env = None

        try:
            train_env = DummyVecEnv(
                [lambda: make_training_env(seed=seed)]
            )
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
                tb_log_name=(
                    f"{args.study_name}_trial_{trial.number:03d}"
                ),
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
                trial_dir = os.path.join(
                    RESULTS_DIR,
                    args.study_name,
                    f"trial_{trial.number:03d}",
                )
                os.makedirs(trial_dir, exist_ok=True)
                model.save(os.path.join(trial_dir, "model"))
                train_env.save(os.path.join(trial_dir, "vecnormalize.pkl"))

            print(
                f"Trial {trial.number:03d} | "
                f"score={score:.3f} | "
                f"success={summary['success_rate']:.2f} | "
                f"failure={summary['failure_rate']:.2f} | "
                f"P3 error={summary['mean_final_position_error']:.2f} m | "
                f"progress={summary['mean_progress_fraction']:.1%} | "
                f"target ratio={summary['mean_final_max_target_ratio']:.2f} | "
                f"params={trial.params}"
            )

            return score
        finally:
            if train_env is not None:
                train_env.close()

    return objective


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for SAC closing phase."
    )
    parser.add_argument(
        "--study-name",
        default="sac_closing_p3augobs_rewardv15_optuna_expanded_v2",
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=300_000)
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--max-eval-steps", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument(
        "--storage",
        default=None,
        help=(
            "Optuna storage URL. Defaults to a SQLite DB in "
            "results/optuna/v2."
        ),
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
        database_path = Path(study_dir, "study.db").resolve().as_posix()
        storage = f"sqlite:///{database_path}"

    metrics_csv = os.path.join(study_dir, "trial_metrics.csv")
    config_path = os.path.join(study_dir, "study_config.json")

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "optuna_version": OPTUNA_VERSION,
                "study_name": args.study_name,
                "storage": storage,
                "metrics_csv": metrics_csv,
                "log_dir": LOG_DIR,
                "results_dir": study_dir,
                "arguments": vars(args),
                "search_space": SEARCH_SPACE,
            },
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")

    if args.fresh:
        try:
            optuna.delete_study(
                study_name=args.study_name,
                storage=storage,
            )
        except KeyError:
            pass
        if os.path.exists(metrics_csv):
            os.remove(metrics_csv)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=not args.fresh,
    )

    print("=" * 80)
    print("Optuna SAC closing-phase tuning")
    print("=" * 80)
    print(f"Study name:     {args.study_name}")
    print(f"Storage:        {storage}")
    print(f"Metrics CSV:    {metrics_csv}")
    print(f"Study config:   {config_path}")
    print(f"TensorBoard:    {LOG_DIR}")
    print(f"Trials:         {args.n_trials}")
    print(f"Train steps:    {args.train_steps}")
    print(f"Buffer size:    {args.buffer_size}")
    print(f"Training seed:  {args.seed} (fixed across trials)")
    print(f"Eval seeds:     {list(range(args.eval_seeds))}")
    print(f"Max eval steps: {args.max_eval_steps}")
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
    print(f"Trial:  {best.number}")
    print(f"Score:  {best.value:.6f}")
    print(f"Params: {best.params}")
    for key, value in best.user_attrs.items():
        print(f"{key}: {value}")
    print("=" * 80)


if __name__ == "__main__":
    main()
