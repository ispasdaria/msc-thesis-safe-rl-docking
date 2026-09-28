"""Train the controlled 0.1 s closing-phase SAC experiment.

This experiment transfers the established V15 Trial-16 setup to a 10 Hz
controller while preserving physical-time discounting, reward meaning,
warm-up duration, evaluation cadence, and optimizer-update cadence.
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..")
)
SRC_PATH = os.path.join(REPO_ROOT, "src")

CLOSING_DIR = os.path.dirname(SCRIPT_DIR)
for path in (SCRIPT_DIR, CLOSING_DIR, SRC_PATH):
    if path not in sys.path:
        sys.path.insert(0, path)

import train_sac_closing as baseline
from envs.docking3dof.gym_env import Docking3DoFGymEnv


REFERENCE_DT = 1.0
EXPERIMENT_DT = 0.1
BASE_GAMMA = baseline.TRIAL_16_PARAMS["gamma"]
EXPERIMENT_GAMMA = BASE_GAMMA ** (EXPERIMENT_DT / REFERENCE_DT)
REWARD_VERSION = "v15_dt"
SUPPORTED_REWARD_VERSIONS = ("v15_dt", "v15_dt_safe")

DEFAULT_TOTAL_TIMESTEPS = 500_000
FULL_TOTAL_TIMESTEPS = 4_000_000
DEFAULT_BUFFER_SIZE = 3_000_000
DEFAULT_LEARNING_STARTS = 100_000
DEFAULT_TRAIN_FREQ = 160
DEFAULT_GRADIENT_STEPS = baseline.TRIAL_16_PARAMS["gradient_steps"]
DEFAULT_EVAL_FREQ = 500_000
DEFAULT_MAX_TIME = 12_000.0
DEFAULT_EVAL_SEEDS = tuple(range(100, 105))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train closing-phase SAC at dt=0.1 s with timestep-normalized "
            "V15 reward shaping."
        )
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dt",
        type=float,
        choices=[EXPERIMENT_DT],
        default=EXPERIMENT_DT,
    )
    parser.add_argument(
        "--reward-version",
        choices=SUPPORTED_REWARD_VERSIONS,
        default=REWARD_VERSION,
    )
    parser.add_argument(
        "--integrator",
        choices=["rk4"],
        default="rk4",
    )
    parser.add_argument(
        "--action-hold-steps",
        type=int,
        choices=[1],
        default=1,
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help=(
            "SAC interaction steps. Use 500000 for the pilot and 4000000 "
            "for the physical-time-matched full run."
        ),
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=DEFAULT_BUFFER_SIZE,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=baseline.TRIAL_16_PARAMS["learning_rate"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=baseline.TRIAL_16_PARAMS["batch_size"],
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=baseline.TRIAL_16_PARAMS["tau"],
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=EXPERIMENT_GAMMA,
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=DEFAULT_LEARNING_STARTS,
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=DEFAULT_TRAIN_FREQ,
        help="Environment transitions between SAC training bursts.",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=DEFAULT_GRADIENT_STEPS,
    )
    parser.add_argument(
        "--ent-coef",
        default=baseline.TRIAL_16_PARAMS["ent_coef"],
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=[256, 256],
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=DEFAULT_EVAL_FREQ,
    )
    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_EVAL_SEEDS),
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=DEFAULT_MAX_TIME,
        help="Maximum physical episode duration in seconds.",
    )
    parser.add_argument("--clip-obs", type=float, default=10.0)
    parser.add_argument("--clip-reward", type=float, default=10.0)
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional artifact directory/model stem.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the configuration without creating artifacts.",
    )
    return parser.parse_args(argv)


def validate_args(args):
    baseline.validate_device(args.device)

    positive_integer_fields = (
        "total_timesteps",
        "buffer_size",
        "learning_starts",
        "train_freq",
        "gradient_steps",
        "eval_freq",
        "batch_size",
        "target_update_interval",
        "action_hold_steps",
    )
    for field in positive_integer_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive.")

    if args.learning_starts >= args.total_timesteps:
        raise ValueError(
            "--learning-starts must be smaller than --total-timesteps."
        )
    if args.buffer_size < args.batch_size:
        raise ValueError("--buffer-size must be at least --batch-size.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if not 0.0 < args.tau <= 1.0:
        raise ValueError("--tau must lie in (0, 1].")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("--gamma must lie in (0, 1].")
    if args.max_time <= 0.0:
        raise ValueError("--max-time must be positive.")
    if args.clip_obs <= 0.0 or args.clip_reward <= 0.0:
        raise ValueError("Normalization clipping bounds must be positive.")
    if not args.net_arch or any(width <= 0 for width in args.net_arch):
        raise ValueError("--net-arch must contain positive layer widths.")
    if not args.eval_seeds:
        raise ValueError("--eval-seeds must contain at least one seed.")


def default_experiment_name(args):
    step_label = f"{args.total_timesteps // 1000}ksteps"
    return (
        "sac_closing_v15dt_dt0p1_trial16_"
        f"{step_label}_seed{args.seed}_buffer{args.buffer_size // 1000}k"
    )


def build_experiment_config(args):
    max_episode_steps = int(math.ceil(args.max_time / args.dt))
    trial_16 = dict(baseline.TRIAL_16_PARAMS)
    hyperparameters = {
        **trial_16,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "tau": args.tau,
        "gamma": args.gamma,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "ent_coef": args.ent_coef,
        "target_update_interval": args.target_update_interval,
        "policy": "MlpPolicy",
        "policy_network": list(args.net_arch),
    }
    return {
        "experiment_name": args.output_name or default_experiment_name(args),
        "purpose": "controlled closing SAC timestep experiment at 10 Hz",
        "seed": args.seed,
        "device": args.device,
        "dt_seconds": args.dt,
        "control_interval_seconds": args.dt * args.action_hold_steps,
        "control_frequency_hz": 1.0 / (args.dt * args.action_hold_steps),
        "action_hold_steps": args.action_hold_steps,
        "integrator": args.integrator,
        "max_time_seconds": args.max_time,
        "max_episode_steps": max_episode_steps,
        "total_timesteps": args.total_timesteps,
        "simulated_training_time_seconds": (
            args.total_timesteps * args.dt
        ),
        "full_physical_time_matched_timesteps": FULL_TOTAL_TIMESTEPS,
        "reward": {
            "version": args.reward_version,
            "reference_dt_seconds": REFERENCE_DT,
            "persistent_cost_scale": args.dt / REFERENCE_DT,
            "progress_potential_scale_changed": False,
            "terminal_event_scale_changed": False,
            "no_progress_rate_threshold_mps": 0.01,
            "terminal_failure_reward": (
                -50_000.0
                if args.reward_version == "v15_dt_safe"
                else -1_000.0
            ),
            "terminal_success_reward": (
                50_000.0
                if args.reward_version == "v15_dt_safe"
                else 1_000.0
            ),
            "terminal_timeout_reward": (
                -10_000.0
                if args.reward_version == "v15_dt_safe"
                else -200.0
            ),
            "task_first_checkpoint_selection": True,
            "early_speed_guard_fraction_of_safe_limit": (
                0.7 if args.reward_version == "v15_dt_safe" else None
            ),
            "tolerance_normalized_alignment": (
                args.reward_version == "v15_dt_safe"
            ),
        },
        "discount": {
            "base_gamma_at_1s": BASE_GAMMA,
            "gamma_per_decision": args.gamma,
            "conversion": "base_gamma ** (dt / reference_dt)",
        },
        "hyperparameters": hyperparameters,
        "evaluation": {
            "frequency_timesteps": args.eval_freq,
            "frequency_physical_seconds": args.eval_freq * args.dt,
            "seeds": list(args.eval_seeds),
            "max_steps": max_episode_steps,
        },
        "normalization": {
            "normalize_observations": True,
            "normalize_rewards": True,
            "clip_observations": args.clip_obs,
            "clip_rewards": args.clip_reward,
            "gamma": args.gamma,
            "fresh_statistics": True,
        },
    }


def make_env(
    mode,
    seed,
    max_time,
    dt=EXPERIMENT_DT,
    reward_version=REWARD_VERSION,
    integrator="rk4",
    action_hold_steps=1,
):
    return Docking3DoFGymEnv(
        phase="closing",
        mode=mode,
        dt=dt,
        max_time=max_time,
        integrator=integrator,
        normalize_action=True,
        seed=seed,
        closing_reward_version=reward_version,
        action_hold_steps=action_hold_steps,
    )


def evaluate_task_metrics(model, eval_seeds, max_steps, eval_env_factory):
    """Evaluate deterministic task and safety metrics for checkpoint ranking."""
    vecnormalize = model.get_vec_normalize_env()
    if vecnormalize is None:
        raise RuntimeError("VecNormalize is required for checkpoint evaluation.")

    old_training = vecnormalize.training
    old_norm_reward = vecnormalize.norm_reward
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    rows = []
    try:
        for seed in eval_seeds:
            env = eval_env_factory(seed=seed)
            try:
                observation, _ = env.reset(seed=seed)
                episode_return = 0.0
                terminal_info = {}
                steps = 0
                for _ in range(max_steps):
                    normalized = vecnormalize.normalize_obs(
                        observation.reshape(1, -1)
                    )
                    action, _ = model.predict(normalized, deterministic=True)
                    observation, reward, terminated, truncated, terminal_info = (
                        env.step(action[0])
                    )
                    episode_return += float(reward)
                    steps += 1
                    if terminated or truncated:
                        break

                error = np.asarray(
                    env.core_env.state - env.core_env.p3_target,
                    dtype=np.float64,
                )
                error[4] = env.core_env.wrap_to_pi(error[4])
                tolerance = np.asarray(
                    env.core_env.p3_target_tolerance,
                    dtype=np.float64,
                )
                ratios = np.abs(error) / np.maximum(tolerance, 1e-12)
                rows.append(
                    {
                        "seed": int(seed),
                        "success": int(
                            bool(terminal_info.get("success", False))
                        ),
                        "safety_failure": int(
                            terminal_info.get("termination_reason")
                            == "safety_violation"
                        ),
                        "timeout": int(
                            bool(terminal_info.get("time_limit", False))
                        ),
                        "episode_return": float(episode_return),
                        "steps": int(steps),
                        "elapsed_time_seconds": float(env.core_env.time),
                        "final_position_error_m": float(
                            np.linalg.norm(error[:2])
                        ),
                        "final_max_target_ratio": float(np.max(ratios)),
                        "termination_reason": terminal_info.get(
                            "termination_reason"
                        ),
                    }
                )
            finally:
                env.close()
    finally:
        vecnormalize.training = old_training
        vecnormalize.norm_reward = old_norm_reward

    def mean(key):
        return float(np.mean([row[key] for row in rows]))

    summary = {
        "rollouts": len(rows),
        "success_rate": mean("success"),
        "safety_failure_rate": mean("safety_failure"),
        "timeout_rate": mean("timeout"),
        "mean_final_max_target_ratio": mean("final_max_target_ratio"),
        "mean_final_position_error_m": mean("final_position_error_m"),
        "mean_return": mean("episode_return"),
        "mean_steps": mean("steps"),
        "mean_elapsed_time_seconds": mean("elapsed_time_seconds"),
    }
    return summary, rows


def task_first_checkpoint_rank(summary):
    """Rank task completion and safety before diagnostic return."""
    return (
        summary["success_rate"],
        -summary["safety_failure_rate"],
        -summary["mean_final_max_target_ratio"],
        -summary["mean_final_position_error_m"],
        summary["mean_return"],
    )


class TaskFirstBestModelCallback(BaseCallback):
    """Save periodic checkpoints and select best by task-first metrics."""

    def __init__(
        self,
        best_model_path,
        best_vecnorm_path,
        checkpoint_dir,
        history_csv_path,
        history_json_path,
        eval_seeds,
        eval_freq,
        max_steps,
        eval_env_factory,
    ):
        super().__init__()
        self.best_model_path = best_model_path
        self.best_vecnorm_path = best_vecnorm_path
        self.checkpoint_dir = checkpoint_dir
        self.history_csv_path = history_csv_path
        self.history_json_path = history_json_path
        self.eval_seeds = list(eval_seeds)
        self.eval_freq = int(eval_freq)
        self.max_steps = int(max_steps)
        self.eval_env_factory = eval_env_factory
        self.last_eval_timestep = 0
        self.completed_episodes = 0
        self.best_rank = None
        self.best_record = None
        self.history = []

    def _write_history(self):
        if not self.history:
            return
        with open(self.history_csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.history[0]))
            writer.writeheader()
            writer.writerows(self.history)
        with open(self.history_json_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "selection_order": [
                        "highest success rate",
                        "lowest safety-failure rate",
                        "lowest final maximum target ratio",
                        "lowest final position error",
                        "highest return",
                    ],
                    "best": self.best_record,
                    "evaluations": self.history,
                },
                file,
                indent=2,
            )

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if info.get("episode") is not None:
                self.completed_episodes += 1

        if self.num_timesteps - self.last_eval_timestep < self.eval_freq:
            return True
        self.last_eval_timestep = self.num_timesteps

        checkpoint_stem = f"checkpoint_{self.num_timesteps:07d}"
        checkpoint_model = os.path.join(self.checkpoint_dir, checkpoint_stem)
        checkpoint_vecnorm = os.path.join(
            self.checkpoint_dir,
            f"{checkpoint_stem}_vecnormalize.pkl",
        )
        self.model.save(checkpoint_model)
        self.training_env.save(checkpoint_vecnorm)

        summary, _ = evaluate_task_metrics(
            self.model,
            eval_seeds=self.eval_seeds,
            max_steps=self.max_steps,
            eval_env_factory=self.eval_env_factory,
        )
        record = {
            "timestep": int(self.num_timesteps),
            "completed_episodes": int(self.completed_episodes),
            **summary,
        }
        self.history.append(record)
        rank = task_first_checkpoint_rank(summary)
        if self.best_rank is None or rank > self.best_rank:
            self.best_rank = rank
            self.best_record = dict(record)
            self.model.save(self.best_model_path)
            self.training_env.save(self.best_vecnorm_path)

        self._write_history()
        print(
            "Task-first checkpoint evaluation "
            f"timestep={self.num_timesteps}, "
            f"success={summary['success_rate']:.2%}, "
            f"safety_failure={summary['safety_failure_rate']:.2%}, "
            f"final_ratio={summary['mean_final_max_target_ratio']:.3f}, "
            f"return={summary['mean_return']:.3f}",
            flush=True,
        )
        return True

    def _on_training_end(self):
        self._write_history()

    def get_best_checkpoint(self):
        return self.best_record


def print_configuration(config):
    hyperparameters = config["hyperparameters"]
    print("=" * 80)
    print("SAC closing dt=0.1 s experiment")
    print("=" * 80)
    print(f"Experiment:       {config['experiment_name']}")
    print(f"Device:           {config['device']}")
    print(f"Training seed:    {config['seed']}")
    print(f"Reward:           {config['reward']['version']}")
    print(f"Control interval: {config['control_interval_seconds']:.1f} s")
    print(f"Total steps:      {config['total_timesteps']}")
    print(
        "Physical time:     "
        f"{config['simulated_training_time_seconds']:.1f} s"
    )
    print(f"Gamma:            {hyperparameters['gamma']:.12f}")
    print(f"Learning rate:    {hyperparameters['learning_rate']:.12g}")
    print(f"Batch size:       {hyperparameters['batch_size']}")
    print(f"Tau:              {hyperparameters['tau']}")
    print(f"Learning starts:  {hyperparameters['learning_starts']}")
    print(f"Train frequency:  {hyperparameters['train_freq']}")
    print(f"Gradient steps:   {hyperparameters['gradient_steps']}")
    print(f"Entropy:          {hyperparameters['ent_coef']}")
    print(f"Network:          {hyperparameters['policy_network']}")
    print(f"Replay buffer:    {hyperparameters['buffer_size']}")
    print(f"Evaluation seeds: {config['evaluation']['seeds']}")
    print(f"Evaluation steps: {config['evaluation']['max_steps']}")
    print("=" * 80)


def run_training(args):
    validate_args(args)
    config = build_experiment_config(args)
    print_configuration(config)

    if args.dry_run:
        return config

    experiment_name = config["experiment_name"]
    model_dir = os.path.join(
        SCRIPT_DIR,
        "models",
        "reward_shaping_trials",
        "dt_0p1",
        experiment_name,
    )
    results_dir = os.path.join(
        SCRIPT_DIR,
        "results",
        "reward_shaping_trials",
        "dt_0p1",
        experiment_name,
    )
    log_dir = os.path.join(
        SCRIPT_DIR,
        "logs",
        "reward_shaping_trials",
        "dt_0p1",
    )
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    config_path = os.path.join(results_dir, "experiment_config.json")
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    final_model_path = os.path.join(model_dir, f"{experiment_name}_final")
    final_vecnorm_path = os.path.join(
        model_dir,
        f"{experiment_name}_final_vecnormalize.pkl",
    )
    best_model_path = os.path.join(model_dir, f"{experiment_name}_best")
    best_vecnorm_path = os.path.join(
        model_dir,
        f"{experiment_name}_best_vecnormalize.pkl",
    )

    raw_training_env = DummyVecEnv(
        [
            lambda: Monitor(
                make_env(
                    "training",
                    args.seed,
                    args.max_time,
                    dt=args.dt,
                    reward_version=args.reward_version,
                    integrator=args.integrator,
                    action_hold_steps=args.action_hold_steps,
                )
            )
        ]
    )
    train_env = VecNormalize(
        raw_training_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=args.clip_obs,
        clip_reward=args.clip_reward,
        gamma=args.gamma,
    )

    hyperparameters = config["hyperparameters"]
    model = SAC(
        policy="MlpPolicy",
        policy_kwargs={"net_arch": list(args.net_arch)},
        env=train_env,
        learning_rate=hyperparameters["learning_rate"],
        buffer_size=hyperparameters["buffer_size"],
        learning_starts=hyperparameters["learning_starts"],
        batch_size=hyperparameters["batch_size"],
        tau=hyperparameters["tau"],
        gamma=hyperparameters["gamma"],
        train_freq=hyperparameters["train_freq"],
        gradient_steps=hyperparameters["gradient_steps"],
        ent_coef=hyperparameters["ent_coef"],
        target_update_interval=hyperparameters["target_update_interval"],
        verbose=1,
        seed=args.seed,
        tensorboard_log=log_dir,
        device=args.device,
    )

    learning_curve_callback = baseline.LearningCurveCallback(
        csv_path=os.path.join(results_dir, "learning_curve.csv"),
        plot_path=os.path.join(results_dir, "learning_curve.png"),
        model_name=experiment_name,
        moving_average_window=10,
    )
    eval_env_factory = lambda seed: make_env(
        "testing",
        seed,
        args.max_time,
        dt=args.dt,
        reward_version=args.reward_version,
        integrator=args.integrator,
        action_hold_steps=args.action_hold_steps,
    )
    best_model_callback = TaskFirstBestModelCallback(
        best_model_path=best_model_path,
        best_vecnorm_path=best_vecnorm_path,
        checkpoint_dir=model_dir,
        history_csv_path=os.path.join(
            results_dir,
            "checkpoint_evaluations.csv",
        ),
        history_json_path=os.path.join(
            results_dir,
            "checkpoint_evaluations.json",
        ),
        eval_seeds=args.eval_seeds,
        eval_freq=args.eval_freq,
        max_steps=config["evaluation"]["max_steps"],
        eval_env_factory=eval_env_factory,
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            tb_log_name=experiment_name,
            progress_bar=True,
            callback=[learning_curve_callback, best_model_callback],
        )
        model.save(final_model_path)
        train_env.save(final_vecnorm_path)

        best_checkpoint = best_model_callback.get_best_checkpoint()
        summary = {
            "experiment_name": experiment_name,
            "completed": True,
            "final_model": f"{final_model_path}.zip",
            "final_vecnormalize": final_vecnorm_path,
            "best_checkpoint": best_checkpoint,
        }
        with open(
            os.path.join(results_dir, "training_summary.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(summary, file, indent=2)
    finally:
        train_env.close()

    print(f"Saved final model: {final_model_path}.zip")
    print(f"Saved VecNormalize: {final_vecnorm_path}")
    print(f"Saved configuration: {config_path}")
    return config


def main(argv=None):
    args = parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
