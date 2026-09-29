"""Train goal-conditioned residual SAC+HER around the closing LQR."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[5]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from her_closing_utils import (
    checkpoint_rank,
    evaluate_model,
    make_her_env,
    summarize_rollouts,
    write_csv,
    write_json,
)
from her_source_warmstart import prefill_her_replay_buffer
from lqr_residual_env import ResidualLQRGoalConditionedClosingEnv
from lqr_residual_lateral_env import (
    LateralProgressResidualLQRGoalConditionedClosingEnv,
)
from lqr_residual_terminal_fade_env import (
    TerminalFadeLateralResidualLQRGoalConditionedClosingEnv,
)
from residual_lqr_config import (
    CLOSING_LQR_CONTROL_HZ,
    CLOSING_LQR_DT_SECONDS,
    discount_for_dt,
    steps_for_duration,
)
from train_sac_closing_v15_baseline_residual_her import (
    ResidualSafetyGuardCallback,
    initialize_zero_residual_actor,
)
from zero_residual_constrained_sac import ZeroResidualConstrainedSAC
from validate_residual_lqr_dt01_baseline import validate_saved_baseline_manifest


CLOSING_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE_MANIFEST = (
    CLOSING_DIR
    / "results"
    / "residual_lqr_baseline"
    / "dt0p1_validation_seeds100_109"
    / "baseline_manifest.json"
)
MODEL_ROOT = CLOSING_DIR / "models" / "her_lqr_residual"
RESULT_ROOT = CLOSING_DIR / "results" / "her_lqr_residual"
LOG_ROOT = CLOSING_DIR / "logs" / "her_lqr_residual"

REWARD_ENV_CLASSES = {
    "lqr_residual": ResidualLQRGoalConditionedClosingEnv,
    "lqr_residual_lateral": LateralProgressResidualLQRGoalConditionedClosingEnv,
    "lqr_residual_terminal_fade": (
        TerminalFadeLateralResidualLQRGoalConditionedClosingEnv
    ),
}


def parse_entropy_coefficient(value):
    """Accept SB3's ``auto[_initial]`` form or a fixed positive float."""
    text = str(value).strip()
    if text == "auto" or text.startswith("auto_"):
        if text.startswith("auto_"):
            try:
                initial_value = float(text.removeprefix("auto_"))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    "automatic entropy must be 'auto' or 'auto_<positive>'."
                ) from exc
            if not np.isfinite(initial_value) or initial_value <= 0.0:
                raise argparse.ArgumentTypeError(
                    "automatic entropy initialization must be positive."
                )
        return text
    try:
        coefficient = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "entropy coefficient must be a positive float or auto[_initial]."
        ) from exc
    if not np.isfinite(coefficient) or coefficient <= 0.0:
        raise argparse.ArgumentTypeError(
            "fixed entropy coefficient must be positive."
        )
    return coefficient


class PeriodicTrainingLoggerCallback(BaseCallback):
    """Flush train metrics even when 120k-step episodes rarely terminate."""

    def __init__(self, interval_steps):
        super().__init__(verbose=0)
        self.interval_steps = int(interval_steps)
        if self.interval_steps <= 0:
            raise ValueError("interval_steps must be positive.")
        self.next_dump_step = self.interval_steps

    def _on_step(self):
        if self.num_timesteps >= self.next_dump_step:
            self.logger.dump(self.num_timesteps)
            while self.next_dump_step <= self.num_timesteps:
                self.next_dump_step += self.interval_steps
        return True


def _copy_observation(observation):
    return {
        key: np.asarray(value, dtype=np.float32).copy()
        for key, value in observation.items()
    }


def _compact_info(info, truncated):
    keys = (
        "failure",
        "time_limit",
        "success",
        "port_speed",
        "s_dot",
        "psi_e",
        "omega",
        "state",
        "violated_constraints",
        "termination_reason",
        "previous_achieved_goal",
        "effective_residual_normalized_action",
        "previous_effective_residual_normalized_action",
    )
    compact = {key: copy.deepcopy(info[key]) for key in keys if key in info}
    compact["TimeLimit.truncated"] = bool(truncated)
    return compact


def collect_lqr_residual_trajectories(
    *,
    minimum_transitions,
    seed_start,
    max_episodes,
    max_steps,
    residual_scale,
    residual_zero_distance=None,
    residual_full_distance=None,
    environment_residual_hold_steps=1,
    residual_std=0.25,
    residual_clip=0.75,
    residual_hold_steps=20,
    zero_episode_fraction=0.5,
    minimum_zero_episodes=0,
    minimum_exploratory_episodes=0,
    random_seed=6_000,
    retain_failure_episodes=False,
    terminate_on_success=False,
    reward_version="lqr_residual",
    reward_kwargs=None,
):
    """Collect complete LQR episodes with deterministic role coverage."""
    if minimum_transitions <= 0:
        raise ValueError("minimum_transitions must be positive.")
    if residual_std < 0.0:
        raise ValueError("residual_std must be non-negative.")
    if not 0.0 < residual_clip <= 1.0:
        raise ValueError("residual_clip must lie in (0, 1].")
    if residual_hold_steps <= 0:
        raise ValueError("residual_hold_steps must be positive.")
    if not 0.0 <= zero_episode_fraction <= 1.0:
        raise ValueError("zero_episode_fraction must lie in [0, 1].")
    if minimum_zero_episodes < 0 or minimum_exploratory_episodes < 0:
        raise ValueError("minimum episode-role counts must be non-negative.")
    if residual_std == 0.0 and minimum_exploratory_episodes > 0:
        raise ValueError(
            "An exploratory prefill episode requires residual_std > 0."
        )
    if reward_version not in REWARD_ENV_CLASSES:
        raise ValueError(f"Unsupported residual reward version: {reward_version}")

    rng = np.random.default_rng(random_seed)
    reward_kwargs = {} if reward_kwargs is None else dict(reward_kwargs)
    env = make_her_env(
        seed=seed_start,
        mode="training",
        max_time=float(max_steps) * CLOSING_LQR_DT_SECONDS,
        reward_version=reward_version,
        dt=CLOSING_LQR_DT_SECONDS,
        residual_scale=residual_scale,
        residual_zero_distance=residual_zero_distance,
        residual_full_distance=residual_full_distance,
        residual_hold_steps=environment_residual_hold_steps,
        terminate_on_success=terminate_on_success,
        **reward_kwargs,
    )
    transitions = []
    summaries = []
    accepted_episode_id = 0
    accepted_zero_episodes = 0
    accepted_exploratory_episodes = 0
    try:
        for attempt in range(max_episodes):
            role_requirements_met = bool(
                accepted_zero_episodes >= minimum_zero_episodes
                and accepted_exploratory_episodes
                >= minimum_exploratory_episodes
            )
            if len(transitions) >= minimum_transitions and role_requirements_met:
                break
            seed = int(seed_start + attempt)
            observation, _ = env.reset(seed=seed)
            episode = []
            final_info = {}
            if accepted_zero_episodes < minimum_zero_episodes:
                zero_episode = True
                prefill_role = "required_zero"
            elif accepted_exploratory_episodes < minimum_exploratory_episodes:
                zero_episode = False
                prefill_role = "required_exploratory"
            else:
                zero_episode = bool(
                    residual_std == 0.0
                    or rng.random() < zero_episode_fraction
                )
                prefill_role = "sampled_zero" if zero_episode else "sampled_exploratory"
            residual_action = np.zeros(3, dtype=np.float32)
            action_square_sum = np.zeros(3, dtype=np.float64)
            action_abs_max = np.zeros(3, dtype=np.float64)
            lqr_saturated_steps = 0

            for step in range(int(max_steps)):
                if not zero_episode and step % residual_hold_steps == 0:
                    residual_action = np.clip(
                        rng.normal(0.0, residual_std, size=3),
                        -residual_clip,
                        residual_clip,
                    ).astype(np.float32)
                next_observation, reward, terminated, truncated, final_info = (
                    env.step(residual_action)
                )
                done = bool(terminated or truncated)
                episode.append(
                    {
                        "observation": _copy_observation(observation),
                        "next_observation": _copy_observation(next_observation),
                        "action": residual_action.copy(),
                        "reward": float(reward),
                        "done": done,
                        "info": _compact_info(final_info, truncated),
                        "episode_id": accepted_episode_id,
                    }
                )
                action_square_sum += np.square(residual_action)
                action_abs_max = np.maximum(action_abs_max, np.abs(residual_action))
                lqr_saturated_steps += int(final_info.get("lqr_saturated", False))
                observation = next_observation
                if done:
                    break

            failure = bool(final_info.get("failure", False))
            retain_episode = bool(not failure or retain_failure_episodes)
            action_rms = np.sqrt(action_square_sum / max(len(episode), 1))
            summaries.append(
                {
                    "attempt": attempt + 1,
                    "seed": seed,
                    "steps": len(episode),
                    "accepted": int(retain_episode),
                    "zero_residual_episode": int(zero_episode),
                    "prefill_role": prefill_role,
                    "success": int(bool(final_info.get("success", False))),
                    "failure": int(failure),
                    "time_limit": int(bool(final_info.get("time_limit", False))),
                    "lqr_saturation_fraction": float(
                        lqr_saturated_steps / max(len(episode), 1)
                    ),
                    "residual_rms_ax": float(action_rms[0]),
                    "residual_rms_ay": float(action_rms[1]),
                    "residual_rms_alpha": float(action_rms[2]),
                    "residual_max_ax": float(action_abs_max[0]),
                    "residual_max_ay": float(action_abs_max[1]),
                    "residual_max_alpha": float(action_abs_max[2]),
                    "termination_reason": final_info.get("termination_reason", ""),
                    "violated_constraints": str(
                        final_info.get("violated_constraints", [])
                    ),
                }
            )
            if retain_episode:
                transitions.extend(episode)
                accepted_episode_id += 1
                if zero_episode:
                    accepted_zero_episodes += 1
                else:
                    accepted_exploratory_episodes += 1
    finally:
        env.close()

    role_requirements_met = bool(
        accepted_zero_episodes >= minimum_zero_episodes
        and accepted_exploratory_episodes >= minimum_exploratory_episodes
    )
    if len(transitions) < minimum_transitions or not role_requirements_met:
        raise RuntimeError(
            f"Collected only {len(transitions):,} retained LQR-residual "
            f"transitions; requested {minimum_transitions:,}. Accepted roles: "
            f"zero={accepted_zero_episodes}/{minimum_zero_episodes}, "
            "exploratory="
            f"{accepted_exploratory_episodes}/{minimum_exploratory_episodes}."
        )
    return transitions, summaries


def build_parser():
    parser = argparse.ArgumentParser(
        description="Residual SAC+HER around the closing LQR controller."
    )
    parser.add_argument(
        "--experiment-name",
        default="sac_closing_goal_her_residual_lqr_dt0p1_seed0",
    )
    parser.add_argument(
        "--reward-version",
        choices=tuple(REWARD_ENV_CLASSES),
        default="lqr_residual",
    )
    parser.add_argument(
        "--replay-mode",
        choices=("her", "standard"),
        default="her",
        help=(
            "Use HER future-goal relabeling or ordinary fixed-P3 dictionary "
            "replay. Standard replay terminates real successes."
        ),
    )
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=[5_000, 10_000, 20_000],
    )
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=list(range(100, 110)),
    )
    parser.add_argument("--source-transition-count", type=int, default=60_000)
    parser.add_argument("--source-seed-start", type=int, default=0)
    parser.add_argument("--source-max-episodes", type=int, default=12)
    parser.add_argument("--max-time", type=float, default=12_000.0)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--residual-zero-distance", type=float, default=100.0)
    parser.add_argument("--residual-full-distance", type=float, default=200.0)
    parser.add_argument("--residual-hold-steps", type=int, default=1)
    parser.add_argument("--initial-log-std", type=float, default=-4.0)
    parser.add_argument("--prefill-residual-std", type=float, default=0.25)
    parser.add_argument("--prefill-residual-clip", type=float, default=0.75)
    parser.add_argument("--prefill-residual-hold-steps", type=int, default=1)
    parser.add_argument("--prefill-zero-episode-fraction", type=float, default=0.5)
    parser.add_argument("--prefill-min-zero-episodes", type=int, default=0)
    parser.add_argument("--prefill-min-exploratory-episodes", type=int, default=0)
    parser.add_argument("--prefill-random-seed", type=int, default=6_000)
    parser.add_argument("--retain-failure-prefill", action="store_true")
    parser.add_argument("--maximum-gate-failure-rate", type=float, default=0.0)
    parser.add_argument(
        "--maximum-online-validation-failure-rate",
        type=float,
        default=0.1,
    )
    parser.add_argument("--maximum-gate-min-position-error", type=float, default=100.0)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--gamma",
        type=float,
        default=discount_for_dt(0.999),
        help="Per-0.1-second discount (default preserves gamma=0.999 per second).",
    )
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument(
        "--ent-coef",
        type=parse_entropy_coefficient,
        default="auto_0.001",
    )
    parser.add_argument("--critic-warmup-steps", type=int, default=10_000)
    parser.add_argument(
        "--critic-loss-type",
        choices=("mse", "huber"),
        default="mse",
    )
    parser.add_argument("--max-critic-grad-norm", type=float, default=None)
    parser.add_argument("--zero-residual-actor-weight", type=float, default=10.0)
    parser.add_argument("--zero-residual-actor-final-weight", type=float, default=None)
    parser.add_argument("--zero-residual-actor-anneal-end-step", type=int, default=None)
    parser.add_argument("--actor-learning-rate-scale", type=float, default=0.1)
    parser.add_argument("--max-actor-grad-norm", type=float, default=None)
    parser.add_argument("--target-entropy", type=float, default=-6.0)
    parser.add_argument("--n-sampled-goal", type=int, default=4)
    parser.add_argument("--progress-weight", type=float, default=2.0)
    parser.add_argument(
        "--goal-component-weights",
        type=float,
        nargs=6,
        default=[1.0, 4.0, 1.0, 2.0, 1.0, 1.0],
        metavar=("AXIAL", "LATERAL", "VX", "VY", "ATTITUDE", "RATE"),
    )
    parser.add_argument("--worst-component-weight", type=float, default=0.75)
    parser.add_argument(
        "--lateral-constraint-progress-weight", type=float, default=4.0
    )
    parser.add_argument("--success-bonus", type=float, default=5.0)
    parser.add_argument("--failure-penalty", type=float, default=25.0)
    parser.add_argument("--timeout-penalty", type=float, default=2.0)
    parser.add_argument("--timeout-violation-weight", type=float, default=2.0)
    parser.add_argument("--terminal-fade-full-ratio", type=float, default=1.2)
    parser.add_argument("--terminal-fade-zero-ratio", type=float, default=1.0)
    parser.add_argument("--residual-magnitude-weight", type=float, default=0.002)
    parser.add_argument("--residual-smoothness-weight", type=float, default=0.001)
    parser.add_argument("--log-interval-steps", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=DEFAULT_BASELINE_MANIFEST,
        help="Passed dt=0.1 zero-residual baseline required before training.",
    )
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(parser, args):
    if args.max_eval_steps is None:
        args.max_eval_steps = steps_for_duration(args.max_time)
    if args.total_timesteps <= 0:
        parser.error("--total-timesteps must be positive.")
    if not 0 <= args.critic_warmup_steps < args.total_timesteps:
        parser.error(
            "--critic-warmup-steps must lie in [0, total-timesteps)."
        )
    if args.zero_residual_actor_weight < 0.0:
        parser.error("--zero-residual-actor-weight must be non-negative.")
    if (
        args.zero_residual_actor_final_weight is not None
        and args.zero_residual_actor_final_weight < 0.0
    ):
        parser.error("--zero-residual-actor-final-weight must be non-negative.")
    if args.zero_residual_actor_anneal_end_step is not None:
        if args.zero_residual_actor_final_weight is None:
            parser.error(
                "--zero-residual-actor-anneal-end-step requires "
                "--zero-residual-actor-final-weight."
            )
        if args.zero_residual_actor_anneal_end_step <= args.critic_warmup_steps:
            parser.error(
                "--zero-residual-actor-anneal-end-step must exceed the "
                "critic warm-up."
            )
    if args.actor_learning_rate_scale < 0.0:
        parser.error("--actor-learning-rate-scale must be non-negative.")
    if (
        args.max_critic_grad_norm is not None
        and args.max_critic_grad_norm <= 0.0
    ):
        parser.error("--max-critic-grad-norm must be positive when set.")
    if args.max_actor_grad_norm is not None and args.max_actor_grad_norm <= 0.0:
        parser.error("--max-actor-grad-norm must be positive when set.")
    if not 0.0 < args.residual_scale <= 1.0:
        parser.error("--residual-scale must lie in (0, 1].")
    if args.residual_zero_distance < 0.0:
        parser.error("--residual-zero-distance must be non-negative.")
    if args.residual_full_distance <= args.residual_zero_distance:
        parser.error(
            "--residual-full-distance must exceed --residual-zero-distance."
        )
    if args.residual_hold_steps <= 0:
        parser.error("--residual-hold-steps must be positive.")
    if args.prefill_residual_std < 0.0:
        parser.error("--prefill-residual-std must be non-negative.")
    if not 0.0 < args.prefill_residual_clip <= 1.0:
        parser.error("--prefill-residual-clip must lie in (0, 1].")
    if args.prefill_residual_hold_steps <= 0:
        parser.error("--prefill-residual-hold-steps must be positive.")
    if not 0.0 <= args.prefill_zero_episode_fraction <= 1.0:
        parser.error("--prefill-zero-episode-fraction must lie in [0, 1].")
    if args.prefill_min_zero_episodes < 0:
        parser.error("--prefill-min-zero-episodes must be non-negative.")
    if args.prefill_min_exploratory_episodes < 0:
        parser.error("--prefill-min-exploratory-episodes must be non-negative.")
    if (
        args.prefill_min_zero_episodes + args.prefill_min_exploratory_episodes
        > args.source_max_episodes
    ):
        parser.error("Required prefill roles exceed --source-max-episodes.")
    if args.prefill_residual_std == 0.0 and args.prefill_min_exploratory_episodes:
        parser.error("Exploratory prefill requires --prefill-residual-std > 0.")
    if len(args.goal_component_weights) != 6 or any(
        not np.isfinite(value) or value <= 0.0
        for value in args.goal_component_weights
    ):
        parser.error("--goal-component-weights must contain six positives.")
    if not 0.0 <= args.worst_component_weight <= 1.0:
        parser.error("--worst-component-weight must lie in [0, 1].")
    for name in (
        "progress_weight",
        "lateral_constraint_progress_weight",
        "success_bonus",
        "failure_penalty",
        "timeout_penalty",
        "timeout_violation_weight",
        "residual_magnitude_weight",
        "residual_smoothness_weight",
    ):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative.")
    if args.terminal_fade_zero_ratio < 0.0:
        parser.error("--terminal-fade-zero-ratio must be non-negative.")
    if args.terminal_fade_full_ratio <= args.terminal_fade_zero_ratio:
        parser.error(
            "--terminal-fade-full-ratio must exceed the zero ratio."
        )
    if args.log_interval_steps <= 0:
        parser.error("--log-interval-steps must be positive.")


def reward_environment_kwargs(args):
    if args.reward_version not in {
        "lqr_residual_lateral",
        "lqr_residual_terminal_fade",
    }:
        return {}
    kwargs = {
        "progress_gamma": args.gamma,
        "progress_weight": args.progress_weight,
        "goal_component_weights": tuple(args.goal_component_weights),
        "worst_component_weight": args.worst_component_weight,
        "lateral_constraint_progress_weight": (
            args.lateral_constraint_progress_weight
        ),
        "success_bonus": args.success_bonus,
        "safety_failure_penalty": args.failure_penalty,
        "timeout_penalty": args.timeout_penalty,
        "timeout_violation_weight": args.timeout_violation_weight,
    }
    if args.reward_version == "lqr_residual_terminal_fade":
        kwargs.update(
            {
                "terminal_fade_full_ratio": args.terminal_fade_full_ratio,
                "terminal_fade_zero_ratio": args.terminal_fade_zero_ratio,
                "residual_magnitude_weight": args.residual_magnitude_weight,
                "residual_smoothness_weight": args.residual_smoothness_weight,
                "time_consistent_smoothness_reward": True,
            }
        )
    return kwargs


def write_pilot_a_model_readme(
    model_dir,
    config,
    checkpoint_summaries,
    best_checkpoint,
    run_complete,
):
    """Write a self-contained model card beside the Pilot A artifacts."""
    rows = []
    hold_steps = int(config["residual_hold_steps"])
    dt_seconds = float(config["dt_seconds"])
    residual_interval = float(config["residual_decision_interval_seconds"])
    for summary in checkpoint_summaries:
        mean_decisions = float(summary["mean_steps"])
        rows.append(
            "| {checkpoint:,} | {success:.1%} | {failure:.1%} | "
            "{timeout:.1%} | {position:.3f} | {ratio:.3f} | "
            "{decisions:.1f} | {seconds:.1f} |".format(
                checkpoint=int(summary["checkpoint"]),
                success=float(summary["success_rate"]),
                failure=float(summary["failure_rate"]),
                timeout=float(summary["time_limit_rate"]),
                position=float(summary["mean_minimum_position_error"]),
                ratio=float(summary["mean_best_max_target_ratio"]),
                decisions=mean_decisions,
                seconds=mean_decisions * residual_interval,
            )
        )
    status = "complete" if run_complete else "in progress"
    her_used = bool(config.get("her", {}).get("used", True))
    replay_row = (
        f"| HER future goals per transition | "
        f"{int(config['her']['n_sampled_goal'])} |"
        if her_used
        else "| Replay mode | Fixed original P3 goal (no HER) |"
    )
    if config["reward"]["design"] in {
        "dt01_lateral_progress_v1",
        "dt01_terminal_fade_action_cost_v2",
    }:
        reward_description = f"""The HER-compatible reward is potential-based:

```text
reward = {config['reward']['progress_weight']:.1f} * (J(previous) - gamma * J(next))
       + {config['reward']['lateral_constraint_progress_weight']:.1f}
         * (L(previous) - gamma * L(next))
       + {config['reward']['success_bonus']:.1f} * exact_P3_success
       - {config['reward']['failure_penalty']:.1f} * hard_failure
       - {config['reward']['timeout_penalty']:.1f}
         * (1 + {config['reward']['timeout_violation_weight']:.1f} * worst_violation)
         * unsuccessful_timeout
```

`J` weights lateral and lateral-velocity errors more heavily, while `L`
measures only lateral P3-set violation. Previous achieved goals are retained in
replay so the progress terms are reproducible from each stored transition."""
        if "residual_magnitude_weight" in config["reward"]:
            reward_description += f"""

The critic-visible reward also subtracts time-consistent action costs:

```text
{config['reward']['magnitude_time_scale']:.1f}
  * {config['reward']['residual_magnitude_weight']:.4f} * ||a_effective||^2
+ {config['reward']['smoothness_time_scale']:.1f}
  * {config['reward']['residual_smoothness_weight']:.4f}
  * ||a_effective - a_previous||^2
```"""
    else:
        reward_description = f"""The absolute-cost reward is:

```text
reward = {dt_seconds:g} * (-0.01 * goal_cost
                           -velocity_warning_depth
                           -geometry_warning_depth
                           -alignment_warning_depth)
         +100 * exact_P3_success
         -1000 * hard_failure
         -100 * unsuccessful_timeout
```

Persistent shaping terms are scaled by the 0.1 s / 1.0 s reference interval.
Terminal events are not time-scaled."""
    if config["residual_gate"]["design"] == "p3_ratio_terminal_fade":
        gate_description = (
            "Residual authority is full while the worst P3 ratio is at or "
            f"above {config['residual_gate']['full_authority_at_or_above_ratio']:.1f}, "
            "uses a smoothstep terminal fade, and becomes zero only at or "
            f"below ratio {config['residual_gate']['zero_authority_at_or_below_ratio']:.1f}."
        )
        gate_formula = "terminal_fade(P3_ratio)"
    else:
        gate_description = (
            "The residual gate is zero at or below "
            f"{config['residual_gate']['zero_authority_at_or_below_m']:.0f} m, "
            "uses a smoothstep ramp, and reaches full authority at or above "
            f"{config['residual_gate']['full_authority_at_or_above_m']:.0f} m."
        )
        gate_formula = "distance_gate"
    content = f"""# Direct residual SAC model card

Run: `{config['experiment_name']}`  
Status: **{status}**  
Selected best checkpoint: **{int(best_checkpoint):,} residual decisions**

## Purpose

This experiment leaves the closing LQR law unchanged. SAC selects one
normalized residual action every {residual_interval:g} s ({hold_steps}
simulator step(s)), while the LQR action is recomputed from the current physical
state every {dt_seconds:g} s. With one hold step, both feedback decisions
operate at {config['control_frequency_hz']:g} Hz.

```text
u_executed = clip(u_LQR_clipped + 0.05 * {gate_formula} * a_residual, -1, 1)
```

{gate_description}

## SAC configuration

| Parameter | Value |
|---|---:|
| Online residual decisions | {int(config['total_timesteps']):,} |
| Residual decision interval | {config['residual_decision_interval_seconds']:.1f} s |
| LQR update interval | {config['lqr_update_interval_seconds']:.1f} s |
| Replay prefill transitions | {int(config['source_transition_count_requested']):,} |
| Critic warm-up | {int(config['critic_warmup_steps']):,} residual decisions/updates |
| Critic learning rate | {config['sac']['learning_rate']:.2g} |
| Actor learning-rate scale | {config['actor_learning_rate_scale']:.3g} |
| Effective actor learning rate | {config['sac']['learning_rate'] * config['actor_learning_rate_scale']:.2g} |
| Train frequency | {int(config['sac']['train_freq'])} |
| Gradient steps | {int(config['sac']['gradient_steps'])} |
| Update-to-data ratio | {config['sac']['update_to_data_ratio']:.3g} |
| Discount factor | {config['sac']['gamma']:.6f} per residual decision |
| Target smoothing coefficient | {config['sac']['tau']:.4f} |
| Entropy coefficient | `{config['sac']['ent_coef']}` |
| Target entropy | {config['sac']['target_entropy']:.1f} |
| Zero-residual actor weight | {config['zero_residual_actor_weight']:.1f} |
| Final zero-residual actor weight | {config['zero_residual_actor_final_weight']:.1f} |
| Batch size | {int(config['sac']['batch_size'])} |
{replay_row}

The first {int(config['critic_warmup_steps']):,} online residual decisions use
the deterministic zero actor and update only the critic. With `train_freq=1`
and `gradient_steps=1`, this produces one critic update per collected residual
transition during warm-up.

## Reward

{reward_description}

The constrained actor additionally optimizes
`{config['zero_residual_actor_weight']:.1f} * ||a_residual||^2`. This is an
actor-only regularizer and is not part of the environment reward or critic
target. The reward returned for a held residual decision is the final
{dt_seconds:g}-second substep reward; the sum across substeps is retained only
as an `info` diagnostic.

## Validation results

Validation uses seeds {min(config['validation_seeds'])}--{max(config['validation_seeds'])}.
`Mean decisions` counts SAC residual decisions; `Approx. seconds` multiplies
that count by {residual_interval:g} s and may overestimate an episode whose final held
action ended early on success or termination.

| Checkpoint | Success | Hard failure | Timeout | Mean minimum position error [m] | Mean best worst ratio | Mean decisions | Approx. seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Artifacts

- `checkpoint_0_model.zip`: exact zero-residual LQR baseline;
- `checkpoint_<N>_model.zip`: evaluated model at residual decision `N`;
- `best_model.zip`: best validation-ranked checkpoint;
- `final_model.zip`: policy at the end of training.

The matching metrics and exact configuration are under
`../../../results/her_lqr_residual/{config['experiment_name']}/`.
"""
    (Path(model_dir) / "README.md").write_text(content, encoding="utf-8")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    baseline_manifest = validate_saved_baseline_manifest(
        args.baseline_manifest,
        expected_seeds=args.validation_seeds,
    )
    reward_kwargs = reward_environment_kwargs(args)
    reward_env_class = REWARD_ENV_CLASSES[args.reward_version]
    progress_reward = args.reward_version in {
        "lqr_residual_lateral",
        "lqr_residual_terminal_fade",
    }
    terminal_fade = args.reward_version == "lqr_residual_terminal_fade"
    use_her = args.replay_mode == "her"
    final_actor_weight = (
        args.zero_residual_actor_weight
        if args.zero_residual_actor_final_weight is None
        else args.zero_residual_actor_final_weight
    )
    checkpoints = sorted(
        {step for step in args.checkpoint_steps if 0 < step <= args.total_timesteps}
    )
    if args.total_timesteps not in checkpoints:
        checkpoints.append(args.total_timesteps)

    config = {
        "experiment_name": args.experiment_name,
        "method": (
            "goal-conditioned residual SAC+HER around closing LQR"
            if use_her
            else "fixed-P3 residual SAC around closing LQR without HER"
        ),
        "reward_version": args.reward_version,
        "reward_scheme": reward_env_class.reward_scheme,
        "baseline_controller": "LQRClosingController",
        "baseline_manifest": str(args.baseline_manifest.resolve()),
        "baseline_controller_fingerprint_sha256": baseline_manifest["controller"][
            "numeric_fingerprint_sha256"
        ],
        "baseline_successes": int(
            baseline_manifest["summary"]["direct_successes"]
        ),
        "baseline_action_semantics": (
            "clip physical LQR command, normalize, then add bounded residual"
        ),
        "residual_scale": args.residual_scale,
        "residual_gate": (
            {
                "design": "p3_ratio_terminal_fade",
                "zero_authority_at_or_below_ratio": (
                    args.terminal_fade_zero_ratio
                ),
                "full_authority_at_or_above_ratio": (
                    args.terminal_fade_full_ratio
                ),
                "interpolation": "smoothstep",
            }
            if terminal_fade
            else {
                "design": "axial_distance",
                "zero_authority_at_or_below_m": args.residual_zero_distance,
                "full_authority_at_or_above_m": args.residual_full_distance,
                "interpolation": "smoothstep",
            }
        ),
        "initial_residual_policy": "zero deterministic residual",
        "pilot_design": (
            "Terminal-fade dt=0.1 direct residual SAC with action costs"
            if terminal_fade
            else "Balanced-prefill dt=0.1 direct residual SAC with lateral progress"
            if args.reward_version == "lqr_residual_lateral"
            else "Canonical dt=0.1 s residual SAC around closing LQR"
        ),
        "critic_warmup_steps": args.critic_warmup_steps,
        "critic_warmup_interpretation": (
            "one critic update per residual transition during warm-up"
        ),
        "zero_residual_actor_weight": args.zero_residual_actor_weight,
        "zero_residual_actor_final_weight": final_actor_weight,
        "zero_residual_actor_anneal_end_step": (
            args.zero_residual_actor_anneal_end_step
        ),
        "actor_learning_rate_scale": args.actor_learning_rate_scale,
        "actual_p3_success_terminal": not use_her,
        "residual_hold_steps": args.residual_hold_steps,
        "dt_seconds": CLOSING_LQR_DT_SECONDS,
        "control_frequency_hz": CLOSING_LQR_CONTROL_HZ,
        "residual_decision_interval_seconds": float(
            args.residual_hold_steps * CLOSING_LQR_DT_SECONDS
        ),
        "lqr_update_interval_seconds": CLOSING_LQR_DT_SECONDS,
        "source_transition_count_requested": args.source_transition_count,
        "source_seed_start": args.source_seed_start,
        "source_max_episodes": args.source_max_episodes,
        "prefill_exploration": {
            "residual_std": args.prefill_residual_std,
            "residual_clip": args.prefill_residual_clip,
            "residual_hold_steps": args.prefill_residual_hold_steps,
            "zero_episode_fraction": args.prefill_zero_episode_fraction,
            "minimum_zero_episodes": args.prefill_min_zero_episodes,
            "minimum_exploratory_episodes": (
                args.prefill_min_exploratory_episodes
            ),
            "role_assignment": "required roles first, then seeded Bernoulli",
            "random_seed": args.prefill_random_seed,
            "retain_failure_episodes": args.retain_failure_prefill,
            "hard_safety_termination": True,
        },
        "training_seed": args.training_seed,
        "validation_seeds": args.validation_seeds,
        "total_timesteps": args.total_timesteps,
        "checkpoint_steps": [0, *checkpoints],
        "max_time": args.max_time,
        "max_eval_steps": args.max_eval_steps,
        "initial_gate": {
            "maximum_failure_rate": args.maximum_gate_failure_rate,
            "maximum_min_position_error": args.maximum_gate_min_position_error,
        },
        "online_guard_maximum_failure_rate": (
            args.maximum_online_validation_failure_rate
        ),
        "training_log_interval_steps": args.log_interval_steps,
        "cbf_filter_used_during_training": False,
        "reward": (
            {
                "design": (
                    "dt01_terminal_fade_action_cost_v2"
                    if terminal_fade
                    else "dt01_lateral_progress_v1"
                ),
                "progress_form": "J(previous) - gamma * J(next)",
                "progress_weight": args.progress_weight,
                "goal_component_weights": list(args.goal_component_weights),
                "worst_component_weight": args.worst_component_weight,
                "lateral_constraint_progress_weight": (
                    args.lateral_constraint_progress_weight
                ),
                "success_bonus": args.success_bonus,
                "failure_penalty": args.failure_penalty,
                "timeout_penalty": args.timeout_penalty,
                "timeout_violation_weight": args.timeout_violation_weight,
                "terminal_terms_time_scaled": False,
                "her_previous_goal_recorded": True,
                **(
                    {
                        "residual_magnitude_weight": (
                            args.residual_magnitude_weight
                        ),
                        "residual_smoothness_weight": (
                            args.residual_smoothness_weight
                        ),
                        "magnitude_time_scale": CLOSING_LQR_DT_SECONDS,
                        "smoothness_time_scale": (
                            1.0 / CLOSING_LQR_DT_SECONDS
                        ),
                        "smoothness_is_action_rate_cost": True,
                    }
                    if terminal_fade
                    else {}
                ),
            }
            if progress_reward
            else {
                "design": "absolute_cost_dt01_v2",
                "persistent_terms_time_scaled": True,
                "reference_dt_seconds": 1.0,
                "persistent_time_scale": CLOSING_LQR_DT_SECONDS,
                "terminal_terms_time_scaled": False,
            }
        ),
        "action_hold_steps": 1,
        "sac": {
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "learning_starts": 0,
            "batch_size": args.batch_size,
            "tau": args.tau,
            "gamma": args.gamma,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "ent_coef": args.ent_coef,
            "target_entropy": args.target_entropy,
            "replay_buffer": (
                "HerReplayBuffer" if use_her else "DictReplayBuffer"
            ),
            "critic_loss_type": args.critic_loss_type,
            "max_critic_grad_norm": args.max_critic_grad_norm,
            "max_actor_grad_norm": args.max_actor_grad_norm,
            "update_to_data_ratio": float(
                args.gradient_steps / args.train_freq
            ),
        },
        "her": {
            "used": use_her,
            "n_sampled_goal": args.n_sampled_goal if use_her else 0,
            "goal_selection_strategy": "future" if use_her else None,
            "copy_info_dict": bool(use_her),
        },
    }
    if args.dry_run:
        import json

        print(json.dumps(config, indent=2, sort_keys=True))
        return

    model_dir = MODEL_ROOT / args.experiment_name
    result_dir = RESULT_ROOT / args.experiment_name
    log_dir = LOG_ROOT / args.experiment_name
    for path in (model_dir, result_dir, log_dir):
        if path.exists():
            raise FileExistsError(f"Refusing to reuse output directory: {path}")
    for path in (model_dir, result_dir, log_dir):
        path.mkdir(parents=True)
    write_json(result_dir / "experiment_config.json", config)

    transitions, episode_rows = collect_lqr_residual_trajectories(
        minimum_transitions=args.source_transition_count,
        seed_start=args.source_seed_start,
        max_episodes=args.source_max_episodes,
        max_steps=steps_for_duration(args.max_time),
        residual_scale=args.residual_scale,
        residual_zero_distance=args.residual_zero_distance,
        residual_full_distance=args.residual_full_distance,
        environment_residual_hold_steps=args.residual_hold_steps,
        residual_std=args.prefill_residual_std,
        residual_clip=args.prefill_residual_clip,
        residual_hold_steps=args.prefill_residual_hold_steps,
        zero_episode_fraction=args.prefill_zero_episode_fraction,
        minimum_zero_episodes=args.prefill_min_zero_episodes,
        minimum_exploratory_episodes=args.prefill_min_exploratory_episodes,
        random_seed=args.prefill_random_seed,
        retain_failure_episodes=args.retain_failure_prefill,
        terminate_on_success=not use_her,
        reward_version=args.reward_version,
        reward_kwargs=reward_kwargs,
    )
    write_csv(result_dir / "source_episode_collection.csv", episode_rows)

    training_env = DummyVecEnv(
        [
            lambda: Monitor(
                make_her_env(
                    seed=args.training_seed,
                    mode="training",
                    max_time=args.max_time,
                    reward_version=args.reward_version,
                    dt=CLOSING_LQR_DT_SECONDS,
                    residual_scale=args.residual_scale,
                    residual_zero_distance=args.residual_zero_distance,
                    residual_full_distance=args.residual_full_distance,
                    residual_hold_steps=args.residual_hold_steps,
                    terminate_on_success=not use_her,
                    **reward_kwargs,
                )
            )
        ]
    )
    replay_buffer_options = {}
    if use_her:
        replay_buffer_options = {
            "replay_buffer_class": HerReplayBuffer,
            "replay_buffer_kwargs": {
                "n_sampled_goal": args.n_sampled_goal,
                "goal_selection_strategy": "future",
                "copy_info_dict": True,
            },
        }
    model = ZeroResidualConstrainedSAC(
        "MultiInputPolicy",
        training_env,
        **replay_buffer_options,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=0,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        target_entropy=args.target_entropy,
        critic_warmup_steps=args.critic_warmup_steps,
        critic_loss_type=args.critic_loss_type,
        max_critic_grad_norm=args.max_critic_grad_norm,
        zero_residual_actor_weight=args.zero_residual_actor_weight,
        zero_residual_actor_final_weight=(
            args.zero_residual_actor_final_weight
        ),
        zero_residual_actor_anneal_end_step=(
            args.zero_residual_actor_anneal_end_step
        ),
        actor_learning_rate_scale=args.actor_learning_rate_scale,
        max_actor_grad_norm=args.max_actor_grad_norm,
        verbose=1,
        seed=args.training_seed,
        tensorboard_log=str(log_dir),
        device=args.device,
    )
    initialize_zero_residual_actor(model, args.initial_log_std)
    prefill_metrics = prefill_her_replay_buffer(model, transitions)
    prefill_metrics["replay_buffer_class"] = type(model.replay_buffer).__name__
    del transitions
    accepted_episode_rows = [row for row in episode_rows if row["accepted"]]
    prefill_metrics.update(
        {
            "prefill_zero_episode_count": sum(
                bool(row["zero_residual_episode"])
                for row in accepted_episode_rows
            ),
            "prefill_exploratory_episode_count": sum(
                not bool(row["zero_residual_episode"])
                for row in accepted_episode_rows
            ),
            "prefill_role_requirements_passed": bool(
                sum(
                    bool(row["zero_residual_episode"])
                    for row in accepted_episode_rows
                )
                >= args.prefill_min_zero_episodes
                and sum(
                    not bool(row["zero_residual_episode"])
                    for row in accepted_episode_rows
                )
                >= args.prefill_min_exploratory_episodes
            ),
        }
    )

    env_kwargs = {
        "residual_scale": args.residual_scale,
        "residual_zero_distance": args.residual_zero_distance,
        "residual_full_distance": args.residual_full_distance,
        "residual_hold_steps": args.residual_hold_steps,
        "dt": CLOSING_LQR_DT_SECONDS,
        "terminate_on_success": True,
        **reward_kwargs,
    }
    initial_rows = evaluate_model(
        model,
        args.validation_seeds,
        max_steps=args.max_eval_steps,
        reward_version=args.reward_version,
        env_kwargs=env_kwargs,
    )
    initial_summary = summarize_rollouts(initial_rows)
    write_csv(result_dir / "zero_residual_validation_rollouts.csv", initial_rows)
    write_json(
        result_dir / "source_initialization.json",
        {**prefill_metrics, "zero_residual_validation": initial_summary},
    )
    if initial_summary["failure_rate"] > args.maximum_gate_failure_rate:
        raise RuntimeError("Exact zero-residual LQR failed the safety gate.")
    if (
        initial_summary["mean_minimum_position_error"]
        > args.maximum_gate_min_position_error
    ):
        raise RuntimeError(
            "Exact zero-residual LQR failed the approach gate: "
            f"{initial_summary['mean_minimum_position_error']:.2f} m."
        )

    initial_rank = checkpoint_rank(initial_summary)
    model.save(model_dir / "checkpoint_0_model")
    model.save(model_dir / "best_model")
    write_json(
        result_dir / "best_checkpoint.json",
        {"checkpoint": 0, "summary": initial_summary},
    )
    write_pilot_a_model_readme(
        model_dir=model_dir,
        config=config,
        checkpoint_summaries=[{"checkpoint": 0, **initial_summary}],
        best_checkpoint=0,
        run_complete=False,
    )
    callback = ResidualSafetyGuardCallback(
        maximum_failure_rate=args.maximum_online_validation_failure_rate,
        checkpoint_steps=checkpoints,
        validation_seeds=args.validation_seeds,
        max_eval_steps=args.max_eval_steps,
        model_dir=model_dir,
        result_dir=result_dir,
        initial_rank=initial_rank,
        initial_summary=initial_summary,
        initial_rows=initial_rows,
        reward_version=args.reward_version,
        env_kwargs=env_kwargs,
    )
    callback._save_tables()
    training_callbacks = CallbackList(
        [
            callback,
            PeriodicTrainingLoggerCallback(args.log_interval_steps),
        ]
    )
    print(f"Experiment:       {args.experiment_name}")
    print(
        f"Exact LQR gate:    failure={initial_summary['failure_rate']:.3f}, "
        f"min P3 error={initial_summary['mean_minimum_position_error']:.2f} m"
    )
    print(f"Residual scale:   {args.residual_scale:.3f}")
    replay_label = "HER" if use_her else "fixed-P3"
    print(f"Online {replay_label} steps: {args.total_timesteps:,}")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=training_callbacks,
        tb_log_name=args.experiment_name,
        progress_bar=args.progress_bar,
    )
    model.save(model_dir / "final_model")
    config.update(
        {
            "actual_timesteps": int(model.num_timesteps),
            "best_checkpoint": int(callback.best_checkpoint),
            "guard_triggered": callback.guard_triggered,
            "guard_reason": callback.guard_reason,
            "source_prefill": prefill_metrics,
        }
    )
    write_json(result_dir / "experiment_config.json", config)
    write_pilot_a_model_readme(
        model_dir=model_dir,
        config=config,
        checkpoint_summaries=callback.checkpoint_summaries,
        best_checkpoint=callback.best_checkpoint,
        run_complete=True,
    )
    training_env.close()


if __name__ == "__main__":
    main()
