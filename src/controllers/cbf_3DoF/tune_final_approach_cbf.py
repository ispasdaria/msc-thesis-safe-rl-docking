"""Reproducible Optuna tuning for final-approach CBF response gains.

This script tunes the cone, speed, attitude, and angular-rate response gains.
The physical cone slope, speed limit, attitude limit, and angular-rate limit
remain fixed safety specifications.
Unsafe, inadmissible, discretely violating, or solver-failing configurations
are pruned rather than traded against docking success.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import optuna
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from controllers.cbf_3DoF.barriers import (
    FinalApproachCBFConfig,
    build_final_approach_constraints,
)
from controllers.cbf_3DoF.diagnostics import (
    attitude_hocbf_admissibility,
    cone_hocbf_admissibility,
    hard_cbf_feasibility_margin,
)
from controllers.cbf_3DoF.qp_filter import FinalApproachCBFFilter
from envs.docking3dof.core import Docking3DoFEnv
from envs.docking3dof.gym_env import Docking3DoFGymEnv


MODEL_DIR = (
    SRC_PATH
    / "controllers"
    / "SAC-3DoF"
    / "final_approach"
    / "models"
    / "optuna12_600ksteps_buffer600k"
)
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "tuning"
MODEL_TEMPLATE = (
    "sac_final_approach_portobs_rewardv9_600ksteps_"
    "seed{seed}_optuna12_buffer600k_best"
)
J = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
GAIN_SEARCH_PLAN_FILENAME = "gain_search_plan.json"
OPTUNA_STUDY_FILENAME = "optuna_gain_search.sqlite3"
GAIN_SEARCH_TRIALS_FILENAME = "gain_search_trial_results.csv"
SELECTED_CONFIG_FILENAME = "selected_cbf_gain_configuration.json"


@dataclass
class ControllerBundle:
    train_seed: int
    model: SAC
    vecnormalize: VecNormalize
    dummy_env: DummyVecEnv

    def close(self):
        self.dummy_env.close()


@dataclass
class AggregateMetrics:
    episodes: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    total_steps: int = 0
    interventions: int = 0
    intervention_scaled_sum: float = 0.0
    intervention_scaled_max: float = 0.0
    actuator_saturation_steps: int = 0
    qp_failures: int = 0
    protected_violations: int = 0
    alignment_failures: int = 0
    min_post_barrier: float = np.inf

    def update_step(self, result, action_limits, post_barrier_min):
        self.total_steps += 1
        self.interventions += int(result.intervened)
        self.intervention_scaled_sum += result.intervention_scaled_norm
        self.intervention_scaled_max = max(
            self.intervention_scaled_max,
            result.intervention_scaled_norm,
        )
        normalized = np.abs(result.action_filtered / action_limits)
        self.actuator_saturation_steps += int(np.any(normalized >= 1.0 - 1.0e-7))
        self.qp_failures += int(result.qp_failed)
        self.min_post_barrier = min(self.min_post_barrier, post_barrier_min)

    def as_rates(self):
        steps = max(self.total_steps, 1)
        episodes = max(self.episodes, 1)
        return {
            "episodes": self.episodes,
            "success_rate": self.successes / episodes,
            "failure_rate": self.failures / episodes,
            "timeout_rate": self.timeouts / episodes,
            "mean_steps": self.total_steps / episodes,
            "intervention_fraction": self.interventions / steps,
            "mean_intervention_scaled_norm": self.intervention_scaled_sum / steps,
            "max_intervention_scaled_norm": self.intervention_scaled_max,
            "actuator_saturation_fraction": self.actuator_saturation_steps / steps,
            "qp_failures": self.qp_failures,
            "protected_violations": self.protected_violations,
            "alignment_failures": self.alignment_failures,
            "min_post_barrier": self.min_post_barrier,
        }


def make_env(seed, mode, dt):
    return Docking3DoFGymEnv(
        phase="final",
        mode=mode,
        dt=dt,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
    )


def load_controller(train_seed, mode, dt):
    stem = MODEL_TEMPLATE.format(seed=train_seed)
    model_path = MODEL_DIR / f"{stem}.zip"
    vecnormalize_path = MODEL_DIR / f"{stem}_vecnormalize.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing SAC checkpoint: {model_path}")
    if not vecnormalize_path.is_file():
        raise FileNotFoundError(f"Missing VecNormalize statistics: {vecnormalize_path}")

    dummy = DummyVecEnv(
        [lambda: Monitor(make_env(seed=999_999 + train_seed, mode=mode, dt=dt))]
    )
    vecnormalize = VecNormalize.load(str(vecnormalize_path), dummy)
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    return ControllerBundle(
        train_seed=train_seed,
        model=SAC.load(str(model_path)),
        vecnormalize=vecnormalize,
        dummy_env=dummy,
    )


def state_from_port_quantities(env, s, y_p, v_p, psi=0.0, omega=0.0):
    rotation = env.rotation_body_to_orbital_2d(psi)
    offset = rotation @ env.r_c_body
    position = np.array([s, y_p]) - offset + env.r_t_orbital
    velocity = np.asarray(v_p, dtype=np.float64) - omega * (J @ offset)
    return np.concatenate((position, velocity, [psi, omega]))


def create_grid_search_space(
    cone_k_values,
    speed_alpha_values,
    cone_k2_values=None,
    attitude_k_values=(0.3,),
    angular_rate_alpha_values=(0.5,),
):
    """Create a tied-gain grid unless independent k2 values are requested."""
    if cone_k2_values is None:
        search_space = {
            "cone_k": list(cone_k_values),
            "speed_alpha": list(speed_alpha_values),
        }
    else:
        search_space = {
            "cone_k1": list(cone_k_values),
            "cone_k2": list(cone_k2_values),
            "speed_alpha": list(speed_alpha_values),
        }
    search_space["attitude_k"] = list(attitude_k_values)
    search_space["angular_rate_alpha"] = list(angular_rate_alpha_values)
    return search_space


def config_from_trial(trial, independent_k2):
    if independent_k2:
        cone_k1 = trial.suggest_float("cone_k1", 1.0e-6, 10.0)
        cone_k2 = trial.suggest_float("cone_k2", 1.0e-6, 10.0)
    else:
        cone_k1 = trial.suggest_float("cone_k", 1.0e-6, 10.0)
        cone_k2 = cone_k1
    speed_alpha = trial.suggest_float("speed_alpha", 1.0e-6, 10.0)
    attitude_k = trial.suggest_float("attitude_k", 1.0e-6, 10.0)
    angular_rate_alpha = trial.suggest_float(
        "angular_rate_alpha",
        1.0e-6,
        10.0,
    )
    return FinalApproachCBFConfig(
        cone_k1=cone_k1,
        cone_k2=cone_k2,
        speed_alpha=speed_alpha,
        attitude_k1=attitude_k,
        attitude_k2=attitude_k,
        angular_rate_alpha=angular_rate_alpha,
        allow_slack=False,
        solver_backend="osqp",
    )


def boundary_precheck(config, dt=1.0, tolerance=1.0e-6):
    """Exercise admissible exact boundaries before expensive SAC rollouts."""
    env = Docking3DoFEnv(phase="final", dt=dt, integrator="rk4", seed=0)
    cbf_filter = FinalApproachCBFFilter(env, config)
    cases = []

    for s in (300.0, 100.0, 20.0, 5.0, 1.0):
        # Select a closing velocity that remains strictly viable for every k1.
        s_dot = -min(0.1, 0.5 * config.cone_k1 * s)
        for side in (-1.0, 1.0):
            cases.append(
                (
                    f"cone_s{s:g}_side{side:+g}",
                    state_from_port_quantities(
                        env,
                        s=s,
                        y_p=side * config.kappa * s,
                        v_p=np.array([s_dot, side * config.kappa * s_dot]),
                    ),
                    np.array(
                        [env.ax_max, -side * env.ay_max, -side * env.alphaz_max]
                    ),
                )
            )

    tangent = np.array([-1.0, -config.kappa])
    tangent *= config.port_speed_max / np.linalg.norm(tangent)
    cases.append(
        (
            "simultaneous_upper_cone_speed",
            state_from_port_quantities(
                env,
                s=20.0,
                y_p=config.kappa * 20.0,
                v_p=tangent,
            ),
            np.array([env.ax_max, -env.ay_max, -env.alphaz_max]),
        )
    )

    for side in (-1.0, 1.0):
        cases.append(
            (
                f"attitude_side{side:+g}",
                state_from_port_quantities(
                    env,
                    s=100.0,
                    y_p=0.0,
                    v_p=np.array([-0.03, 0.0]),
                    psi=side * config.attitude_max,
                    omega=0.0,
                ),
                np.array([0.0, 0.0, side * env.alphaz_max]),
            )
        )
        cases.append(
            (
                f"angular_rate_side{side:+g}",
                state_from_port_quantities(
                    env,
                    s=100.0,
                    y_p=0.0,
                    v_p=np.array([-0.03, 0.0]),
                    psi=0.0,
                    omega=side * config.angular_rate_max,
                ),
                np.array([0.0, 0.0, side * env.alphaz_max]),
            )
        )
    cases.append(
        (
            "axial_speed",
            state_from_port_quantities(
                env,
                s=100.0,
                y_p=0.0,
                v_p=np.array([-config.port_speed_max, 0.0]),
            ),
            np.array([env.ax_max, 0.0, 0.0]),
        )
    )

    minimum_margin = np.inf
    minimum_post_barrier = np.inf
    for name, state, nominal in cases:
        constraints = build_final_approach_constraints(env, state, config)
        psi_1 = cone_hocbf_admissibility(constraints, config)
        psi_1.update(attitude_hocbf_admissibility(constraints, config))
        if min(constraints.barrier_values.values()) < -tolerance:
            return False, f"{name}:initial_barrier", minimum_margin, minimum_post_barrier
        if min(psi_1.values()) < -tolerance:
            return False, f"{name}:initial_hocbf", minimum_margin, minimum_post_barrier

        feasibility = hard_cbf_feasibility_margin(constraints, env.action_high)
        minimum_margin = min(minimum_margin, feasibility.margin)
        if not feasibility.feasible:
            return False, f"{name}:hard_infeasible", minimum_margin, minimum_post_barrier

        result = cbf_filter.filter(state, nominal)
        if result.qp_failed:
            return False, f"{name}:qp_failure", minimum_margin, minimum_post_barrier
        if min(result.cbf_residuals.values()) < -tolerance:
            return False, f"{name}:negative_residual", minimum_margin, minimum_post_barrier

        next_state = env.integrate(state, result.action_filtered)
        next_constraints = build_final_approach_constraints(env, next_state, config)
        post_minimum = min(next_constraints.barrier_values.values())
        minimum_post_barrier = min(minimum_post_barrier, post_minimum)
        if post_minimum < -tolerance:
            return False, f"{name}:discrete_crossing", minimum_margin, minimum_post_barrier

    return True, "passed", minimum_margin, minimum_post_barrier


def reject_trial(trial, reason, metrics=None):
    trial.set_user_attr("rejection_reason", reason)
    if metrics is not None:
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
    raise optuna.TrialPruned(reason)


def evaluate_trial(
    trial,
    config,
    bundles,
    eval_seeds,
    mode,
    dt,
    max_steps,
    barrier_tolerance,
    run_precheck,
):
    if run_precheck:
        passed, reason, margin, post_minimum = boundary_precheck(
            config,
            dt=dt,
            tolerance=barrier_tolerance,
        )
        trial.set_user_attr("boundary_min_feasibility_margin", margin)
        trial.set_user_attr("boundary_min_post_barrier", post_minimum)
        if not passed:
            reject_trial(trial, f"boundary:{reason}")

    aggregate = AggregateMetrics()
    controller_summaries = {}
    termination_reasons = Counter()

    for bundle in bundles:
        controller = AggregateMetrics()
        for eval_seed in eval_seeds:
            env = make_env(seed=eval_seed, mode=mode, dt=dt)
            observation, final_info = env.reset(seed=eval_seed)
            initial_constraints = build_final_approach_constraints(
                env.core_env,
                env.get_physical_state(),
                config,
            )
            psi_1 = cone_hocbf_admissibility(initial_constraints, config)
            psi_1.update(
                attitude_hocbf_admissibility(initial_constraints, config)
            )
            if (
                min(initial_constraints.barrier_values.values()) < -barrier_tolerance
                or min(psi_1.values()) < -barrier_tolerance
            ):
                env.close()
                reject_trial(
                    trial,
                    f"initial_state_not_admissible:model{bundle.train_seed}:seed{eval_seed}",
                )

            cbf_filter = FinalApproachCBFFilter(env.core_env, config)
            for episode_step in range(max_steps):
                state = env.get_physical_state()
                normalized_observation = bundle.vecnormalize.normalize_obs(
                    observation.reshape(1, -1)
                )
                normalized_action, _ = bundle.model.predict(
                    normalized_observation,
                    deterministic=True,
                )
                nominal = env.normalized_action_to_physical(normalized_action[0])
                result = cbf_filter.filter(state, nominal)
                if result.qp_failed:
                    aggregate.qp_failures += 1
                    controller.qp_failures += 1
                    env.close()
                    reject_trial(
                        trial,
                        f"qp_failure:model{bundle.train_seed}:seed{eval_seed}",
                    )

                action = env.physical_action_to_normalized(result.action_filtered)
                observation, _, terminated, truncated, final_info = env.step(action)
                post_constraints = build_final_approach_constraints(
                    env.core_env,
                    env.get_physical_state(),
                    config,
                )
                post_minimum = min(post_constraints.barrier_values.values())
                aggregate.update_step(result, env.get_physical_action_limits(), post_minimum)
                controller.update_step(result, env.get_physical_action_limits(), post_minimum)

                if min(result.cbf_residuals.values()) < -barrier_tolerance:
                    env.close()
                    reject_trial(
                        trial,
                        f"negative_qp_residual:model{bundle.train_seed}:seed{eval_seed}",
                    )
                violating_barriers = {
                    name: float(value)
                    for name, value in post_constraints.barrier_values.items()
                    if value < -barrier_tolerance
                }
                if violating_barriers:
                    aggregate.protected_violations += 1
                    controller.protected_violations += 1
                    trial.set_user_attr(
                        "first_protected_discrete_violation",
                        {
                            "model_seed": bundle.train_seed,
                            "evaluation_seed": eval_seed,
                            "episode_step": episode_step,
                            "barriers": violating_barriers,
                            "post_port_speed": float(
                                np.linalg.norm(post_constraints.kinematics.v_p)
                            ),
                            "qp_residuals": result.cbf_residuals,
                        },
                    )
                    env.close()
                    barrier_description = ",".join(
                        f"{name}={value:.6g}"
                        for name, value in violating_barriers.items()
                    )
                    reject_trial(
                        trial,
                        "protected_discrete_violation:"
                        f"{barrier_description}:model{bundle.train_seed}:"
                        f"seed{eval_seed}:episode_step{episode_step}",
                    )
                if terminated or truncated:
                    break

            env.close()
            aggregate.episodes += 1
            controller.episodes += 1
            success = bool(final_info.get("success", False))
            failure = bool(final_info.get("failure", False))
            timed_out = bool(final_info.get("time_limit", False)) or not (success or failure)
            aggregate.successes += int(success)
            controller.successes += int(success)
            aggregate.failures += int(failure)
            controller.failures += int(failure)
            aggregate.timeouts += int(timed_out)
            controller.timeouts += int(timed_out)
            violated = final_info.get("violated_constraints", [])
            alignment_failure = failure and "alignment" in violated
            aggregate.alignment_failures += int(alignment_failure)
            controller.alignment_failures += int(alignment_failure)
            reason = final_info.get("termination_reason") or "evaluation_step_limit"
            termination_reasons[reason] += 1

        controller_summaries[str(bundle.train_seed)] = controller.as_rates()

    summary = aggregate.as_rates()
    summary["worst_controller_success_rate"] = min(
        item["success_rate"] for item in controller_summaries.values()
    )
    summary["termination_reasons"] = dict(termination_reasons)
    summary["controller_summaries"] = controller_summaries
    for key, value in summary.items():
        trial.set_user_attr(key, value)
    trial.set_user_attr("rejection_reason", "")
    return summary["success_rate"], summary["intervention_fraction"]


def select_recommended_trial(study):
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.user_attrs.get("qp_failures", 0) == 0
        and trial.user_attrs.get("protected_violations", 0) == 0
    ]
    if not completed:
        return None
    return min(
        completed,
        key=lambda trial: (
            -trial.user_attrs["success_rate"],
            -trial.user_attrs["worst_controller_success_rate"],
            trial.user_attrs["timeout_rate"],
            trial.user_attrs["intervention_fraction"],
            trial.user_attrs["max_intervention_scaled_norm"],
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune final-approach CBF gains with a reproducible Optuna grid."
    )
    parser.add_argument("--controller-seeds", nargs="+", type=int, default=[0, 1, 10, 16])
    parser.add_argument("--eval-seed-start", type=int, default=100)
    parser.add_argument("--num-eval-seeds", type=int, default=20)
    parser.add_argument("--mode", choices=("testing", "training"), default="testing")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument(
        "--cone-k-values",
        nargs="+",
        type=float,
        default=[0.03, 0.05, 0.1, 0.2, 0.3],
    )
    parser.add_argument(
        "--independent-cone-k2-values",
        nargs="+",
        type=float,
        default=None,
        help="If provided, search k1 and k2 independently; otherwise set k1=k2.",
    )
    parser.add_argument(
        "--speed-alpha-values",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.5, 1.0],
    )
    parser.add_argument(
        "--attitude-k-values",
        nargs="+",
        type=float,
        default=[0.3],
    )
    parser.add_argument(
        "--angular-rate-alpha-values",
        nargs="+",
        type=float,
        default=[0.5],
    )
    parser.add_argument("--barrier-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--skip-boundary-precheck", action="store_true")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--grid-seed", type=int, default=2026)
    parser.add_argument(
        "--study-name",
        default="final_approach_cbf_with_alignment_grid",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def validate_args(args):
    if args.num_eval_seeds <= 0 or args.max_steps <= 0 or args.dt <= 0.0:
        raise ValueError("Evaluation counts, max steps, and dt must be positive.")
    if args.barrier_tolerance < 0.0:
        raise ValueError("--barrier-tolerance must be nonnegative.")
    for name, values in (
        ("cone-k-values", args.cone_k_values),
        ("independent-cone-k2-values", args.independent_cone_k2_values),
        ("speed-alpha-values", args.speed_alpha_values),
        ("attitude-k-values", args.attitude_k_values),
        ("angular-rate-alpha-values", args.angular_rate_alpha_values),
    ):
        if values is not None and any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(f"Every --{name} entry must be finite and positive.")


def main():
    args = parse_args()
    validate_args(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (RESULTS_DIR / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    search_space = create_grid_search_space(
        args.cone_k_values,
        args.speed_alpha_values,
        args.independent_cone_k2_values,
        args.attitude_k_values,
        args.angular_rate_alpha_values,
    )
    total_grid_size = int(np.prod([len(values) for values in search_space.values()]))
    n_trials = total_grid_size if args.n_trials is None else args.n_trials
    if n_trials <= 0:
        raise ValueError("--n-trials must be positive.")

    with (output_dir / GAIN_SEARCH_PLAN_FILENAME).open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "controller_seeds": args.controller_seeds,
                "eval_seed_start": args.eval_seed_start,
                "num_eval_seeds": args.num_eval_seeds,
                "mode": args.mode,
                "dt": args.dt,
                "max_steps": args.max_steps,
                "barrier_tolerance": args.barrier_tolerance,
                "search_space": search_space,
                "total_grid_size": total_grid_size,
                "requested_trials": n_trials,
                "physical_parameters_not_tuned": {
                    "kappa": 0.1,
                    "port_speed_max": 0.15,
                    "attitude_max_deg": 3.0,
                    "angular_rate_max_deg_s": 0.15,
                },
            },
            stream,
            indent=2,
        )

    print(f"Loading {len(args.controller_seeds)} SAC controllers...")
    bundles = [load_controller(seed, args.mode, args.dt) for seed in args.controller_seeds]
    eval_seeds = range(args.eval_seed_start, args.eval_seed_start + args.num_eval_seeds)
    sampler = optuna.samplers.GridSampler(search_space, seed=args.grid_seed)
    storage = f"sqlite:///{(output_dir / OPTUNA_STUDY_FILENAME).resolve().as_posix()}"
    study = optuna.create_study(
        study_name=args.study_name,
        directions=("maximize", "minimize"),
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial):
        config = config_from_trial(
            trial,
            independent_k2=args.independent_cone_k2_values is not None,
        )
        trial.set_user_attr("cbf_config", asdict(config))
        return evaluate_trial(
            trial=trial,
            config=config,
            bundles=bundles,
            eval_seeds=eval_seeds,
            mode=args.mode,
            dt=args.dt,
            max_steps=args.max_steps,
            barrier_tolerance=args.barrier_tolerance,
            run_precheck=not args.skip_boundary_precheck,
        )

    try:
        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    finally:
        for bundle in bundles:
            bundle.close()

    dataframe = study.trials_dataframe(
        attrs=("number", "state", "values", "params", "user_attrs")
    )
    dataframe.to_csv(output_dir / GAIN_SEARCH_TRIALS_FILENAME, index=False)
    recommended = select_recommended_trial(study)
    if recommended is None:
        print("No safe completed configuration was found.")
    else:
        recommendation = {
            "trial_number": recommended.number,
            "parameters": recommended.params,
            "objectives": {
                "success_rate": recommended.values[0],
                "intervention_fraction": recommended.values[1],
            },
            "metrics": recommended.user_attrs,
        }
        with (output_dir / SELECTED_CONFIG_FILENAME).open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(recommendation, stream, indent=2)
        print("Recommended safe configuration:")
        print(json.dumps(recommendation, indent=2))

    print(f"Study database: {(output_dir / OPTUNA_STUDY_FILENAME).resolve()}")
    print(f"Trial table:    {(output_dir / GAIN_SEARCH_TRIALS_FILENAME).resolve()}")


if __name__ == "__main__":
    main()
