"""
Evaluate multiple SAC final-approach models over a common evaluation seed set.

This is intended for training-seed robustness checks after running
train_sac_final_multiseed.py.
"""

import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np

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
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "eval_multiseed")


def model_name_for_seed(seed, total_timesteps):
    steps_label = f"{total_timesteps // 1000}ksteps"
    return f"sac_final_approach_portobs_rewardv9_{steps_label}_seed{seed}_optuna12"


def make_eval_env(seed=0, mode="testing"):
    return Docking3DoFGymEnv(
        phase="final",
        mode=mode,
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
    )


def load_vecnormalize_stats(vecnorm_path):
    dummy_env = DummyVecEnv([lambda: Monitor(make_eval_env(seed=999))])
    vecnorm = VecNormalize.load(vecnorm_path, dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return vecnorm, dummy_env


def normalize_observation(vecnorm, obs):
    return vecnorm.normalize_obs(obs.reshape(1, -1))


def run_one_rollout(model, vecnorm, eval_seed, mode, max_steps):
    env = make_eval_env(seed=eval_seed, mode=mode)
    obs, info = env.reset(seed=eval_seed)
    final_info = info

    port_distances = []
    port_speeds = []
    s_values = []
    rho_values = []
    s_dot_values = []
    psi_e_values = []
    omega_values = []

    for step in range(max_steps):
        norm_obs = normalize_observation(vecnorm, obs)
        action, _ = model.predict(norm_obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action[0])

        final_info = info
        port_distances.append(info.get("port_distance", np.nan))
        port_speeds.append(info.get("port_speed", np.nan))
        s_values.append(info.get("s", np.nan))
        rho_values.append(info.get("rho", np.nan))
        s_dot_values.append(info.get("s_dot", np.nan))
        psi_e_values.append(info.get("psi_e", np.nan))
        omega_values.append(info.get("omega", np.nan))

        if terminated or truncated:
            break

    env.close()

    port_distances = np.array(port_distances, dtype=np.float64)
    port_speeds = np.array(port_speeds, dtype=np.float64)
    s_values = np.array(s_values, dtype=np.float64)
    rho_values = np.array(rho_values, dtype=np.float64)
    s_dot_values = np.array(s_dot_values, dtype=np.float64)
    psi_e_values = np.array(psi_e_values, dtype=np.float64)
    omega_values = np.array(omega_values, dtype=np.float64)

    return {
        "eval_seed": eval_seed,
        "steps": step + 1,
        "success": bool(final_info.get("success", False)),
        "failure": bool(final_info.get("failure", False)),
        "time_limit": bool(final_info.get("time_limit", False)),
        "termination_reason": final_info.get("termination_reason"),
        "violated_constraints": ",".join(final_info.get("violated_constraints", [])),
        "final_port_distance": float(port_distances[-1]),
        "min_port_distance": float(np.nanmin(port_distances)),
        "final_port_speed": float(port_speeds[-1]),
        "max_port_speed": float(np.nanmax(port_speeds)),
        "final_s": float(s_values[-1]),
        "min_s": float(np.nanmin(s_values)),
        "final_rho": float(rho_values[-1]),
        "max_rho": float(np.nanmax(rho_values)),
        "final_s_dot": float(s_dot_values[-1]),
        "max_abs_s_dot": float(np.nanmax(np.abs(s_dot_values))),
        "final_psi_e_deg": float(np.rad2deg(psi_e_values[-1])),
        "max_abs_psi_e_deg": float(np.rad2deg(np.nanmax(np.abs(psi_e_values)))),
        "final_omega_deg_s": float(np.rad2deg(omega_values[-1])),
        "max_abs_omega_deg_s": float(np.rad2deg(np.nanmax(np.abs(omega_values)))),
    }


def summarize_model(train_seed, model_name, results):
    reasons = Counter(r["termination_reason"] for r in results)
    constraints = Counter()

    for result in results:
        for constraint in result["violated_constraints"].split(","):
            if constraint:
                constraints[constraint] += 1

    return {
        "train_seed": train_seed,
        "model_name": model_name,
        "n_eval": len(results),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "failure_rate": float(np.mean([r["failure"] for r in results])),
        "time_limit_rate": float(np.mean([r["time_limit"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "mean_final_port_distance": float(
            np.nanmean([r["final_port_distance"] for r in results])
        ),
        "max_final_port_distance": float(
            np.nanmax([r["final_port_distance"] for r in results])
        ),
        "mean_min_port_distance": float(
            np.nanmean([r["min_port_distance"] for r in results])
        ),
        "mean_final_rho": float(np.nanmean([r["final_rho"] for r in results])),
        "max_final_rho": float(np.nanmax([r["final_rho"] for r in results])),
        "mean_max_rho": float(np.nanmean([r["max_rho"] for r in results])),
        "termination_reasons": ";".join(
            f"{reason}:{count}" for reason, count in reasons.items()
        ),
        "violated_constraints": ";".join(
            f"{constraint}:{count}" for constraint, count in constraints.items()
        ),
    }


def write_rows(csv_path, fieldnames, rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text_summary(summary_path, model_summaries):
    success_rates = [s["success_rate"] for s in model_summaries]
    failure_rates = [s["failure_rate"] for s in model_summaries]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SAC final-approach training-seed robustness summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Number of trained models: {len(model_summaries)}\n")
        f.write(f"Mean success rate:        {np.mean(success_rates):.4f}\n")
        f.write(f"Std success rate:         {np.std(success_rates):.4f}\n")
        f.write(f"Mean failure rate:        {np.mean(failure_rates):.4f}\n")
        f.write(f"Std failure rate:         {np.std(failure_rates):.4f}\n\n")

        f.write("Per-model summary:\n")
        for summary in model_summaries:
            f.write(
                f"train_seed={summary['train_seed']:<3} | "
                f"success={summary['success_rate']:.4f} | "
                f"failure={summary['failure_rate']:.4f} | "
                f"time_limit={summary['time_limit_rate']:.4f} | "
                f"mean_final_port={summary['mean_final_port_distance']:.4f} m | "
                f"mean_final_rho={summary['mean_final_rho']:.4f} m | "
                f"reasons={summary['termination_reasons']} | "
                f"constraints={summary['violated_constraints']}\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple SAC models over common eval seeds."
    )
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--train-seed-end", type=int, default=9)
    parser.add_argument("--eval-seed-start", type=int, default=0)
    parser.add_argument("--eval-seed-end", type=int, default=49)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--mode", default="testing", choices=["nominal", "testing", "training"])
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip models that have not been trained yet instead of failing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.train_seed_end < args.train_seed_start:
        raise ValueError("--train-seed-end must be >= --train-seed-start.")
    if args.eval_seed_end < args.eval_seed_start:
        raise ValueError("--eval-seed-end must be >= --eval-seed-start.")

    run_dir = os.path.join(
        RESULTS_DIR,
        f"train_seeds_{args.train_seed_start}_{args.train_seed_end}",
        f"{args.mode}_eval_seeds_{args.eval_seed_start}_{args.eval_seed_end}",
    )
    os.makedirs(run_dir, exist_ok=True)

    all_rollouts_path = os.path.join(run_dir, "all_rollouts.csv")
    model_summary_path = os.path.join(run_dir, "model_summary.csv")
    text_summary_path = os.path.join(run_dir, "summary.txt")

    all_rollouts = []
    model_summaries = []

    print("=" * 80)
    print("Evaluating SAC final-approach models over multiple training seeds")
    print("=" * 80)
    print(f"Training seeds:   {args.train_seed_start}..{args.train_seed_end}")
    print(f"Evaluation seeds: {args.eval_seed_start}..{args.eval_seed_end}")
    print(f"Mode:             {args.mode}")
    print(f"Results dir:      {run_dir}")
    print("=" * 80)

    for train_seed in range(args.train_seed_start, args.train_seed_end + 1):
        model_name = model_name_for_seed(train_seed, args.total_timesteps)
        model_path = os.path.join(MODEL_DIR, model_name + ".zip")
        vecnorm_path = os.path.join(MODEL_DIR, model_name + "_vecnormalize.pkl")

        if not os.path.exists(model_path) or not os.path.exists(vecnorm_path):
            message = (
                f"Missing model files for training seed {train_seed}: "
                f"{model_path}, {vecnorm_path}"
            )
            if args.skip_missing:
                print(message)
                print("Skipping.")
                continue
            raise FileNotFoundError(message)

        print("-" * 80)
        print(f"Training seed {train_seed} | model={model_name}")

        vecnorm, dummy_env = load_vecnormalize_stats(vecnorm_path)
        model = SAC.load(model_path)

        model_results = []
        for eval_seed in range(args.eval_seed_start, args.eval_seed_end + 1):
            result = run_one_rollout(
                model=model,
                vecnorm=vecnorm,
                eval_seed=eval_seed,
                mode=args.mode,
                max_steps=args.max_steps,
            )
            result["train_seed"] = train_seed
            result["model_name"] = model_name
            model_results.append(result)
            all_rollouts.append(result)

            print(
                f"  eval_seed={eval_seed:<3} | "
                f"success={result['success']} | "
                f"failure={result['failure']} | "
                f"reason={result['termination_reason']} | "
                f"final_port={result['final_port_distance']:.4f} m"
            )

        dummy_env.close()

        summary = summarize_model(train_seed, model_name, model_results)
        model_summaries.append(summary)

        print(
            f"Summary seed {train_seed}: "
            f"success={summary['success_rate']:.4f}, "
            f"failure={summary['failure_rate']:.4f}, "
            f"time_limit={summary['time_limit_rate']:.4f}"
        )

    if not model_summaries:
        raise RuntimeError("No models were evaluated.")

    rollout_fieldnames = [
        "train_seed",
        "model_name",
        "eval_seed",
        "steps",
        "success",
        "failure",
        "time_limit",
        "termination_reason",
        "violated_constraints",
        "final_port_distance",
        "min_port_distance",
        "final_port_speed",
        "max_port_speed",
        "final_s",
        "min_s",
        "final_rho",
        "max_rho",
        "final_s_dot",
        "max_abs_s_dot",
        "final_psi_e_deg",
        "max_abs_psi_e_deg",
        "final_omega_deg_s",
        "max_abs_omega_deg_s",
    ]
    summary_fieldnames = [
        "train_seed",
        "model_name",
        "n_eval",
        "success_rate",
        "failure_rate",
        "time_limit_rate",
        "mean_steps",
        "mean_final_port_distance",
        "max_final_port_distance",
        "mean_min_port_distance",
        "mean_final_rho",
        "max_final_rho",
        "mean_max_rho",
        "termination_reasons",
        "violated_constraints",
    ]

    write_rows(all_rollouts_path, rollout_fieldnames, all_rollouts)
    write_rows(model_summary_path, summary_fieldnames, model_summaries)
    write_text_summary(text_summary_path, model_summaries)

    success_rates = [s["success_rate"] for s in model_summaries]
    failure_rates = [s["failure_rate"] for s in model_summaries]

    print("\n" + "=" * 80)
    print("Training-seed robustness summary")
    print("=" * 80)
    print(f"Models evaluated:  {len(model_summaries)}")
    print(f"Mean success rate: {np.mean(success_rates):.4f}")
    print(f"Std success rate:  {np.std(success_rates):.4f}")
    print(f"Mean failure rate: {np.mean(failure_rates):.4f}")
    print(f"Std failure rate:  {np.std(failure_rates):.4f}")
    print(f"Saved rollouts:    {all_rollouts_path}")
    print(f"Saved summaries:   {model_summary_path}")
    print(f"Saved text:        {text_summary_path}")


if __name__ == "__main__":
    main()
