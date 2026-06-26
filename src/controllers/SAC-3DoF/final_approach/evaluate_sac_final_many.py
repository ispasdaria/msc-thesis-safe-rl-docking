"""
Evaluate a trained SAC final-approach model over many randomized test seeds.

This script is intended for validation/test sweeps. It does not save full
rollout histories; it saves one CSV row per evaluation seed plus a compact
text summary.
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
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "eval_many")

DEFAULT_MODEL_NAME = "sac_final_approach_portobs_rewardv9_100ksteps_seed12_optuna12"


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


def run_one_rollout(model, vecnorm, seed, mode, max_steps):
    env = make_eval_env(seed=seed, mode=mode)
    obs, info = env.reset(seed=seed)
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
        "seed": seed,
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


def write_csv(csv_path, results):
    fieldnames = [
        "seed",
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

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def summarize(results):
    success_rate = np.mean([r["success"] for r in results])
    failure_rate = np.mean([r["failure"] for r in results])
    time_limit_rate = np.mean([r["time_limit"] for r in results])

    reasons = Counter(r["termination_reason"] for r in results)
    constraints = Counter()
    for result in results:
        for constraint in result["violated_constraints"].split(","):
            if constraint:
                constraints[constraint] += 1

    return {
        "n": len(results),
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "time_limit_rate": time_limit_rate,
        "mean_steps": np.mean([r["steps"] for r in results]),
        "mean_final_port_distance": np.nanmean(
            [r["final_port_distance"] for r in results]
        ),
        "max_final_port_distance": np.nanmax(
            [r["final_port_distance"] for r in results]
        ),
        "mean_min_port_distance": np.nanmean(
            [r["min_port_distance"] for r in results]
        ),
        "mean_final_rho": np.nanmean([r["final_rho"] for r in results]),
        "max_final_rho": np.nanmax([r["final_rho"] for r in results]),
        "mean_max_rho": np.nanmean([r["max_rho"] for r in results]),
        "reasons": reasons,
        "constraints": constraints,
    }


def write_summary(summary_path, model_name, model_path, vecnorm_path, csv_path, summary):
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SAC final-approach many-seed evaluation\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model name: {model_name}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"VecNorm path: {vecnorm_path}\n")
        f.write(f"CSV path: {csv_path}\n\n")

        f.write(f"Number of rollouts: {summary['n']}\n")
        f.write(f"Success rate:       {summary['success_rate']:.4f}\n")
        f.write(f"Failure rate:       {summary['failure_rate']:.4f}\n")
        f.write(f"Time-limit rate:    {summary['time_limit_rate']:.4f}\n")
        f.write(f"Mean steps:         {summary['mean_steps']:.1f}\n\n")

        f.write("Distance / geometry summary:\n")
        f.write(
            f"Mean final port distance: {summary['mean_final_port_distance']:.4f} m\n"
        )
        f.write(
            f"Max final port distance:  {summary['max_final_port_distance']:.4f} m\n"
        )
        f.write(
            f"Mean min port distance:   {summary['mean_min_port_distance']:.4f} m\n"
        )
        f.write(f"Mean final rho:           {summary['mean_final_rho']:.4f} m\n")
        f.write(f"Max final rho:            {summary['max_final_rho']:.4f} m\n")
        f.write(f"Mean max rho:             {summary['mean_max_rho']:.4f} m\n\n")

        f.write("Termination reasons:\n")
        for reason, count in summary["reasons"].items():
            f.write(f"  {reason}: {count}\n")

        f.write("\nViolated constraints:\n")
        if summary["constraints"]:
            for constraint, count in summary["constraints"].items():
                f.write(f"  {constraint}: {count}\n")
        else:
            f.write("  none\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a SAC final-approach model over many seeds."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=199)
    parser.add_argument("--mode", default="testing", choices=["nominal", "testing", "training"])
    parser.add_argument("--max-steps", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = os.path.join(MODEL_DIR, args.model_name + ".zip")
    vecnorm_path = os.path.join(MODEL_DIR, args.model_name + "_vecnormalize.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_path}")

    run_dir = os.path.join(
        RESULTS_DIR,
        args.model_name,
        f"{args.mode}_seeds_{args.seed_start}_{args.seed_end}",
    )
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "evaluation_many.csv")
    summary_path = os.path.join(run_dir, "evaluation_summary.txt")

    print("=" * 80)
    print("Evaluating SAC final-approach model over many seeds")
    print("=" * 80)
    print(f"Model name: {args.model_name}")
    print(f"Seeds:      {args.seed_start}..{args.seed_end}")
    print(f"Mode:       {args.mode}")
    print(f"CSV:        {csv_path}")
    print("=" * 80)

    vecnorm, dummy_env = load_vecnormalize_stats(vecnorm_path)
    model = SAC.load(model_path)

    results = []
    for seed in range(args.seed_start, args.seed_end + 1):
        result = run_one_rollout(
            model=model,
            vecnorm=vecnorm,
            seed=seed,
            mode=args.mode,
            max_steps=args.max_steps,
        )
        results.append(result)

        print(
            f"seed={seed:<4} | steps={result['steps']:<5} | "
            f"success={result['success']} | failure={result['failure']} | "
            f"time_limit={result['time_limit']} | "
            f"reason={result['termination_reason']} | "
            f"final_port={result['final_port_distance']:.4f} m | "
            f"final_rho={result['final_rho']:.4f} m"
        )

    dummy_env.close()

    write_csv(csv_path, results)
    summary = summarize(results)
    write_summary(
        summary_path=summary_path,
        model_name=args.model_name,
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        csv_path=csv_path,
        summary=summary,
    )

    print("\n" + "=" * 80)
    print("Evaluation summary")
    print("=" * 80)
    print(f"Number of rollouts: {summary['n']}")
    print(f"Success rate:       {summary['success_rate']:.4f}")
    print(f"Failure rate:       {summary['failure_rate']:.4f}")
    print(f"Time-limit rate:    {summary['time_limit_rate']:.4f}")
    print(f"Mean steps:         {summary['mean_steps']:.1f}")
    print(f"Mean final port:    {summary['mean_final_port_distance']:.4f} m")
    print(f"Mean final rho:     {summary['mean_final_rho']:.4f} m")
    print(f"Saved CSV:          {csv_path}")
    print(f"Saved summary:      {summary_path}")


if __name__ == "__main__":
    main()
