"""
Evaluate a SAC final-approach model on the near-final debugging task.

This uses the same reset distribution as train_sac_final_near.py by default:
port-axial distance s in 35-70 m, small lateral offset, and low attitude/rate
error. It is meant to answer whether the near-final model can solve the last
segment, not whether it can start from the normal P3 range around 300 m.
"""

import argparse
import csv
import os
import sys
from collections import Counter

import numpy.core.multiarray as np_core_multiarray
import numpy.core.numeric as np_core_numeric
import numpy as np
import numpy.random._pickle as np_random_pickle

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from train_sac_final_near import NearFinalApproachEnv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "eval_near")

DEFAULT_MODEL_NAME = "sac_final_approach_nearfinal_gymreward3_200ksteps_seed0_best"


def install_numpy_bitgenerator_pickle_compat():
    sys.modules.setdefault("numpy._core.multiarray", np_core_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_core_numeric)

    original_ctor = np_random_pickle.__bit_generator_ctor
    original_generator_ctor = np_random_pickle.__generator_ctor

    class DummyGenerator:
        def __setstate__(self, state):
            return None

    class DummyBitGenerator:
        def __setstate__(self, state):
            return None

    def bit_generator_ctor(bit_generator_name="MT19937"):
        if not isinstance(bit_generator_name, str) and hasattr(
            bit_generator_name, "__name__"
        ):
            return DummyBitGenerator()
        return original_ctor(bit_generator_name)

    def generator_ctor(bit_generator_name="MT19937", bit_generator_ctor=None):
        if not isinstance(bit_generator_name, str):
            return DummyGenerator()
        if bit_generator_ctor is None:
            bit_generator_ctor = np_random_pickle.__bit_generator_ctor
        return original_generator_ctor(bit_generator_name, bit_generator_ctor)

    np_random_pickle.__bit_generator_ctor = bit_generator_ctor
    np_random_pickle.__generator_ctor = generator_ctor


def make_eval_env(seed, args):
    return NearFinalApproachEnv(
        s_range=(args.s_min, args.s_max),
        rho_abs_max=args.rho_abs_max,
        vp_x_range=(args.vp_x_min, args.vp_x_max),
        vp_y_abs_max=args.vp_y_abs_max,
        psi_deg_abs_max=args.psi_deg_abs_max,
        omega_deg_abs_max=args.omega_deg_abs_max,
        phase="final",
        mode="training",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
    )


def load_vecnormalize_stats(vecnorm_path, args):
    install_numpy_bitgenerator_pickle_compat()
    dummy_env = DummyVecEnv([lambda: Monitor(make_eval_env(seed=999, args=args))])
    vecnorm = VecNormalize.load(vecnorm_path, dummy_env)
    vecnorm.observation_space._np_random = None
    vecnorm.action_space._np_random = None
    vecnorm.training = False
    vecnorm.norm_reward = False
    return vecnorm, dummy_env


def normalize_observation(vecnorm, obs):
    return vecnorm.normalize_obs(obs.reshape(1, -1))


def run_one_rollout(model, vecnorm, seed, args):
    env = make_eval_env(seed=seed, args=args)
    obs, info = env.reset(seed=seed)
    initial_info = info
    final_info = info

    port_distances = []
    port_speeds = []
    s_values = []
    rho_values = []
    s_dot_values = []
    psi_e_values = []
    omega_values = []

    for step in range(args.max_steps):
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
        "initial_s": float(initial_info.get("s", np.nan)),
        "initial_rho": float(initial_info.get("rho", np.nan)),
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
        "initial_s",
        "initial_rho",
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
    reasons = Counter(r["termination_reason"] for r in results)
    constraints = Counter()
    for result in results:
        for constraint in result["violated_constraints"].split(","):
            if constraint:
                constraints[constraint] += 1

    return {
        "n": len(results),
        "success_rate": np.mean([r["success"] for r in results]),
        "failure_rate": np.mean([r["failure"] for r in results]),
        "time_limit_rate": np.mean([r["time_limit"] for r in results]),
        "mean_steps": np.mean([r["steps"] for r in results]),
        "mean_initial_s": np.nanmean([r["initial_s"] for r in results]),
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
        f.write("SAC near-final evaluation\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model name: {model_name}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"VecNorm path: {vecnorm_path}\n")
        f.write(f"CSV path: {csv_path}\n\n")

        f.write(f"Number of rollouts: {summary['n']}\n")
        f.write(f"Success rate:       {summary['success_rate']:.4f}\n")
        f.write(f"Failure rate:       {summary['failure_rate']:.4f}\n")
        f.write(f"Time-limit rate:    {summary['time_limit_rate']:.4f}\n")
        f.write(f"Mean steps:         {summary['mean_steps']:.1f}\n")
        f.write(f"Mean initial s:     {summary['mean_initial_s']:.4f} m\n\n")

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
        description="Evaluate a near-final SAC model from 35-70 m reset states."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=49)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--s-min", type=float, default=35.0)
    parser.add_argument("--s-max", type=float, default=70.0)
    parser.add_argument("--rho-abs-max", type=float, default=1.5)
    parser.add_argument("--vp-x-min", type=float, default=-0.035)
    parser.add_argument("--vp-x-max", type=float, default=-0.003)
    parser.add_argument("--vp-y-abs-max", type=float, default=0.006)
    parser.add_argument("--psi-deg-abs-max", type=float, default=1.0)
    parser.add_argument("--omega-deg-abs-max", type=float, default=0.04)
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
        f"near_seeds_{args.seed_start}_{args.seed_end}",
    )
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "evaluation_near.csv")
    summary_path = os.path.join(run_dir, "evaluation_summary.txt")

    print("=" * 80)
    print("Evaluating SAC final-approach model on near-final resets")
    print("=" * 80)
    print(f"Model name: {args.model_name}")
    print(f"Seeds:      {args.seed_start}..{args.seed_end}")
    print(f"s range:    [{args.s_min}, {args.s_max}] m")
    print(f"CSV:        {csv_path}")
    print("=" * 80)

    vecnorm, dummy_env = load_vecnormalize_stats(vecnorm_path, args)
    model = SAC.load(model_path)

    results = []
    for seed in range(args.seed_start, args.seed_end + 1):
        result = run_one_rollout(
            model=model,
            vecnorm=vecnorm,
            seed=seed,
            args=args,
        )
        results.append(result)

        print(
            f"seed={seed:<4} | init_s={result['initial_s']:.2f} m | "
            f"steps={result['steps']:<5} | success={result['success']} | "
            f"failure={result['failure']} | reason={result['termination_reason']} | "
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
    print("Near-final evaluation summary")
    print("=" * 80)
    print(f"Number of rollouts: {summary['n']}")
    print(f"Success rate:       {summary['success_rate']:.4f}")
    print(f"Failure rate:       {summary['failure_rate']:.4f}")
    print(f"Time-limit rate:    {summary['time_limit_rate']:.4f}")
    print(f"Mean final port:    {summary['mean_final_port_distance']:.4f} m")
    print(f"Mean final rho:     {summary['mean_final_rho']:.4f} m")
    print(f"Saved CSV:          {csv_path}")
    print(f"Saved summary:      {summary_path}")


if __name__ == "__main__":
    main()
