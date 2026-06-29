"""
Train SAC for the final-approach near-field debugging task.

This is intentionally not curriculum learning. Every episode starts from a
near-final port-relative state around 35-70 m from the docking port. The goal is
to test whether the existing Gym reward can learn the last segment before
mixing this task back into the full 250-350 m final-approach distribution.
"""

import argparse
import csv
import os
import sys
from collections import deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
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
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "learning_curves")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def rotation_body_to_orbital_2d(psi):
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([[-c, s], [-s, -c]], dtype=np.float64)


class NearFinalApproachEnv(Docking3DoFGymEnv):
    """
    Final-approach environment with a fixed near-field reset distribution.

    Reward is inherited directly from Docking3DoFGymEnv. Only reset sampling is
    changed.
    """

    def __init__(
        self,
        s_range,
        rho_abs_max,
        vp_x_range,
        vp_y_abs_max,
        psi_deg_abs_max,
        omega_deg_abs_max,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.s_range = s_range
        self.rho_abs_max = rho_abs_max
        self.vp_x_range = vp_x_range
        self.vp_y_abs_max = vp_y_abs_max
        self.psi_deg_abs_max = psi_deg_abs_max
        self.omega_deg_abs_max = omega_deg_abs_max

    def _sample_near_final_state(self):
        rng = self.core_env.rng

        s = rng.uniform(*self.s_range)
        cone_radius = self.core_env.cone_slope * s
        rho_limit = min(self.rho_abs_max, 0.8 * cone_radius)
        rho = rng.uniform(-rho_limit, rho_limit)

        psi = np.deg2rad(rng.uniform(-self.psi_deg_abs_max, self.psi_deg_abs_max))
        omega = np.deg2rad(
            rng.uniform(-self.omega_deg_abs_max, self.omega_deg_abs_max)
        )

        r_c_orbital = rotation_body_to_orbital_2d(psi) @ self.core_env.r_c_body
        port_position = np.array([s, rho], dtype=np.float64)
        com_position = port_position - r_c_orbital + self.core_env.r_t_orbital

        port_velocity = np.array(
            [
                rng.uniform(*self.vp_x_range),
                rng.uniform(-self.vp_y_abs_max, self.vp_y_abs_max),
            ],
            dtype=np.float64,
        )
        omega_cross_r_c = np.array(
            [-omega * r_c_orbital[1], omega * r_c_orbital[0]],
            dtype=np.float64,
        )
        com_velocity = port_velocity - omega_cross_r_c

        return np.array(
            [
                com_position[0],
                com_position[1],
                com_velocity[0],
                com_velocity[1],
                psi,
                omega,
            ],
            dtype=np.float64,
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.core_env.rng = np.random.default_rng(seed)

        options = dict(options or {})
        options["initial_state"] = self._sample_near_final_state()
        options["mode"] = "training"

        obs, info = super().reset(seed=seed, options=options)
        info["initial_set"] = "near_final_debug"
        return obs, info


def make_training_env(seed, args):
    env = NearFinalApproachEnv(
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
    return Monitor(env)


class LearningCurveCallback(BaseCallback):
    def __init__(self, csv_path, plot_path, rolling_window=20):
        super().__init__()

        self.csv_path = csv_path
        self.plot_path = plot_path
        self.rolling_window = rolling_window

        self.timesteps = []
        self.episode_returns = []
        self.rolling_returns = []
        self.termination_reasons = []
        self._recent_returns = deque(maxlen=rolling_window)

    def _on_step(self):
        for info in self.locals.get("infos", []):
            episode_info = info.get("episode")

            if episode_info is None:
                continue

            episode_return = float(episode_info["r"])
            self._recent_returns.append(episode_return)

            self.timesteps.append(self.num_timesteps)
            self.episode_returns.append(episode_return)
            self.rolling_returns.append(float(np.mean(self._recent_returns)))
            self.termination_reasons.append(
                info.get("termination_reason", episode_info.get("termination_reason", ""))
            )

        return True

    def _on_training_end(self):
        self._save_csv()
        self._save_plot()

    def _save_csv(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "timestep",
                    "episode_return",
                    "rolling_mean_return",
                    "termination_reason",
                ]
            )

            for episode, timestep, episode_return, rolling_return, reason in zip(
                np.arange(1, len(self.episode_returns) + 1),
                self.timesteps,
                self.episode_returns,
                self.rolling_returns,
                self.termination_reasons,
            ):
                writer.writerow(
                    [
                        int(episode),
                        int(timestep),
                        f"{episode_return:.10f}",
                        f"{rolling_return:.10f}",
                        reason,
                    ]
                )

    def _save_plot(self):
        if not self.episode_returns:
            return

        plt.figure(figsize=(10, 6))
        plt.plot(
            self.timesteps,
            self.episode_returns,
            color="tab:blue",
            alpha=0.35,
            linewidth=1.0,
            label="Episode return",
        )
        plt.plot(
            self.timesteps,
            self.rolling_returns,
            color="tab:orange",
            linewidth=2.0,
            label=f"Rolling mean ({self.rolling_window} episodes)",
        )
        plt.xlabel("Training timesteps")
        plt.ylabel("Episode return")
        plt.title("SAC Near-Final Debug Learning Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=200)
        plt.close()


class NearFinalCheckpointEvalCallback(BaseCallback):
    """
    Periodically saves checkpoints and keeps the best deterministic eval model.

    The saved VecNormalize statistics must match the model checkpoint, so this
    callback always saves both files together.
    """

    def __init__(
        self,
        model_name,
        checkpoint_dir,
        best_model_path,
        best_vecnorm_path,
        eval_csv_path,
        eval_args,
        checkpoint_freq=10_000,
        eval_freq=10_000,
        eval_seed_start=0,
        eval_seed_end=9,
        max_eval_steps=5_000,
    ):
        super().__init__()
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.best_model_path = best_model_path
        self.best_vecnorm_path = best_vecnorm_path
        self.eval_csv_path = eval_csv_path
        self.eval_args = eval_args
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.eval_seed_start = eval_seed_start
        self.eval_seed_end = eval_seed_end
        self.max_eval_steps = max_eval_steps

        self.best_score = None
        self.last_checkpoint_step = 0
        self.last_eval_step = 0
        self.eval_rows = []

    def _on_training_start(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _on_step(self):
        if (
            self.checkpoint_freq > 0
            and self.num_timesteps - self.last_checkpoint_step >= self.checkpoint_freq
        ):
            self._save_checkpoint()
            self.last_checkpoint_step = self.num_timesteps

        if self.eval_freq > 0 and self.num_timesteps - self.last_eval_step >= self.eval_freq:
            summary = self._evaluate_policy()
            self._record_eval(summary)
            self._maybe_save_best(summary)
            self.last_eval_step = self.num_timesteps

        return True

    def _on_training_end(self):
        self._save_checkpoint()
        if self.eval_freq > 0:
            summary = self._evaluate_policy()
            self._record_eval(summary)
            self._maybe_save_best(summary)

    def _save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        step_label = f"{self.num_timesteps:07d}"
        checkpoint_base = os.path.join(
            self.checkpoint_dir,
            f"{self.model_name}_ckpt_{step_label}_steps",
        )
        self.model.save(checkpoint_base)

        vecnorm = self.model.get_vec_normalize_env()
        if vecnorm is not None:
            vecnorm_path = checkpoint_base + "_vecnormalize.pkl"
            os.makedirs(os.path.dirname(vecnorm_path), exist_ok=True)
            vecnorm.save(vecnorm_path)

        print(f"Saved checkpoint at {self.num_timesteps} steps: {checkpoint_base}.zip")

    def _evaluate_policy(self):
        vecnorm = self.model.get_vec_normalize_env()
        if vecnorm is None:
            raise RuntimeError("VecNormalize environment is required for evaluation.")

        old_training = vecnorm.training
        old_norm_reward = vecnorm.norm_reward
        vecnorm.training = False
        vecnorm.norm_reward = False

        results = []
        try:
            for seed in range(self.eval_seed_start, self.eval_seed_end + 1):
                results.append(self._run_one_eval_rollout(seed, vecnorm))
        finally:
            vecnorm.training = old_training
            vecnorm.norm_reward = old_norm_reward

        success_rate = float(np.mean([r["success"] for r in results]))
        failure_rate = float(np.mean([r["failure"] for r in results]))
        mean_final_port = float(
            np.nanmean([r["final_port_distance"] for r in results])
        )
        mean_final_rho = float(np.nanmean([r["final_rho"] for r in results]))
        mean_steps = float(np.mean([r["steps"] for r in results]))

        return {
            "timestep": self.num_timesteps,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "mean_final_port_distance": mean_final_port,
            "mean_final_rho": mean_final_rho,
            "mean_steps": mean_steps,
        }

    def _run_one_eval_rollout(self, seed, vecnorm):
        env = NearFinalApproachEnv(
            s_range=(self.eval_args.s_min, self.eval_args.s_max),
            rho_abs_max=self.eval_args.rho_abs_max,
            vp_x_range=(self.eval_args.vp_x_min, self.eval_args.vp_x_max),
            vp_y_abs_max=self.eval_args.vp_y_abs_max,
            psi_deg_abs_max=self.eval_args.psi_deg_abs_max,
            omega_deg_abs_max=self.eval_args.omega_deg_abs_max,
            phase="final",
            mode="training",
            dt=1.0,
            max_time=None,
            integrator="rk4",
            normalize_action=True,
            seed=seed,
        )

        obs, info = env.reset(seed=seed)
        final_info = info
        final_port_distance = info.get("port_distance", np.nan)
        final_rho = info.get("rho", np.nan)

        for step in range(self.max_eval_steps):
            norm_obs = vecnorm.normalize_obs(obs.reshape(1, -1))
            action, _ = self.model.predict(norm_obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action[0])
            final_info = info
            final_port_distance = info.get("port_distance", np.nan)
            final_rho = info.get("rho", np.nan)

            if terminated or truncated:
                break

        env.close()

        return {
            "seed": seed,
            "steps": step + 1,
            "success": bool(final_info.get("success", False)),
            "failure": bool(final_info.get("failure", False)),
            "final_port_distance": float(final_port_distance),
            "final_rho": float(final_rho),
        }

    def _record_eval(self, summary):
        self.eval_rows.append(summary)
        os.makedirs(os.path.dirname(self.eval_csv_path), exist_ok=True)

        with open(self.eval_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestep",
                    "success_rate",
                    "failure_rate",
                    "mean_final_port_distance",
                    "mean_final_rho",
                    "mean_steps",
                ],
            )
            writer.writeheader()
            writer.writerows(self.eval_rows)

        print(
            "Eval "
            f"step={summary['timestep']} | "
            f"success={summary['success_rate']:.3f} | "
            f"failure={summary['failure_rate']:.3f} | "
            f"mean_final_port={summary['mean_final_port_distance']:.4f} m | "
            f"mean_final_rho={summary['mean_final_rho']:.4f} m"
        )

    def _maybe_save_best(self, summary):
        score = (
            summary["success_rate"],
            -summary["mean_final_port_distance"],
            -summary["failure_rate"],
        )

        if self.best_score is not None and score <= self.best_score:
            return

        self.best_score = score
        self.model.save(self.best_model_path)

        vecnorm = self.model.get_vec_normalize_env()
        if vecnorm is not None:
            vecnorm.save(self.best_vecnorm_path)

        print(
            "Saved new best near-final model: "
            f"{self.best_model_path}.zip at {summary['timestep']} steps"
        )


def model_name_for_args(args):
    steps_label = f"{args.total_timesteps // 1000}ksteps"
    return f"sac_final_approach_nearfinal_gymreward3_{steps_label}_seed{args.seed}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SAC only on near-final 35-70 m reset states."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--s-min", type=float, default=35.0)
    parser.add_argument("--s-max", type=float, default=70.0)
    parser.add_argument("--rho-abs-max", type=float, default=1.5)
    parser.add_argument("--vp-x-min", type=float, default=-0.035)
    parser.add_argument("--vp-x-max", type=float, default=-0.003)
    parser.add_argument("--vp-y-abs-max", type=float, default=0.006)
    parser.add_argument("--psi-deg-abs-max", type=float, default=1.0)
    parser.add_argument("--omega-deg-abs-max", type=float, default=0.04)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-seed-start", type=int, default=0)
    parser.add_argument("--eval-seed-end", type=int, default=9)
    parser.add_argument("--max-eval-steps", type=int, default=5_000)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.s_min <= 0.0 or args.s_max <= args.s_min:
        raise ValueError("--s-max must be greater than --s-min, and both must be positive.")

    model_name = model_name_for_args(args)
    model_path = os.path.join(MODEL_DIR, model_name)
    model_zip_path = model_path + ".zip"
    vecnorm_path = os.path.join(MODEL_DIR, f"{model_name}_vecnormalize.pkl")
    learning_curve_dir = os.path.join(RESULTS_DIR, model_name)
    learning_curve_csv = os.path.join(learning_curve_dir, "learning_curve.csv")
    learning_curve_plot = os.path.join(learning_curve_dir, "learning_curve.png")
    checkpoint_dir = os.path.join(MODEL_DIR, "checkpoints", model_name)
    eval_progress_csv = os.path.join(learning_curve_dir, "checkpoint_eval.csv")
    best_model_path = os.path.join(MODEL_DIR, f"{model_name}_best")
    best_vecnorm_path = os.path.join(MODEL_DIR, f"{model_name}_best_vecnormalize.pkl")

    if not args.overwrite and os.path.exists(model_zip_path) and os.path.exists(vecnorm_path):
        print("=" * 80)
        print(f"Skipping existing near-final model: {model_name}")
        print(f"Model:   {model_zip_path}")
        print(f"VecNorm: {vecnorm_path}")
        print("Use --overwrite to retrain.")
        return

    os.makedirs(learning_curve_dir, exist_ok=True)

    print("=" * 80)
    print("SAC final-approach near-final debug training")
    print("=" * 80)
    print(f"Training seed:    {args.seed}")
    print(f"Model path:       {model_path}")
    print(f"VecNorm path:     {vecnorm_path}")
    print(f"Learning curve:   {learning_curve_plot}")
    print(f"Checkpoint dir:   {checkpoint_dir}")
    print(f"Best model path:  {best_model_path}.zip")
    print(f"Eval progress:    {eval_progress_csv}")
    print(f"Total steps:      {args.total_timesteps}")
    print(f"s range:          [{args.s_min}, {args.s_max}] m")
    print(f"rho abs max:      {args.rho_abs_max} m")
    print(f"vp_x range:       [{args.vp_x_min}, {args.vp_x_max}] m/s")
    print(f"vp_y abs max:     {args.vp_y_abs_max} m/s")
    print(f"psi abs max:      {args.psi_deg_abs_max} deg")
    print(f"omega abs max:    {args.omega_deg_abs_max} deg/s")
    print("Reward source:    Docking3DoFGymEnv._compute_sac_reward")
    print("=" * 80)

    train_env = DummyVecEnv([lambda: make_training_env(seed=args.seed, args=args)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.98,
    )

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=0.0003395686575052769,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        train_freq=8,
        gradient_steps=8,
        ent_coef="auto_0.1",
        target_update_interval=1,
        verbose=1,
        seed=args.seed,
        tensorboard_log=LOG_DIR,
        device=args.device,
    )

    callback = LearningCurveCallback(
        csv_path=learning_curve_csv,
        plot_path=learning_curve_plot,
        rolling_window=20,
    )
    checkpoint_eval_callback = NearFinalCheckpointEvalCallback(
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        best_model_path=best_model_path,
        best_vecnorm_path=best_vecnorm_path,
        eval_csv_path=eval_progress_csv,
        eval_args=args,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        eval_seed_start=args.eval_seed_start,
        eval_seed_end=args.eval_seed_end,
        max_eval_steps=args.max_eval_steps,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        tb_log_name=model_name,
        progress_bar=args.progress_bar,
        callback=CallbackList([callback, checkpoint_eval_callback]),
    )

    model.save(model_path)
    train_env.save(vecnorm_path)
    train_env.close()

    print("=" * 80)
    print("Near-final training completed.")
    print(f"Saved model:   {model_zip_path}")
    print(f"Saved VecNorm: {vecnorm_path}")
    print(f"Saved best:    {best_model_path}.zip")
    print(f"Best VecNorm:  {best_vecnorm_path}")
    print(f"Saved curve:   {learning_curve_csv}")
    print(f"Saved evals:   {eval_progress_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
