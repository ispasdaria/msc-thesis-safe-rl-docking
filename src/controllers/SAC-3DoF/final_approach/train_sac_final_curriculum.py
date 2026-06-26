"""
Train SAC for final approach with a simple near-to-far curriculum.

The same agent is trained sequentially on:
    1. close port-relative initial states near docking,
    2. medium-range port-relative initial states,
    3. the standard final-approach training initialization range.

This script intentionally uses a local reward variant for the curriculum
experiment. The reward penalizes bad terminal/instantaneous behavior and
rewards generic reduction of port distance, without prescribing an axial
velocity profile or a specific path through the docking cone.
"""

import argparse
import csv
import os
import sys
from collections import deque
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
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


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    timesteps: int
    s_range: tuple[float, float] | None = None
    rho_max: float | None = None
    vp_x_range: tuple[float, float] | None = None
    vp_y_abs_max: float | None = None
    psi_deg_abs_max: float | None = None
    omega_deg_abs_max: float | None = None
    use_standard_training_reset: bool = False


DEFAULT_STAGES = (
    CurriculumStage(
        name="close",
        timesteps=50_000,
        s_range=(15.0, 60.0),
        rho_max=2.0,
        vp_x_range=(-0.035, -0.003),
        vp_y_abs_max=0.008,
        psi_deg_abs_max=1.0,
        omega_deg_abs_max=0.04,
    ),
    CurriculumStage(
        name="middle",
        timesteps=50_000,
        s_range=(60.0, 160.0),
        rho_max=4.0,
        vp_x_range=(-0.05, -0.005),
        vp_y_abs_max=0.015,
        psi_deg_abs_max=1.5,
        omega_deg_abs_max=0.07,
    ),
    CurriculumStage(
        name="standard",
        timesteps=100_000,
        use_standard_training_reset=True,
    ),
)


def rotation_body_to_orbital_2d(psi):
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([[-c, s], [-s, -c]], dtype=np.float64)


class CurriculumFinalApproachEnv(Docking3DoFGymEnv):
    def __init__(self, stage, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = stage

    def _sample_curriculum_state(self):
        rng = self.core_env.rng

        s = rng.uniform(*self.stage.s_range)
        cone_radius = self.core_env.cone_slope * s
        rho_limit = min(self.stage.rho_max, 0.8 * cone_radius)
        rho = rng.uniform(-rho_limit, rho_limit)

        psi = np.deg2rad(rng.uniform(-self.stage.psi_deg_abs_max, self.stage.psi_deg_abs_max))
        omega = np.deg2rad(
            rng.uniform(-self.stage.omega_deg_abs_max, self.stage.omega_deg_abs_max)
        )

        r_c_orbital = rotation_body_to_orbital_2d(psi) @ self.core_env.r_c_body
        port_position = np.array([s, rho], dtype=np.float64)
        com_position = port_position - r_c_orbital + self.core_env.r_t_orbital

        port_velocity = np.array(
            [
                rng.uniform(*self.stage.vp_x_range),
                rng.uniform(-self.stage.vp_y_abs_max, self.stage.vp_y_abs_max),
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
        if self.stage.use_standard_training_reset:
            obs, info = super().reset(seed=seed, options=options)
            info["curriculum_stage"] = self.stage.name
            return obs, info

        if seed is not None:
            self.core_env.rng = np.random.default_rng(seed)

        options = dict(options or {})
        options["initial_state"] = self._sample_curriculum_state()
        options["mode"] = "training"
        obs, info = super().reset(seed=seed, options=options)
        info["curriculum_stage"] = self.stage.name
        return obs, info

    def _compute_sac_reward(
        self,
        state,
        normalized_action,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        q = self.core_env.get_derived_quantities(state)

        port_distance = q["port_distance"]
        port_speed = q["port_speed"]
        psi_abs = abs(q["psi_e"])
        omega_abs = abs(q["omega"])

        action_scale = np.array(
            [self.core_env.ax_max, self.core_env.ay_max, self.core_env.alphaz_max],
            dtype=np.float64,
        )
        action_norm = np.linalg.norm(physical_action / action_scale)

        progress_port = 0.0
        if hasattr(self, "prev_port_distance"):
            progress_port = self.prev_port_distance - port_distance

        vp_safe = getattr(self.core_env, "vp_safe_final", 0.15)
        speed_ratio = port_speed / vp_safe
        psi_ratio = psi_abs / self.core_env.psi_safe
        omega_ratio = omega_abs / self.core_env.omega_safe

        reward_position = -0.02 * port_distance
        reward_velocity = -2.0 * port_speed
        reward_attitude = -3.0 * psi_abs
        reward_rate = -0.75 * omega_abs
        reward_action = -0.015 * action_norm**2
        reward_alpha = -1.0 * normalized_action[2] ** 2

        reward_progress = 60.0 * progress_port
        reward_moving_away = -20.0 * max(0.0, -progress_port)
        reward_speed_margin = -15.0 * max(0.0, speed_ratio - 0.8) ** 2
        reward_psi_margin = -15.0 * max(0.0, psi_ratio - 0.8) ** 2
        reward_omega_margin = -10.0 * max(0.0, omega_ratio - 0.8) ** 2

        close_bonus = 5.0 * np.exp(-port_distance / 2.0)
        slow_bonus = 2.0 * np.exp(-port_speed / 0.05)
        aligned_bonus = 2.0 * np.exp(-(psi_ratio**2 + omega_ratio**2))
        reward_bonus = close_bonus + slow_bonus + aligned_bonus

        reward = (
            reward_position
            + reward_velocity
            + reward_attitude
            + reward_rate
            + reward_action
            + reward_alpha
            + reward_progress
            + reward_moving_away
            + reward_speed_margin
            + reward_psi_margin
            + reward_omega_margin
            + reward_bonus
        )

        self.prev_port_distance = port_distance
        self.prev_s = q["s"]
        self.prev_rho = q["rho"]

        reward_components = {
            "phase": self.phase,
            "curriculum_stage": self.stage.name,
            "port_distance": float(port_distance),
            "port_speed": float(port_speed),
            "psi_abs": float(psi_abs),
            "omega_abs": float(omega_abs),
            "action_norm": float(action_norm),
            "s": float(q["s"]),
            "rho": float(q["rho"]),
            "s_dot": float(q["s_dot"]),
            "pos_error": float(port_distance),
            "vel_error": float(port_speed),
            "attitude_error": float(psi_abs),
            "rate_error": float(omega_abs),
            "progress_port": float(progress_port),
            "speed_ratio": float(speed_ratio),
            "psi_ratio": float(psi_ratio),
            "omega_ratio": float(omega_ratio),
            "reward_position": float(reward_position),
            "reward_velocity": float(reward_velocity),
            "reward_attitude": float(reward_attitude),
            "reward_rate": float(reward_rate),
            "reward_action": float(reward_action),
            "reward_alpha": float(reward_alpha),
            "reward_progress": float(reward_progress),
            "reward_moving_away": float(reward_moving_away),
            "reward_speed_margin": float(reward_speed_margin),
            "reward_psi_margin": float(reward_psi_margin),
            "reward_omega_margin": float(reward_omega_margin),
            "reward_close_bonus": float(close_bonus),
            "reward_slow_bonus": float(slow_bonus),
            "reward_aligned_bonus": float(aligned_bonus),
            "reward_warning": 0.0,
            "reward_failure": 0.0,
            "reward_success": 0.0,
            "reward_timeout": 0.0,
            "reward_total": 0.0,
        }

        if info.get("warning", False):
            reward_warning = -10.0
            reward += reward_warning
            reward_components["reward_warning"] = float(reward_warning)

        if info.get("failure", False):
            reward_failure = -1000.0
            reward += reward_failure
            reward_components["reward_failure"] = float(reward_failure)

        if info.get("success", False):
            reward_success = 1000.0
            reward += reward_success
            reward_components["reward_success"] = float(reward_success)

        if truncated:
            reward_timeout = -200.0
            reward += reward_timeout
            reward_components["reward_timeout"] = float(reward_timeout)

        reward_components["reward_total"] = float(reward)
        return reward, reward_components


def make_training_env(seed, stage):
    env = CurriculumFinalApproachEnv(
        stage=stage,
        phase="final",
        mode="training",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=seed,
    )
    return Monitor(env)


class CurriculumLearningCurveCallback(BaseCallback):
    def __init__(self, csv_path, plot_path, rolling_window=20):
        super().__init__()

        self.csv_path = csv_path
        self.plot_path = plot_path
        self.rolling_window = rolling_window

        self.timesteps = []
        self.stage_timesteps = []
        self.stages = []
        self.episode_returns = []
        self.rolling_returns = []
        self.termination_reasons = []
        self._recent_returns = deque(maxlen=rolling_window)
        self.current_stage = ""
        self.stage_start_timestep = 0

    def set_stage(self, stage_name, stage_start_timestep):
        self.current_stage = stage_name
        self.stage_start_timestep = stage_start_timestep

    def _on_step(self):
        for info in self.locals.get("infos", []):
            episode_info = info.get("episode")

            if episode_info is None:
                continue

            episode_return = float(episode_info["r"])
            self._recent_returns.append(episode_return)

            self.timesteps.append(self.num_timesteps)
            self.stage_timesteps.append(self.num_timesteps - self.stage_start_timestep)
            self.stages.append(self.current_stage)
            self.episode_returns.append(episode_return)
            self.rolling_returns.append(float(np.mean(self._recent_returns)))
            self.termination_reasons.append(
                info.get("termination_reason", episode_info.get("termination_reason", ""))
            )

        return True

    def save(self):
        self._save_csv()
        self._save_plot()

    def _save_csv(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "global_timestep",
                    "stage_timestep",
                    "stage",
                    "episode_return",
                    "rolling_mean_return",
                    "termination_reason",
                ]
            )

            for episode, timestep, stage_timestep, stage, episode_return, rolling_return, reason in zip(
                np.arange(1, len(self.episode_returns) + 1),
                self.timesteps,
                self.stage_timesteps,
                self.stages,
                self.episode_returns,
                self.rolling_returns,
                self.termination_reasons,
            ):
                writer.writerow(
                    [
                        int(episode),
                        int(timestep),
                        int(stage_timestep),
                        stage,
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
        plt.title("SAC Final-Approach Curriculum Learning Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=200)
        plt.close()


def model_name_for_seed(seed, stage_timesteps):
    total_timesteps = sum(stage_timesteps)
    steps_label = f"{total_timesteps // 1000}ksteps"
    return f"sac_final_approach_curriculum_basisreward_{steps_label}_seed{seed}"


def create_vec_env(seed, stage):
    env = DummyVecEnv([lambda: make_training_env(seed=seed, stage=stage)])
    return VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.98,
    )


def train_one_seed(seed, stages, device, progress_bar, overwrite):
    model_name = model_name_for_seed(seed, [stage.timesteps for stage in stages])

    model_path = os.path.join(MODEL_DIR, model_name)
    model_zip_path = model_path + ".zip"
    vecnorm_path = os.path.join(MODEL_DIR, f"{model_name}_vecnormalize.pkl")
    learning_curve_dir = os.path.join(RESULTS_DIR, model_name)
    learning_curve_csv = os.path.join(learning_curve_dir, "learning_curve.csv")
    learning_curve_plot = os.path.join(learning_curve_dir, "learning_curve.png")

    if not overwrite and os.path.exists(model_zip_path) and os.path.exists(vecnorm_path):
        print("=" * 80)
        print(f"Skipping seed {seed}: existing model and VecNormalize found.")
        print(f"Model:   {model_zip_path}")
        print(f"VecNorm: {vecnorm_path}")
        print("Use --overwrite to retrain this seed.")
        return

    os.makedirs(learning_curve_dir, exist_ok=True)

    print("=" * 80)
    print("SAC final-approach curriculum training")
    print("=" * 80)
    print(f"Training seed: {seed}")
    print(f"Model path:    {model_path}")
    print(f"VecNorm path:  {vecnorm_path}")
    print(f"Stages:        {', '.join(f'{s.name}:{s.timesteps}' for s in stages)}")
    print("=" * 80)

    train_env = create_vec_env(seed=seed, stage=stages[0])

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
        seed=seed,
        tensorboard_log=LOG_DIR,
        device=device,
    )

    callback = CurriculumLearningCurveCallback(
        csv_path=learning_curve_csv,
        plot_path=learning_curve_plot,
        rolling_window=20,
    )

    for stage_idx, stage in enumerate(stages):
        if stage_idx > 0:
            previous_env = train_env
            train_env = create_vec_env(seed=seed + stage_idx * 10_000, stage=stage)
            train_env.obs_rms = previous_env.obs_rms
            train_env.ret_rms = previous_env.ret_rms
            model.set_env(train_env)
            previous_env.close()

        callback.set_stage(
            stage_name=stage.name,
            stage_start_timestep=model.num_timesteps,
        )

        print("-" * 80)
        print(f"Stage {stage_idx + 1}/{len(stages)}: {stage.name}")
        print(f"Stage steps: {stage.timesteps}")
        print("-" * 80)

        model.learn(
            total_timesteps=stage.timesteps,
            tb_log_name=f"{model_name}_{stage.name}",
            progress_bar=progress_bar,
            callback=callback,
            reset_num_timesteps=(stage_idx == 0),
        )

        callback.save()

    model.save(model_path)
    train_env.save(vecnorm_path)
    train_env.close()

    print("=" * 80)
    print(f"Curriculum training completed for seed {seed}.")
    print(f"Saved model:   {model_zip_path}")
    print(f"Saved VecNorm: {vecnorm_path}")
    print(f"Saved curve:   {learning_curve_csv}")
    print(f"Total steps:   {model.num_timesteps}")
    print("=" * 80)


def build_stages(args):
    stage_timesteps = [
        args.close_timesteps,
        args.middle_timesteps,
        args.standard_timesteps,
    ]

    return tuple(
        CurriculumStage(
            name=stage.name,
            timesteps=timesteps,
            s_range=stage.s_range,
            rho_max=stage.rho_max,
            vp_x_range=stage.vp_x_range,
            vp_y_abs_max=stage.vp_y_abs_max,
            psi_deg_abs_max=stage.psi_deg_abs_max,
            omega_deg_abs_max=stage.omega_deg_abs_max,
            use_standard_training_reset=stage.use_standard_training_reset,
        )
        for stage, timesteps in zip(DEFAULT_STAGES, stage_timesteps)
        if timesteps > 0
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SAC final approach with a near-to-far curriculum."
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=0)
    parser.add_argument("--close-timesteps", type=int, default=50_000)
    parser.add_argument("--middle-timesteps", type=int, default=50_000)
    parser.add_argument("--standard-timesteps", type=int, default=100_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain seeds even when model files already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be greater than or equal to --seed-start.")

    stages = build_stages(args)
    if not stages:
        raise ValueError("At least one curriculum stage must have positive timesteps.")

    for seed in range(args.seed_start, args.seed_end + 1):
        train_one_seed(
            seed=seed,
            stages=stages,
            device=args.device,
            progress_bar=args.progress_bar,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
