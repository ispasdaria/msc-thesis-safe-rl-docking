from numbers import Integral

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.docking3dof.core import Docking3DoFEnv
from envs.docking3dof.rewards.closing import compute_sac_reward


class Docking3DoFGymEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for the existing Docking3DoFEnv.

    SAC sees phase-specific observations.

    Final approach:
        observation = [rpx, rpy, vpx, vpy, psi_e, omegaz]

        rpx, rpy:
            chaser docking-port center relative to the target docking-port
            center, expressed in the target-centered orbital frame [m]
        vpx, vpy:
            chaser docking-port velocity relative to the target docking-port
            velocity, expressed in the target-centered orbital frame [m/s]
        psi_e:
            wrapped relative attitude error [rad]
        omegaz:
            relative angular velocity [rad/s]
        action      = normalized action in [-1, 1]^3

    Internally:
        action is scaled to physical accelerations:
            [ax_Bc, ay_Bc, alphaz_Bc]

    Closing:
        observation = [
            rpx,
            rpy,
            vpx,
            vpy,
            psi_e,
            omegaz,
            rx - rx_P3,
            ry - ry_P3,
            vx - vx_P3,
            vy - vy_P3,
            wrap(psi - psi_P3),
            omegaz - omegaz_P3,
        ]

    The simulator state remains [rx, ry, vx, vy, psi, omegaz]. The closing
    agent observes both the docking-port state used by the safety constraints
    and its error relative to the episode's sampled P3 target. The
    final-approach agent observes only docking-port errors because docking is
    accepted at port level.
    The phase changes:
        - reset distribution
        - terminal condition
        - safety logic
        - reward definition
        - default maximum episode time
    """

    metadata = {"render_modes": []}
    closing_observation_labels = (
        "rpx",
        "rpy",
        "vpx",
        "vpy",
        "psi_e",
        "omegaz",
        "p3_rx_error",
        "p3_ry_error",
        "p3_vx_error",
        "p3_vy_error",
        "p3_psi_error",
        "p3_omega_error",
    )
    final_observation_labels = ("rpx", "rpy", "vpx", "vpy", "psi_e", "omegaz")
    closing_reward_versions = (
        "v15",
        "v15_dt",
        "v15_dt_safe",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
    )

    # Reward v15_dt is the time-consistent form of v15. Persistent costs are
    # interpreted as cost rates and integrated over the simulator step. The
    # progress-potential and terminal-event terms retain their original scale.
    closing_reward_reference_dt = 1.0

    # Reward v15_dt_safe keeps v15_dt's physical-time interpretation while
    # correcting the long-horizon incentive to terminate early. It also uses
    # bounded tolerance-normalized alignment costs and introduces velocity
    # guard penalties before the existing warning boundary.
    closing_safe_failure_reward = -50_000.0
    closing_safe_success_reward = 50_000.0
    closing_safe_timeout_reward = -10_000.0
    closing_safe_attitude_state_scale = 1.0
    closing_safe_rate_state_scale = 0.5
    closing_safe_attitude_progress_scale = 100.0
    closing_safe_rate_progress_scale = 25.0
    closing_safe_speed_guard_scale = 10.0
    closing_safe_speed_guard_fraction = 0.7

    # Reward v16 adds bounded P3-terminal error shaping to v15. The
    # ratios are clipped so the very small velocity/alignment tolerances cannot
    # dominate learning far outside the terminal region.
    closing_v16_ratio_clip = 10.0
    closing_v16_aggregate_weight = 0.05
    closing_v16_worst_weight = 0.10

    # Reward v17 replaces v16's persistent clipped cost with progress through a
    # smooth, tolerance-normalized terminal potential. log1p preserves a useful
    # gradient above ten tolerances without allowing large errors to dominate.
    closing_v17_smooth_max_beta = 4.0
    closing_v17_aggregate_weight = 0.5
    closing_v17_worst_weight = 0.5
    closing_v17_progress_scale = 20.0

    # Reward v18 is the controlled scale-correction experiment: it keeps the
    # complete v17 potential unchanged and increases only its progress scale so
    # terminal-dimensional improvement is material beside distance progress.
    closing_v18_progress_scale = 500.0

    # Reward v19 is a deliberately simple terminal-state experiment. It uses
    # the same smooth tolerance-normalized cost as v17, but applies that cost
    # directly at every step instead of rewarding its undiscounted change.
    # This aligns the dense reward with the exact six-component P3 success set.
    closing_v19_state_cost_scale = 10.0

    # Reward v20 keeps v19's terminal-state objective and restores only v15's
    # two velocity warning-band penalties. They are zero throughout the safe
    # velocity set and become dense before the hard velocity boundary.

    def __init__(
        self,
        phase="final",
        mode="training",
        dt=1.0,
        max_time=None,
        integrator="rk4",
        normalize_action=True,
        seed=None,
        closing_reward_version="v15",
        action_hold_steps=1,
        adaptive_action_hold_distance_m=None,
    ):
        super().__init__()

        if phase not in ["closing", "final"]:
            raise ValueError("phase must be either 'closing' or 'final'.")

        if mode not in ["nominal", "testing", "training"]:
            raise ValueError("mode must be one of: 'nominal', 'testing', or 'training'.")

        if closing_reward_version not in self.closing_reward_versions:
            raise ValueError(
                "closing_reward_version must be one of: "
                f"{', '.join(self.closing_reward_versions)}."
            )

        if (
            not isinstance(action_hold_steps, Integral)
            or isinstance(action_hold_steps, (bool, np.bool_))
            or action_hold_steps <= 0
        ):
            raise ValueError("action_hold_steps must be a positive integer.")
        if adaptive_action_hold_distance_m is not None:
            if phase != "closing":
                raise ValueError(
                    "adaptive action hold is supported only for closing."
                )
            if (
                not np.isfinite(adaptive_action_hold_distance_m)
                or adaptive_action_hold_distance_m <= 0.0
            ):
                raise ValueError(
                    "adaptive_action_hold_distance_m must be positive."
                )

        if max_time is None:
            if phase == "closing":
                max_time = 12000.0
            elif phase == "final":
                max_time = 5000.0

        self.phase = phase
        self.mode = mode
        self.normalize_action = normalize_action
        self.max_time = max_time
        self.closing_reward_version = closing_reward_version
        self.action_hold_steps = int(action_hold_steps)
        self.adaptive_action_hold_distance_m = (
            None
            if adaptive_action_hold_distance_m is None
            else float(adaptive_action_hold_distance_m)
        )

        self.core_env = Docking3DoFEnv(
            dt=dt,
            max_time=max_time,
            integrator=integrator,
            phase=phase,
            seed=seed,
        )

        observation_size = 12 if self.phase == "closing" else 6
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

        if self.normalize_action:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=self.core_env.action_low.astype(np.float32),
                high=self.core_env.action_high.astype(np.float32),
                dtype=np.float32,
            )

    def get_physical_state(self):
        """Return an unnormalized copy of ``[rx, ry, vx, vy, psi, omega]``."""
        if self.core_env.state is None:
            raise RuntimeError("Environment must be reset before reading its state.")

        return np.asarray(self.core_env.state, dtype=np.float64).copy()

    def get_physical_action_limits(self):
        """Return positive body-frame limits ``[ax_max, ay_max, alpha_max]``."""
        return np.asarray(self.core_env.action_high, dtype=np.float64).copy()

    def normalized_action_to_physical(self, action):
        """Convert an environment action to physical body-frame acceleration."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (3,):
            raise ValueError(f"Action must have shape (3,), got {action.shape}.")

        if not self.normalize_action:
            return action.copy()

        return action * self.get_physical_action_limits()

    def physical_action_to_normalized(self, action):
        """Convert a physical body-frame command to the environment action space."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (3,):
            raise ValueError(f"Action must have shape (3,), got {action.shape}.")

        if not self.normalize_action:
            return np.clip(
                action,
                self.core_env.action_low,
                self.core_env.action_high,
            )

        normalized = action / self.get_physical_action_limits()
        return np.clip(normalized, -1.0, 1.0)

    def _scale_action(self, action):
        """Backward-compatible private alias for action conversion."""
        return self.normalized_action_to_physical(action)

    def _get_obs(self):
        """
        Returns the current Gymnasium observation.

        In the closing phase, the SAC policy receives both the docking-port
        state used by the safety constraints and the error relative to the
        sampled P3 handoff state. In the final phase, it receives only the
        docking-port relative state used by the docking acceptance conditions.
        """

        if self.phase == "closing":
            if self.core_env.p3_target is None:
                raise RuntimeError(
                    "Closing observation requires a sampled P3 target. "
                    "Call reset() without a provided initial state, or set "
                    "core_env.p3_target before requesting an observation."
                )

            error = self.core_env.state - self.core_env.p3_target
            error[4] = self.core_env.wrap_to_pi(error[4])

            q = self.core_env.get_derived_quantities(self.core_env.state)
            port_state = np.array(
                [
                    q["r_p"][0],
                    q["r_p"][1],
                    q["v_p"][0],
                    q["v_p"][1],
                    q["psi_e"],
                    q["omega"],
                ],
                dtype=np.float64,
            )

            return np.concatenate((port_state, error)).astype(np.float32)

        q = self.core_env.get_derived_quantities(self.core_env.state)

        obs = np.array(
            [
                q["r_p"][0],
                q["r_p"][1],
                q["v_p"][0],
                q["v_p"][1],
                q["psi_e"],
                q["omega"],
            ],
            dtype=np.float32,
        )

        return obs

    def _get_observation_labels(self):
        if self.phase == "final":
            return self.final_observation_labels

        return self.closing_observation_labels

    def _get_reset_info(self):
        """
        Returns diagnostic information after reset.
        """

        q = self.core_env.get_derived_quantities(self.core_env.state)
        monitor = self.core_env.monitor_safety(self.core_env.state)

        return {
            "phase": self.phase,
            "mode": self.core_env.mode,
            "max_time": self.max_time,
            "safe": monitor["safe"],
            "warning": monitor["warning"],
            "unsafe": monitor["unsafe"],
            "port_distance": q["port_distance"],
            "port_speed": q["port_speed"],
            "com_distance": q["com_distance"],
            "com_speed": q["com_speed"],
            "s": q["s"],
            "rho": q["rho"],
            "s_dot": q["s_dot"],
            "psi_e": q["psi_e"],
            "omega": q["omega"],
            "observation_labels": self._get_observation_labels(),
            "integration_step_seconds": float(self.core_env.dt),
            "action_hold_steps": self.action_hold_steps,
            "control_interval_seconds": float(
                self.action_hold_steps * self.core_env.dt
            ),
            "adaptive_action_hold": (
                self.adaptive_action_hold_distance_m is not None
            ),
            "adaptive_action_hold_distance_m": (
                self.adaptive_action_hold_distance_m
            ),
        }

    def reset(self, seed=None, options=None):
        """
        Gymnasium reset API.

        Returns:
            obs, info
        """

        super().reset(seed=seed)

        if seed is not None:
            self.core_env.rng = np.random.default_rng(seed)

        initial_state = None
        p3_target = None
        mode = self.mode

        if options is not None:
            initial_state = options.get("initial_state", None)
            p3_target = options.get("p3_target", None)
            mode = options.get("mode", self.mode)

            if mode not in ["nominal", "testing", "training"]:
                raise ValueError(
                    "mode must be one of: 'nominal', 'testing', or 'training'."
                )

        self.core_env.reset(
            initial_state=initial_state,
            mode=mode,
            p3_target=p3_target,
        )

        q = self.core_env.get_derived_quantities(self.core_env.state)

        self.prev_port_distance = q["port_distance"]
        self.prev_s = q["s"]
        self.prev_rho = q["rho"]
        if self.phase == "closing" and self.core_env.p3_target is not None:
            target_error = self.core_env.state - self.core_env.p3_target
            target_error[4] = self.core_env.wrap_to_pi(
                self.core_env.state[4] - self.core_env.p3_target[4]
            )
            self.prev_closing_pos_error = np.linalg.norm(target_error[0:2])
            self.prev_closing_lateral_error = abs(target_error[1])
            self.prev_closing_attitude_error = abs(target_error[4])
            self.prev_closing_rate_error = abs(target_error[5])
            self.prev_closing_terminal_cost = self._closing_v17_terminal_cost(
                target_error,
                self.core_env.p3_target_tolerance,
            )["cost"]

        obs = self._get_obs()
        info = self._get_reset_info()

        return obs, info

    def _closing_v17_terminal_cost(self, error, tolerance):
        """Return the smooth tolerance-normalized P3 terminal potential."""
        ratios = np.abs(np.asarray(error, dtype=np.float64)) / np.maximum(
            np.asarray(tolerance, dtype=np.float64),
            1e-12,
        )
        log_components = np.log1p(ratios)
        aggregate = float(np.mean(log_components))

        beta = self.closing_v17_smooth_max_beta
        maximum = float(np.max(log_components))
        smooth_worst = float(
            maximum
            + np.log(np.mean(np.exp(beta * (log_components - maximum))))
            / beta
        )
        cost = (
            self.closing_v17_aggregate_weight * aggregate
            + self.closing_v17_worst_weight * smooth_worst
        )
        return {
            "ratios": ratios,
            "log_components": log_components,
            "aggregate": aggregate,
            "smooth_worst": smooth_worst,
            "cost": float(cost),
        }

    @staticmethod
    def _aggregate_reward_components(component_sets, total_reward):
        """Keep final diagnostics and sum additive reward terms over a hold."""
        aggregated = dict(component_sets[-1])
        reward_keys = {
            key
            for components in component_sets
            for key, value in components.items()
            if key.startswith("reward_")
            and isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
        }
        for key in reward_keys:
            aggregated[key] = float(
                sum(float(components.get(key, 0.0)) for components in component_sets)
            )
        aggregated["reward_total"] = float(total_reward)
        return aggregated

    @staticmethod
    def _action_hold_classification_counts(infos):
        """Count safety classifications at the simulator integration cadence."""
        counts = {}
        for info in infos:
            for level in ("safe", "warning", "unsafe"):
                counts[level] = counts.get(level, 0) + int(
                    bool(info.get(level, False))
                )
            for name in ("geometry", "velocity", "alignment"):
                classification = info.get(name, {})
                for level in ("safe", "warn", "unsafe"):
                    key = f"{name}_{level}"
                    counts[key] = counts.get(key, 0) + int(
                        bool(classification.get(level, False))
                    )
        return counts

    def _step_once(self, action):
        """Advance one simulator integration step for the supplied action."""
        physical_action = self._scale_action(action)

        _, done, info = self.core_env.step(physical_action)
        obs = self._get_obs()

        terminated = bool(
            info.get("success", False)
            or info.get("failure", False)
        )

        truncated = bool(info.get("time_limit", False))

        reward, reward_components = self._compute_sac_reward(
            state=self.core_env.state,
            normalized_action=np.asarray(action, dtype=np.float64),
            physical_action=physical_action,
            info=info,
            terminated=terminated,
            truncated=truncated,
        )

        info["normalized_action"] = np.asarray(action, dtype=np.float32)
        info["physical_action"] = physical_action.astype(np.float32)
        info["reward_components"] = reward_components

        return obs, float(reward), terminated, truncated, info

    def _p3_position_error_m(self):
        if self.phase != "closing" or self.core_env.p3_target is None:
            return np.inf
        error = self.core_env.state[0:2] - self.core_env.p3_target[0:2]
        return float(np.linalg.norm(error))

    def _resolve_action_hold(self):
        """Return this decision's hold length and the cadence reason."""
        distance = self._p3_position_error_m()
        if self.adaptive_action_hold_distance_m is None:
            return self.action_hold_steps, "fixed", distance

        monitor = self.core_env.monitor_safety(self.core_env.state)
        if monitor["warning"]:
            return 1, "safety_warning", distance
        if distance <= self.adaptive_action_hold_distance_m:
            return 1, "near_p3", distance
        return self.action_hold_steps, "far_from_p3", distance

    def step(self, action):
        """
        Gymnasium step API.

        Returns:
            obs, reward, terminated, truncated, info
        """

        rewards = []
        infos = []
        requested_steps, cadence_reason, start_distance = (
            self._resolve_action_hold()
        )
        break_reason = None

        for substep_index in range(requested_steps):
            obs, reward, terminated, truncated, info = self._step_once(action)
            rewards.append(float(reward))
            infos.append(info)
            if terminated or truncated:
                break
            if (
                self.adaptive_action_hold_distance_m is not None
                and substep_index + 1 < requested_steps
            ):
                if info.get("warning", False):
                    break_reason = "entered_safety_warning"
                    break
                if (
                    self._p3_position_error_m()
                    <= self.adaptive_action_hold_distance_m
                ):
                    break_reason = "entered_near_p3_region"
                    break

        total_reward = float(sum(rewards))
        info = dict(infos[-1])
        info["reward_components"] = self._aggregate_reward_components(
            [item["reward_components"] for item in infos],
            total_reward,
        )
        info["integration_step_seconds"] = float(self.core_env.dt)
        info["action_hold_steps"] = self.action_hold_steps
        info["action_hold_steps_requested"] = requested_steps
        info["action_hold_steps_executed"] = len(infos)
        info["action_hold_seconds_executed"] = float(
            len(infos) * self.core_env.dt
        )
        info["control_interval_seconds"] = float(
            requested_steps * self.core_env.dt
        )
        info["action_hold_seconds_requested"] = float(
            requested_steps * self.core_env.dt
        )
        info["adaptive_action_hold"] = (
            self.adaptive_action_hold_distance_m is not None
        )
        info["adaptive_action_hold_distance_m"] = (
            self.adaptive_action_hold_distance_m
        )
        info["action_hold_cadence_reason"] = cadence_reason
        info["action_hold_break_reason"] = break_reason
        info["p3_position_error_at_decision_m"] = start_distance
        info["p3_position_error_m"] = self._p3_position_error_m()
        info["action_hold_reward_aggregation"] = "undiscounted_sum"
        info["action_hold_classification_counts"] = (
            self._action_hold_classification_counts(infos)
        )

        violated_counts = {}
        for substep_info in infos:
            for constraint in substep_info.get("violated_constraints", []):
                constraint = str(constraint)
                violated_counts[constraint] = violated_counts.get(constraint, 0) + 1
        info["action_hold_violated_constraint_counts"] = violated_counts

        return obs, total_reward, terminated, truncated, info

    def _compute_sac_reward(
        self,
        state,
        normalized_action,
        physical_action,
        info,
        terminated,
        truncated,
    ):
        """Return the phase reward and its diagnostic component mapping."""
        return compute_sac_reward(
            self,
            state,
            normalized_action,
            physical_action,
            info,
            terminated,
            truncated,
        )

    def render(self):
        """
        No visual renderer yet.
        Use core_env.get_history_arrays() for plotting.
        """
        return None

    def close(self):
        return None

    def get_history_arrays(self):
        return self.core_env.get_history_arrays()
