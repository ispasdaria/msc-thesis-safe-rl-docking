"""SAC with critic warm-up and behavior-constrained actor fine-tuning."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update

SCRIPT_DIR = Path(__file__).resolve().parent
CLOSING_DIR = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, CLOSING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from behavior_cloned_warmup_sac import BehaviorClonedWarmupSAC


class BehaviorClonedConstrainedSAC(BehaviorClonedWarmupSAC):
    """Preserve a cloned actor with a decaying supervised loss.

    The actor remains frozen during the inherited critic warm-up. Afterwards,
    ordinary SAC actor updates are augmented with behavior cloning on a fresh
    demonstration batch. Demonstration actions supervise both the deterministic
    action mean and, optionally, the actor's log standard deviation.
    """

    def __init__(
        self,
        *args,
        bc_finetune_weight_start=10.0,
        bc_finetune_weight_end=1.0,
        bc_finetune_decay_steps=250_000,
        bc_finetune_batch_size=None,
        bc_action_clip=0.999,
        bc_action_loss_scales=None,
        bc_log_std_target=-3.0,
        bc_log_std_weight=0.01,
        actor_learning_rate_scale=0.1,
        deterministic_finetune_steps=50_000,
        **kwargs,
    ):
        values = {
            "bc_finetune_weight_start": bc_finetune_weight_start,
            "bc_finetune_weight_end": bc_finetune_weight_end,
            "bc_log_std_weight": bc_log_std_weight,
            "actor_learning_rate_scale": actor_learning_rate_scale,
        }
        if any(float(value) < 0.0 for value in values.values()):
            raise ValueError(f"Non-negative constrained-SAC values required: {values}")
        if int(bc_finetune_decay_steps) < 0:
            raise ValueError("bc_finetune_decay_steps must be non-negative.")
        if bc_finetune_batch_size is not None and int(bc_finetune_batch_size) <= 0:
            raise ValueError("bc_finetune_batch_size must be positive when set.")
        if not 0.0 < float(bc_action_clip) < 1.0:
            raise ValueError("bc_action_clip must lie strictly between zero and one.")
        if bc_action_loss_scales is None:
            bc_action_loss_scales = np.ones(3, dtype=np.float32)
        bc_action_loss_scales = np.asarray(
            bc_action_loss_scales,
            dtype=np.float32,
        )
        if (
            bc_action_loss_scales.shape != (3,)
            or np.any(~np.isfinite(bc_action_loss_scales))
            or np.any(bc_action_loss_scales <= 0.0)
        ):
            raise ValueError(
                "bc_action_loss_scales must contain three positive finite values."
            )
        if int(deterministic_finetune_steps) < 0:
            raise ValueError("deterministic_finetune_steps must be non-negative.")

        self.bc_finetune_weight_start = float(bc_finetune_weight_start)
        self.bc_finetune_weight_end = float(bc_finetune_weight_end)
        self.bc_finetune_decay_steps = int(bc_finetune_decay_steps)
        self.bc_finetune_batch_size = (
            None
            if bc_finetune_batch_size is None
            else int(bc_finetune_batch_size)
        )
        self.bc_action_clip = float(bc_action_clip)
        self.bc_action_loss_scales = bc_action_loss_scales
        self.bc_log_std_target = float(bc_log_std_target)
        self.bc_log_std_weight = float(bc_log_std_weight)
        self.actor_learning_rate_scale = float(actor_learning_rate_scale)
        self.deterministic_finetune_steps = int(deterministic_finetune_steps)
        self.constrained_actor_gradient_updates = 0
        super().__init__(*args, **kwargs)

    @property
    def bc_finetune_weight(self):
        """Linearly interpolate the BC weight after critic warm-up."""

        elapsed = max(0, self.num_timesteps - self.critic_warmup_steps)
        if self.bc_finetune_decay_steps == 0:
            fraction = 1.0
        else:
            fraction = min(1.0, elapsed / self.bc_finetune_decay_steps)
        return (
            self.bc_finetune_weight_start
            + fraction
            * (self.bc_finetune_weight_end - self.bc_finetune_weight_start)
        )

    @property
    def use_deterministic_collection(self):
        return self.num_timesteps < (
            self.critic_warmup_steps + self.deterministic_finetune_steps
        )

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.use_deterministic_collection:
            assert self._last_obs is not None, "The latest observation is missing."
            unscaled_action, _ = self.predict(
                self._last_obs,
                deterministic=True,
            )
            scaled_action = self.policy.scale_action(unscaled_action)
            if action_noise is not None:
                scaled_action = np.clip(
                    scaled_action + action_noise(),
                    -1.0,
                    1.0,
                )
            return self.policy.unscale_action(scaled_action), scaled_action
        return super()._sample_action(
            learning_starts,
            action_noise=action_noise,
            n_envs=n_envs,
        )

    def _demonstration_bc_loss(self, batch_size):
        replay_buffer = self.replay_buffer
        sampler = getattr(
            replay_buffer,
            "_get_behavior_cloning_samples",
            None,
        )
        if sampler is None:
            demonstration_data = replay_buffer._get_demonstration_samples(
                batch_size,
                env=self._vec_normalize_env,
            )
            observations = demonstration_data.observations
            actions = demonstration_data.actions
        else:
            observations, actions = sampler(
                batch_size,
                env=self._vec_normalize_env,
            )
        mean_actions, log_std, _ = self.actor.get_action_dist_params(
            observations
        )
        clipped_targets = th.clamp(
            actions,
            -self.bc_action_clip,
            self.bc_action_clip,
        )
        latent_targets = th.atanh(clipped_targets)
        action_scales = th.as_tensor(
            self.bc_action_loss_scales,
            dtype=mean_actions.dtype,
            device=mean_actions.device,
        )
        scaled_error = (mean_actions - latent_targets) / action_scales
        mean_loss = F.smooth_l1_loss(
            scaled_error,
            th.zeros_like(scaled_error),
        )
        target_log_std = th.full_like(log_std, self.bc_log_std_target)
        log_std_loss = F.mse_loss(log_std, target_log_std)
        total_loss = mean_loss + self.bc_log_std_weight * log_std_loss
        return total_loss, mean_loss, log_std_loss

    def train(self, gradient_steps, batch_size=64):
        if self.num_timesteps <= self.critic_warmup_steps:
            super().train(gradient_steps=gradient_steps, batch_size=batch_size)
            return
        self._train_constrained(gradient_steps, batch_size)

    def _train_constrained(self, gradient_steps, batch_size):
        self.policy.set_training_mode(True)

        # Critics and entropy use the configured SAC rate. The actor uses a
        # smaller rate so that the handoff from the cloned policy is gradual.
        optimizers = [self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)
        actor_learning_rate = (
            self.lr_schedule(self._current_progress_remaining)
            * self.actor_learning_rate_scale
        )
        for parameter_group in self.actor.optimizer.param_groups:
            parameter_group["lr"] = actor_learning_rate

        entropy_losses = []
        entropy_coefficients = []
        sac_actor_losses = []
        bc_losses = []
        bc_mean_losses = []
        bc_log_std_losses = []
        total_actor_losses = []
        critic_losses = []
        bc_batch_size = self.bc_finetune_batch_size or batch_size

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env,
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            if self.use_sde:
                self.actor.reset_noise()
            actions_pi, log_prob = self.actor.action_log_prob(
                replay_data.observations
            )
            log_prob = log_prob.reshape(-1, 1)

            entropy_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                entropy_coefficient = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                entropy_loss = -(
                    self.log_ent_coef
                    * (log_prob + self.target_entropy).detach()
                ).mean()
                entropy_losses.append(float(entropy_loss.item()))
            else:
                entropy_coefficient = self.ent_coef_tensor
            entropy_coefficients.append(float(entropy_coefficient.item()))

            if entropy_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                entropy_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = th.cat(
                    self.critic_target(
                        replay_data.next_observations,
                        next_actions,
                    ),
                    dim=1,
                )
                next_q_values, _ = th.min(
                    next_q_values,
                    dim=1,
                    keepdim=True,
                )
                next_q_values = next_q_values - (
                    entropy_coefficient * next_log_prob.reshape(-1, 1)
                )
                target_q_values = replay_data.rewards + (
                    (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.critic(
                replay_data.observations,
                replay_data.actions,
            )
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            critic_losses.append(float(critic_loss.item()))
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(
                self.critic(replay_data.observations, actions_pi),
                dim=1,
            )
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            sac_actor_loss = (
                entropy_coefficient * log_prob - min_qf_pi
            ).mean()
            bc_loss, bc_mean_loss, bc_log_std_loss = (
                self._demonstration_bc_loss(bc_batch_size)
            )
            weight = self.bc_finetune_weight
            total_actor_loss = sac_actor_loss + weight * bc_loss

            sac_actor_losses.append(float(sac_actor_loss.item()))
            bc_losses.append(float(bc_loss.item()))
            bc_mean_losses.append(float(bc_mean_loss.item()))
            bc_log_std_losses.append(float(bc_log_std_loss.item()))
            total_actor_losses.append(float(total_actor_loss.item()))

            self.actor.optimizer.zero_grad()
            total_actor_loss.backward()
            self.actor.optimizer.step()
            self.constrained_actor_gradient_updates += 1

            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.batch_norm_stats,
                    self.batch_norm_stats_target,
                    1.0,
                )

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(entropy_coefficients))
        self.logger.record("train/actor_loss", np.mean(total_actor_losses))
        self.logger.record("train/sac_actor_loss", np.mean(sac_actor_losses))
        self.logger.record("train/bc_finetune_loss", np.mean(bc_losses))
        self.logger.record("train/bc_mean_loss", np.mean(bc_mean_losses))
        self.logger.record("train/bc_log_std_loss", np.mean(bc_log_std_losses))
        self.logger.record("train/bc_finetune_weight", self.bc_finetune_weight)
        self.logger.record("train/actor_learning_rate", actor_learning_rate)
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record(
            "train/deterministic_collection",
            float(self.use_deterministic_collection),
        )
        if entropy_losses:
            self.logger.record("train/ent_coef_loss", np.mean(entropy_losses))
