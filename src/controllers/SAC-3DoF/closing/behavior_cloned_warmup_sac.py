"""SAC variant that protects a behavior-cloned actor during critic warm-up."""

from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class BehaviorClonedWarmupSAC(SAC):
    """Use a deterministic frozen actor, then transition to ordinary SAC.

    Until ``critic_warmup_steps`` online transitions have been collected:

    * environment actions use the actor's deterministic mean;
    * only critic parameters and critic target parameters are updated;
    * actor and entropy-coefficient parameters remain unchanged.

    Once the threshold is crossed, action sampling and optimization delegate
    to the unmodified Stable-Baselines3 SAC implementation.
    """

    def __init__(
        self,
        *args,
        critic_warmup_steps=50_000,
        critic_loss_type="mse",
        max_critic_grad_norm=None,
        **kwargs,
    ):
        if int(critic_warmup_steps) < 0:
            raise ValueError("critic_warmup_steps must be non-negative.")
        if critic_loss_type not in {"mse", "huber"}:
            raise ValueError("critic_loss_type must be 'mse' or 'huber'.")
        if max_critic_grad_norm is not None and float(max_critic_grad_norm) <= 0:
            raise ValueError("max_critic_grad_norm must be positive when set.")
        self.critic_warmup_steps = int(critic_warmup_steps)
        self.critic_loss_type = str(critic_loss_type)
        self.max_critic_grad_norm = (
            None
            if max_critic_grad_norm is None
            else float(max_critic_grad_norm)
        )
        self.critic_warmup_gradient_updates = 0
        super().__init__(*args, **kwargs)

    def critic_regression_loss(self, current_q_values, target_q_values):
        """Return the configured two-critic regression objective."""
        loss_type = getattr(self, "critic_loss_type", "mse")
        loss_function = F.smooth_l1_loss if loss_type == "huber" else F.mse_loss
        return 0.5 * sum(
            loss_function(current_q, target_q_values)
            for current_q in current_q_values
        )

    def clip_critic_gradients(self):
        """Apply the optional global critic gradient-norm guard."""
        maximum_norm = getattr(self, "max_critic_grad_norm", None)
        if maximum_norm is None:
            return None
        return th.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            maximum_norm,
        )

    @property
    def in_critic_warmup(self):
        return self.num_timesteps < self.critic_warmup_steps

    def _sample_action(
        self,
        learning_starts,
        action_noise=None,
        n_envs=1,
    ):
        if not self.in_critic_warmup:
            return super()._sample_action(
                learning_starts,
                action_noise=action_noise,
                n_envs=n_envs,
            )

        assert self._last_obs is not None, "The latest observation is missing."
        unscaled_action, _ = self.predict(
            self._last_obs,
            deterministic=True,
        )
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)
            if action_noise is not None:
                scaled_action = np.clip(
                    scaled_action + action_noise(),
                    -1.0,
                    1.0,
                )
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def train(self, gradient_steps, batch_size=64):
        # A train call occurs after collecting one complete train_freq block.
        # Keep the final block that reaches the threshold critic-only as well;
        # the next rollout then begins standard stochastic SAC behavior.
        if self.num_timesteps <= self.critic_warmup_steps:
            self._train_critics_only(gradient_steps, batch_size)
            return
        super().train(gradient_steps=gradient_steps, batch_size=batch_size)

    def _train_critics_only(self, gradient_steps, batch_size):
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.critic.optimizer])
        critic_losses = []

        # Entropy is deliberately omitted during this phase: the critic learns
        # the deterministic cloned policy that is actually collecting data.
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
            with th.no_grad():
                next_actions = self.actor(
                    replay_data.next_observations,
                    deterministic=True,
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
                target_q_values = replay_data.rewards + (
                    (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.critic(
                replay_data.observations,
                replay_data.actions,
            )
            critic_loss = self.critic_regression_loss(
                current_q_values,
                target_q_values,
            )
            critic_losses.append(float(critic_loss.item()))
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.clip_critic_gradients()
            self.critic.optimizer.step()

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
        self.critic_warmup_gradient_updates += gradient_steps
        self.logger.record(
            "train/n_updates",
            self._n_updates,
            exclude="tensorboard",
        )
        self.logger.record("train/phase_critic_warmup", 1.0)
        self.logger.record(
            "train/critic_warmup_loss",
            float(np.mean(critic_losses)),
        )
