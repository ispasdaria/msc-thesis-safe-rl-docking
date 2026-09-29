"""SAC with critic warm-up and an explicit zero-residual actor prior."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch as th
from stable_baselines3.common.utils import polyak_update

SCRIPT_DIR = Path(__file__).resolve().parent
CLOSING_DIR = SCRIPT_DIR.parent
if str(CLOSING_DIR) not in sys.path:
    sys.path.insert(0, str(CLOSING_DIR))

from behavior_cloned_warmup_sac import BehaviorClonedWarmupSAC


class ZeroResidualConstrainedSAC(BehaviorClonedWarmupSAC):
    """Prevent residual-policy collapse into unsupported action-space corners.

    During critic warm-up the inherited deterministic actor remains frozen at
    zero.  Subsequent SAC actor updates include ``weight * ||action||^2`` in
    normalized residual coordinates.  The critic is still trained on the task
    reward, so a nonzero residual is selected only when its learned value gain
    exceeds the explicit preference for the nominal LQR controller.
    """

    def __init__(
        self,
        *args,
        zero_residual_actor_weight=20.0,
        zero_residual_actor_final_weight=None,
        zero_residual_actor_anneal_end_step=None,
        actor_learning_rate_scale=0.1,
        max_actor_grad_norm=None,
        **kwargs,
    ):
        if float(zero_residual_actor_weight) < 0.0:
            raise ValueError("zero_residual_actor_weight must be non-negative.")
        if (
            zero_residual_actor_final_weight is not None
            and float(zero_residual_actor_final_weight) < 0.0
        ):
            raise ValueError(
                "zero_residual_actor_final_weight must be non-negative."
            )
        if (
            zero_residual_actor_anneal_end_step is not None
            and int(zero_residual_actor_anneal_end_step) < 0
        ):
            raise ValueError(
                "zero_residual_actor_anneal_end_step must be non-negative."
            )
        if float(actor_learning_rate_scale) < 0.0:
            raise ValueError("actor_learning_rate_scale must be non-negative.")
        if max_actor_grad_norm is not None and float(max_actor_grad_norm) <= 0:
            raise ValueError("max_actor_grad_norm must be positive when set.")
        self.zero_residual_actor_weight = float(zero_residual_actor_weight)
        self.zero_residual_actor_final_weight = (
            self.zero_residual_actor_weight
            if zero_residual_actor_final_weight is None
            else float(zero_residual_actor_final_weight)
        )
        self.zero_residual_actor_anneal_end_step = (
            None
            if zero_residual_actor_anneal_end_step is None
            else int(zero_residual_actor_anneal_end_step)
        )
        self.actor_learning_rate_scale = float(actor_learning_rate_scale)
        self.max_actor_grad_norm = (
            None if max_actor_grad_norm is None else float(max_actor_grad_norm)
        )
        self.constrained_actor_gradient_updates = 0
        super().__init__(*args, **kwargs)
        if (
            self.zero_residual_actor_anneal_end_step is not None
            and self.zero_residual_actor_anneal_end_step
            <= self.critic_warmup_steps
        ):
            raise ValueError(
                "zero_residual_actor_anneal_end_step must be greater than "
                "critic_warmup_steps."
            )

    def current_zero_residual_actor_weight(self, timestep=None):
        """Return the scheduled prior weight at an absolute interaction step."""
        timestep = self.num_timesteps if timestep is None else int(timestep)
        final_weight = getattr(
            self,
            "zero_residual_actor_final_weight",
            self.zero_residual_actor_weight,
        )
        anneal_end = getattr(
            self,
            "zero_residual_actor_anneal_end_step",
            None,
        )
        if anneal_end is None or timestep <= self.critic_warmup_steps:
            return self.zero_residual_actor_weight
        if timestep >= anneal_end:
            return final_weight
        fraction = (
            (timestep - self.critic_warmup_steps)
            / (anneal_end - self.critic_warmup_steps)
        )
        return self.zero_residual_actor_weight + fraction * (
            final_weight - self.zero_residual_actor_weight
        )

    def train(self, gradient_steps, batch_size=64):
        if self.num_timesteps <= self.critic_warmup_steps:
            super().train(gradient_steps=gradient_steps, batch_size=batch_size)
            return
        self._train_zero_residual_constrained(gradient_steps, batch_size)

    def _train_zero_residual_constrained(self, gradient_steps, batch_size):
        self.policy.set_training_mode(True)
        optimizers = [self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)
        actor_learning_rate = (
            self.lr_schedule(self._current_progress_remaining)
            * self.actor_learning_rate_scale
        )
        current_prior_weight = self.current_zero_residual_actor_weight()
        for parameter_group in self.actor.optimizer.param_groups:
            parameter_group["lr"] = actor_learning_rate

        entropy_losses = []
        entropy_coefficients = []
        sac_actor_losses = []
        residual_prior_losses = []
        total_actor_losses = []
        critic_losses = []
        target_q_means = []
        target_q_abs_maxima = []
        current_q_means = []
        current_q_abs_maxima = []

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
            critic_loss = self.critic_regression_loss(
                current_q_values,
                target_q_values,
            )
            critic_losses.append(float(critic_loss.item()))
            target_q_means.append(float(target_q_values.mean().item()))
            target_q_abs_maxima.append(float(target_q_values.abs().max().item()))
            current_q_tensor = th.cat(current_q_values, dim=1)
            current_q_means.append(float(current_q_tensor.mean().item()))
            current_q_abs_maxima.append(
                float(current_q_tensor.abs().max().item())
            )
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.clip_critic_gradients()
            self.critic.optimizer.step()

            q_values_pi = th.cat(
                self.critic(replay_data.observations, actions_pi),
                dim=1,
            )
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            sac_actor_loss = (
                entropy_coefficient * log_prob - min_qf_pi
            ).mean()
            residual_prior_loss = th.mean(
                th.sum(th.square(actions_pi), dim=1)
            )
            total_actor_loss = sac_actor_loss + (
                current_prior_weight * residual_prior_loss
            )

            sac_actor_losses.append(float(sac_actor_loss.item()))
            residual_prior_losses.append(float(residual_prior_loss.item()))
            total_actor_losses.append(float(total_actor_loss.item()))
            self.actor.optimizer.zero_grad()
            total_actor_loss.backward()
            maximum_actor_norm = getattr(self, "max_actor_grad_norm", None)
            if maximum_actor_norm is not None:
                th.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    maximum_actor_norm,
                )
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
        self.logger.record(
            "train/zero_residual_prior_loss",
            np.mean(residual_prior_losses),
        )
        self.logger.record(
            "train/zero_residual_actor_weight",
            current_prior_weight,
        )
        self.logger.record("train/actor_learning_rate", actor_learning_rate)
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/target_q_mean", np.mean(target_q_means))
        self.logger.record(
            "train/target_q_abs_max",
            np.max(target_q_abs_maxima),
        )
        self.logger.record("train/current_q_mean", np.mean(current_q_means))
        self.logger.record(
            "train/current_q_abs_max",
            np.max(current_q_abs_maxima),
        )
        self.logger.record("train/phase_critic_warmup", 0.0)
        if entropy_losses:
            self.logger.record("train/ent_coef_loss", np.mean(entropy_losses))
