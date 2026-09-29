"""Safe V15 behavior collection and actor distillation for SAC+HER."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.docking3dof.gym_env import Docking3DoFGymEnv
from her_closing_utils import make_her_env, write_csv


def load_source_policy(model_path, vecnormalize_path, device="auto"):
    dummy = DummyVecEnv(
        [
            lambda: Monitor(
                Docking3DoFGymEnv(
                    phase="closing",
                    mode="testing",
                    dt=1.0,
                    max_time=12_000.0,
                    integrator="rk4",
                    normalize_action=True,
                    seed=999_999,
                    closing_reward_version="v15",
                )
            )
        ]
    )
    vecnormalize = VecNormalize.load(str(vecnormalize_path), dummy)
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    model = SAC.load(str(model_path), device=device)
    return model, vecnormalize, dummy


def source_action(source_model, source_vecnormalize, base_observation):
    normalized = source_vecnormalize.normalize_obs(
        np.asarray(base_observation, dtype=np.float32).reshape(1, -1)
    )
    action, _ = source_model.predict(normalized, deterministic=True)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def _copy_goal_observation(observation):
    return {
        key: np.asarray(value, dtype=np.float32).copy()
        for key, value in observation.items()
    }


def _compact_info(info, *, truncated):
    keys = (
        "failure",
        "time_limit",
        "success",
        "port_speed",
        "s_dot",
        "psi_e",
        "omega",
        "state",
        "violated_constraints",
        "termination_reason",
    )
    compact = {key: copy.deepcopy(info[key]) for key in keys if key in info}
    compact["TimeLimit.truncated"] = bool(truncated)
    return compact


def collect_safe_source_trajectories(
    *,
    source_model,
    source_vecnormalize,
    minimum_transitions,
    seed_start,
    max_episodes,
    max_steps,
):
    """Collect complete non-failure V15 episodes in the V3 goal env."""
    accepted = []
    episode_summaries = []
    attempted = 0
    accepted_episode_id = 0

    while len(accepted) < minimum_transitions and attempted < max_episodes:
        seed = int(seed_start + attempted)
        attempted += 1
        env = make_her_env(
            seed=seed,
            mode="training",
            max_time=max_steps,
            reward_version="v3",
        )
        observation, _ = env.reset(seed=seed)
        episode = []
        final_info = {}
        for step in range(1, int(max_steps) + 1):
            base_observation = env.env._get_obs()
            action = source_action(
                source_model,
                source_vecnormalize,
                base_observation,
            )
            next_observation, reward, terminated, truncated, final_info = (
                env.step(action)
            )
            done = bool(terminated or truncated)
            episode.append(
                {
                    "observation": _copy_goal_observation(observation),
                    "next_observation": _copy_goal_observation(
                        next_observation
                    ),
                    "action": action.copy(),
                    "reward": float(reward),
                    "done": done,
                    "info": _compact_info(final_info, truncated=truncated),
                    "episode_id": accepted_episode_id,
                }
            )
            observation = next_observation
            if done:
                break

        failure = bool(final_info.get("failure", False))
        episode_summaries.append(
            {
                "attempt": attempted,
                "seed": seed,
                "steps": len(episode),
                "accepted": int(not failure),
                "failure": int(failure),
                "termination_reason": final_info.get(
                    "termination_reason",
                    "",
                ),
                "violated_constraints": str(
                    final_info.get("violated_constraints", [])
                ),
            }
        )
        if not failure:
            accepted.extend(episode)
            accepted_episode_id += 1
        env.close()

    if len(accepted) < minimum_transitions:
        raise RuntimeError(
            "The V15 source did not provide enough complete safe behavior: "
            f"collected {len(accepted):,} of {minimum_transitions:,} requested "
            f"transitions after {attempted} episodes."
        )
    return accepted, episode_summaries


def prefill_her_replay_buffer(model, transitions):
    replay_buffer = model.replay_buffer
    for transition in transitions:
        observation = {
            key: value.reshape(1, -1)
            for key, value in transition["observation"].items()
        }
        next_observation = {
            key: value.reshape(1, -1)
            for key, value in transition["next_observation"].items()
        }
        replay_buffer.add(
            observation,
            next_observation,
            transition["action"].reshape(1, -1),
            np.asarray([transition["reward"]], dtype=np.float32),
            np.asarray([transition["done"]], dtype=np.float32),
            [transition["info"]],
        )
    return {
        "prefill_transition_count": len(transitions),
        "prefill_episode_count": len(
            {transition["episode_id"] for transition in transitions}
        ),
        "replay_buffer_position_after_prefill": int(replay_buffer.pos),
        "replay_buffer_full_after_prefill": bool(replay_buffer.full),
    }


def _stack_observations(transitions):
    return {
        key: np.stack(
            [transition["observation"][key] for transition in transitions]
        ).astype(np.float32)
        for key in transitions[0]["observation"]
    }


def _actor_metrics(actor, observations, actions, indices, batch_size):
    squared_error = 0.0
    absolute_error = np.zeros(3, dtype=np.float64)
    count = 0
    actor.set_training_mode(False)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            obs_batch = {
                key: torch.as_tensor(values[batch_indices], device=actor.device)
                for key, values in observations.items()
            }
            targets = torch.as_tensor(actions[batch_indices], device=actor.device)
            predictions = actor(obs_batch, deterministic=True)
            error = predictions - targets
            squared_error += float(torch.sum(error.square()).item())
            absolute_error += np.sum(
                np.abs(error.detach().cpu().numpy()),
                axis=0,
            )
            count += len(batch_indices)
    return {
        "action_mse": squared_error / max(3 * count, 1),
        "action_mae": float(np.sum(absolute_error) / max(3 * count, 1)),
        "action_mae_ax": float(absolute_error[0] / max(count, 1)),
        "action_mae_ay": float(absolute_error[1] / max(count, 1)),
        "action_mae_alpha": float(absolute_error[2] / max(count, 1)),
    }


def behavior_distill_source_actor(
    *,
    model,
    transitions,
    epochs,
    batch_size,
    learning_rate,
    validation_fraction,
    split_seed,
    log_std_target,
    log_std_weight,
    result_path,
):
    observations = _stack_observations(transitions)
    actions = np.stack([item["action"] for item in transitions]).astype(
        np.float32
    )
    episode_ids = np.asarray(
        [item["episode_id"] for item in transitions],
        dtype=np.int64,
    )
    unique_episodes = np.unique(episode_ids)
    rng = np.random.default_rng(split_seed)
    shuffled_episodes = rng.permutation(unique_episodes)
    validation_count = max(
        1,
        int(round(validation_fraction * len(unique_episodes))),
    )
    validation_episodes = shuffled_episodes[:validation_count]
    training_episodes = shuffled_episodes[validation_count:]
    if len(training_episodes) == 0:
        training_episodes = validation_episodes
    training_indices = np.flatnonzero(np.isin(episode_ids, training_episodes))
    validation_indices = np.flatnonzero(
        np.isin(episode_ids, validation_episodes)
    )

    actor = model.actor
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    latent_targets = np.arctanh(np.clip(actions, -0.999, 0.999)).astype(
        np.float32
    )
    best_state = copy.deepcopy(actor.state_dict())
    best_mse = np.inf
    rows = []

    for epoch in range(1, epochs + 1):
        actor.set_training_mode(True)
        shuffled = rng.permutation(training_indices)
        latent_loss_sum = 0.0
        std_loss_sum = 0.0
        batches = 0
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            obs_batch = {
                key: torch.as_tensor(values[indices], device=model.device)
                for key, values in observations.items()
            }
            target_batch = torch.as_tensor(
                latent_targets[indices],
                device=model.device,
            )
            mean_actions, log_std, _ = actor.get_action_dist_params(obs_batch)
            latent_loss = F.smooth_l1_loss(mean_actions, target_batch)
            std_loss = F.mse_loss(
                log_std,
                torch.full_like(log_std, log_std_target),
            )
            loss = latent_loss + log_std_weight * std_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            optimizer.step()
            latent_loss_sum += float(latent_loss.item())
            std_loss_sum += float(std_loss.item())
            batches += 1

        metrics = _actor_metrics(
            actor,
            observations,
            actions,
            validation_indices,
            batch_size,
        )
        row = {
            "epoch": epoch,
            "mean_latent_huber_loss": latent_loss_sum / max(batches, 1),
            "mean_log_std_loss": std_loss_sum / max(batches, 1),
            **{f"validation_{key}": value for key, value in metrics.items()},
        }
        rows.append(row)
        print(
            f"Source BC epoch {epoch:02d}/{epochs:02d} | "
            f"MSE={metrics['action_mse']:.6f} | "
            f"MAE={metrics['action_mae']:.6f}",
            flush=True,
        )
        if metrics["action_mse"] < best_mse:
            best_mse = metrics["action_mse"]
            best_state = copy.deepcopy(actor.state_dict())

    actor.load_state_dict(best_state)
    write_csv(result_path, rows)
    return {
        "training_transition_count": int(len(training_indices)),
        "validation_transition_count": int(len(validation_indices)),
        "training_episode_ids": training_episodes.tolist(),
        "validation_episode_ids": validation_episodes.tolist(),
        "best_validation_action_mse": float(best_mse),
        **_actor_metrics(
            actor,
            observations,
            actions,
            validation_indices,
            batch_size,
        ),
    }
