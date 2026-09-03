import json
from pathlib import Path
import sys

import numpy as np


CLOSING_PATH = Path(__file__).resolve().parents[1]
if str(CLOSING_PATH) not in sys.path:
    sys.path.insert(0, str(CLOSING_PATH))

from generate_lqr_replay_dataset import (
    _metadata_from_dataset,
    build_lqr_controller,
    collect_lqr_episode,
    combine_successful_episodes,
    make_demonstration_env,
    parse_args as parse_dataset_args,
)
from generate_lqr_replay_dataset_dt01 import parse_args as parse_dt01_dataset_args
from curate_lqr_demonstrations_dt01 import (
    evenly_spaced_episode_indices,
    select_episode_view,
)
from train_sac_closing_lqr_bc_constrained import (
    parse_args as parse_bc_args,
    resolve_bc_action_loss_scales,
)
from train_sac_closing_lqr_bc_constrained_dt01 import (
    build_argv as bc_dt01_argv,
)
from train_sac_closing_best_standard_dt01 import (
    build_argv as standard_dt01_argv,
)
from train_sac_closing_dt01 import (
    build_experiment_config as build_standard_config,
    parse_args as parse_standard_args,
)


def test_dt01_dataset_cli_records_canonical_timing():
    args = parse_dt01_dataset_args(["--target-successes", "1"])
    assert args.target_successes == 1
    assert args.seed_start == 20_000
    assert args.checkpoint_every_successes == 25


def test_bc_dt01_defaults_are_physical_time_equivalents():
    args = parse_bc_args(bc_dt01_argv([]))
    assert args.dt == 0.1
    assert args.total_timesteps == 6_000_000
    assert args.critic_warmup_steps == 500_000
    assert args.deterministic_finetune_steps == 500_000
    assert args.bc_finetune_decay_steps == 2_500_000
    assert args.evaluation_interval == 100_000
    assert args.max_eval_steps == 120_000
    assert args.train_freq == 160
    assert args.gradient_steps == 8
    assert np.isclose(args.gamma, 0.99**0.1)
    assert args.bc_action_loss_scaling == "mean_abs"
    assert args.bc_demonstrations.name.endswith("bc_stride10_78episodes.npz")
    assert args.replay_demonstrations.name.endswith("replay_20episodes.npz")
    assert args.bc_demonstrations != args.replay_demonstrations


def test_best_standard_dt01_preset_uses_full_physical_budget():
    args = parse_standard_args(standard_dt01_argv([]))
    config = build_standard_config(args)
    assert config["dt_seconds"] == 0.1
    assert config["total_timesteps"] == 4_000_000
    assert config["simulated_training_time_seconds"] == 400_000.0
    assert config["hyperparameters"]["buffer_size"] == 3_000_000
    assert config["hyperparameters"]["learning_starts"] == 100_000
    assert config["hyperparameters"]["train_freq"] == 160
    assert config["hyperparameters"]["gradient_steps"] == 8
    assert np.isclose(config["hyperparameters"]["gamma"], 0.99**0.1)


def test_dt01_lqr_environment_uses_retuned_controller_and_reward():
    env = make_demonstration_env(
        mode="testing",
        seed=41,
        reward_version="v15_dt",
        dt=0.1,
        max_time=1.0,
        action_hold_steps=1,
    )
    try:
        _, info = env.reset(seed=41)
        controller = build_lqr_controller(env.core_env)
        action = controller.compute_action(env.core_env, env.core_env.state)
        assert action.shape == (3,)
        assert np.isclose(info["integration_step_seconds"], 0.1)
        assert np.isclose(info["control_interval_seconds"], 0.1)
        assert env.closing_reward_version == "v15_dt"
    finally:
        env.close()


def test_dataset_metadata_can_pin_transition_timing():
    episode = {
        "seed": 7,
        "success": True,
        "termination_reason": "success_P3_target",
        "observations": np.zeros((1, 12), dtype=np.float32),
        "actions": np.zeros((1, 3), dtype=np.float32),
        "rewards": np.zeros(1, dtype=np.float32),
        "next_observations": np.zeros((1, 12), dtype=np.float32),
        "terminated": np.ones(1, dtype=np.bool_),
        "truncated": np.zeros(1, dtype=np.bool_),
        "initial_state": np.zeros(6),
        "p3_target": np.zeros(6),
        "final_state": np.zeros(6),
        "episode_return": 0.0,
        "saturation_fraction": 0.0,
    }
    dataset = combine_successful_episodes(
        [episode],
        reward_version="v15_dt",
        mode="training",
        metadata_overrides={
            "dt_seconds": 0.1,
            "control_interval_seconds": 0.1,
            "control_frequency_hz": 10.0,
            "action_hold_steps": 1,
            "lqr_q_timestep_seconds": 0.1,
        },
    )
    metadata = _metadata_from_dataset(dataset)
    assert json.loads(str(dataset["metadata_json"].item())) == metadata
    assert metadata["dt_seconds"] == 0.1
    assert metadata["lqr_q_timestep_seconds"] == 0.1


def test_dt01_curation_thins_bc_but_preserves_complete_replay_episodes():
    episodes = []
    for seed, length in ((10, 11), (11, 8), (12, 6), (13, 9)):
        terminated = np.zeros(length, dtype=np.bool_)
        terminated[-1] = True
        episodes.append(
            {
                "seed": seed,
                "success": True,
                "termination_reason": "success_P3_target",
                "observations": np.full((length, 12), seed, dtype=np.float32),
                "actions": np.zeros((length, 3), dtype=np.float32),
                "rewards": np.zeros(length, dtype=np.float32),
                "next_observations": np.full(
                    (length, 12), seed + 0.1, dtype=np.float32
                ),
                "terminated": terminated,
                "truncated": np.zeros(length, dtype=np.bool_),
                "initial_state": np.zeros(6),
                "p3_target": np.zeros(6),
                "final_state": np.zeros(6),
                "episode_return": 0.0,
                "saturation_fraction": 0.0,
            }
        )
    source = combine_successful_episodes(
        episodes,
        reward_version="v15_dt",
        mode="training",
        metadata_overrides={"dt_seconds": 0.1, "action_hold_steps": 1},
    )

    bc_view = select_episode_view(
        source,
        episode_indices=np.arange(4),
        temporal_stride=3,
        role="behavior_cloning_supervision",
    )
    replay_indices = evenly_spaced_episode_indices(4, 2)
    replay_view = select_episode_view(
        source,
        episode_indices=replay_indices,
        temporal_stride=1,
        role="protected_critic_replay",
    )

    bc_metadata = _metadata_from_dataset(bc_view)
    replay_metadata = _metadata_from_dataset(replay_view)
    assert bc_metadata["successful_episodes"] == 4
    assert bc_metadata["temporal_stride"] == 3
    assert bc_metadata["contains_consecutive_transitions"] is False
    assert np.all(bc_view["terminated"][bc_view["episode_offsets"][1:] - 1])
    assert replay_indices.tolist() == [0, 3]
    assert replay_metadata["contains_consecutive_transitions"] is True
    assert replay_metadata["transition_count"] == 20


def test_mean_absolute_action_scaling_preserves_small_control_signal():
    actions = np.asarray(
        [[0.001, -0.004, 0.0], [-0.003, 0.002, 0.0002]],
        dtype=np.float32,
    )
    scales = resolve_bc_action_loss_scales(
        actions,
        scaling="mean_abs",
        scale_floor=1e-3,
    )
    assert np.allclose(scales, [0.002, 0.003, 0.001])
