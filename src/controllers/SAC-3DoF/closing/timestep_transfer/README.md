# Timestep Transfer Experiments

This directory evaluates how closing policies behave when the simulator integration step or policy decision cadence changes.

## Evaluation modes

- `native_1hz` uses 1.0 s integration and one policy decision per second.
- `dt0p1_hold_1s` integrates at 0.1 s while holding each policy action for ten simulator steps.
- `dt0p1_recompute_10hz` integrates at 0.1 s and evaluates the policy at every simulator step.
- Actor-transfer runs copy the retained 1 Hz actor into a new 0.1 s SAC model while reinitializing critics, optimizers, replay, and normalization.

`evaluate_sac_closing_timestep_transfer.py` performs deterministic paired-seed comparisons. `train_sac_closing_uniform_dt01_actor_transfer.py` performs actor-only transfer. The horizon plotter compares the same retained vanilla policy at 12,000 s, 13,000 s, and 20,000 s horizons.

## Contents

- `models/uniform_dt01_actor_transfer/`, `logs/uniform_dt01_actor_transfer/`, and `results/uniform_dt01_actor_transfer/` contain actor-transfer training artifacts.
- `results/timestep_transfer/` contains paired evaluation groups named by policy, seed range, horizon, and execution mode.
- `plotters/` contains timing-comparison plotting code.
- `tests/` verifies physical-time conversions, evaluation modes, actor copying, replay warm-up, and source-checkpoint resolution.
- `episode_stratified_replay_buffer.py` implements the family-specific episode-balanced replay used by actor-transfer training.

The retained source checkpoints remain in the phase-balanced and vanilla families and are referenced through paths derived from each script location.
