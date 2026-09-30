# Phase-Balanced Replay Experiments

This directory contains SAC closing experiments that partition online experience by mission phase so terminal and safety-relevant transitions remain represented during training.

## Variants

- `train_sac_closing_phase_balanced_replay_v12.py` applies phase-balanced replay to the isolated V12 P3-reaching environment.
- `train_sac_closing_v15_phase_balanced_replay.py` continues the selected reward-v15 checkpoint with global and capture replay partitions.
- `train_sac_closing_v15_guarded_simultaneous_target_replay.py` adds protected replay partitions and guarded checkpoint selection.
- `train_sac_closing_v15_deterministic_near_p3_replay.py` configures deterministic near-P3 resets without event-priority sampling.
- `train_sac_closing_v15_deterministic_near_p3_replay_cbf.py` adds the deployment CBF to the deterministic near-P3 configuration.

The short variant launchers are retained because they encode distinct replay and checkpoint-selection configurations.

## Contents

- `phase_balanced_replay_buffer_implementation.py` contains the original phase-balanced replay-buffer implementation, while `guarded_simultaneous_target_replay_buffer.py` extends it with guarded simultaneous-target sampling.
- `models/multiseed/`, `logs/multiseed/`, and `results/multiseed/` contain the associated training runs.
- `results/multiseed_evaluation/` contains deterministic held-out evaluations.
- `results/retained_figures/` contains figures and diagnostics for the retained checkpoint.
- `plotters/` regenerates retained-policy comparisons and diagnostics.
- `tests/` verifies replay sampling, partition protection, and checkpoint readiness.

The continuation source checkpoint remains under `../vanilla_sac/models/`. Timestep-transfer results used by retained-policy comparisons remain with the timestep-transfer family until that family is migrated.
