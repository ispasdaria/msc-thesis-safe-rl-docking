# Curriculum and P3-Reach Experiments

This directory contains SAC closing experiments that modify the reset distribution, terminal conditions, or reward structure to acquire the P3 handover state.

## Experiment progression

- `train_sac_closing_curriculum.py` expands training from a bounded near-P3 reset distribution to the full P2 closing distribution when fixed validation thresholds are met.
- `train_sac_closing_no_safety_termination.py` continues a reward-v15 policy while safety violations remain penalized but non-terminal.
- `train_sac_closing_p3reach_from_scratch.py` trains a new policy and normalization state on the isolated P3-reaching reward.
- `train_sac_closing_braking_envelope.py` selects the V19 axial braking-envelope environment.
- `train_sac_closing_braking_terminal_v20.py` selects the V20 terminal-reward formulation.
- `train_sac_closing_port_lqr_envelope_v21.py` selects the V21 docking-port tracking envelope.
- `train_sac_closing_adaptive_dt01_terminal.py` continues the selected guarded policy with 0.1 s integration and distance-dependent action cadence.
- `tune_sac_closing_p3reach_v12_optuna.py` tunes the V12 P3-reaching configuration from newly initialized policies.
- `evaluate_sac_closing_p3reach.py` evaluates stored P3-reaching checkpoints on fixed seeds.

The short V19, V20, and V21 launchers are retained because each selects a distinct environment and checkpoint-ranking rule. They do not duplicate the common training implementation.

## Contents

- `models/` contains checkpoints and matching `VecNormalize` states.
- `logs/` contains TensorBoard event data and tuning logs.
- `results/` contains rollout summaries, diagnostics, and generated figures.
- `plotters/` contains family-specific plotting entry points.
- `tests/` verifies the adaptive-cadence implementation and P3 diagnostic calculations.

The vanilla source checkpoints used by continuation experiments remain in `../vanilla_sac/models/`. Shared reset wrappers and phase-balanced replay support remain at the closing root until their owning families are organized.

All executable paths are derived from the script location, so commands can be run from the repository root.
