# LQR Demonstration Learning

This directory contains SAC closing experiments that use successful LQR rollouts as offline training data.

## Experiment families

- `generate_lqr_replay_dataset.py` collects validated 1 Hz LQR demonstrations.
- `generate_lqr_expanded_coverage_dataset.py` expands the successful seed coverage with resumable checkpoints.
- `generate_lqr_replay_dataset_dt01.py` collects the equivalent 0.1 s dataset.
- `curate_lqr_demonstrations_dt01.py` creates separate behavior-cloning and consecutive-transition replay views.
- `train_sac_closing_lqr_replay.py` initializes replay with demonstrations before online SAC training.
- `train_sac_closing_lqr_mixed_replay.py` preserves a demonstration partition while sampling online experience.
- `train_sac_closing_lqr_bc_warmstart.py` initializes the actor with behavior cloning before SAC updates.
- `train_sac_closing_lqr_bc_constrained.py` retains a decaying supervised actor constraint during online training.
- `train_sac_closing_lqr_bc_constrained_dt01.py` provides the physical-time-equivalent 0.1 s configuration.
- `evaluate_sac_closing_lqr_replay_components.py` evaluates retained replay and cloning checkpoints.

The short training presets are retained because replay prefill, mixed replay, behavior-cloning warm start, persistent behavior constraint, and 0.1 s timing represent different algorithms or timing contracts.

## Artifacts

- `demonstrations/` contains the 1 Hz and 0.1 s datasets, collection progress, and collection logs.
- `models/`, `logs/`, and `results/` are grouped by training method.
- `results/retained_figures/` contains retained replay-versus-LQR comparisons.
- Run-local `retained_figures/` directories contain behavior-cloning diagnostics.
- `plotters/` regenerates family-specific comparisons.
- `tests/` verifies dataset integrity, timing equivalence, replay partitioning, and behavior-cloning losses.

The large demonstration `.npz` files are managed by Git LFS. A normal clone
may contain small pointer files instead of the dataset payloads; this is
intentional and keeps the initial checkout manageable. The datasets are only
required for regenerating or continuing the demonstration-replay and
behavior-cloning experiments. Users who need those workflows can retrieve the
uploaded payloads with `git lfs pull`. The retained SAC policy checkpoints and
ordinary controller evaluations do not require these demonstration datasets.

`behavior_cloned_warmup_sac.py` remains at the closing root because the residual-LQR family also imports it.
