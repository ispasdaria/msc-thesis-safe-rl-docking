# Residual LQR Experiments

This directory contains SAC controllers that learn corrections around the closing LQR controller or its reference trajectory.

## Experiment groups

- Direct residual environments add bounded SAC actuator commands to the LQR command.
- Reference-residual environments modify the position, velocity, attitude, and rate references tracked by the LQR controller.
- HER variants relabel achieved closing goals during replay.
- Non-HER variants use ordinary replay for controlled comparisons.
- Terminal-reference experiments use a curated near-P3 state bank and protected replay partitions.
- Pilot E, G, and J launchers retain the exact configurations used for successive reference-residual evaluations.
- `validate_residual_lqr_dt01_baseline.py` verifies the zero-residual 0.1 s controller before learned residuals are evaluated.

## Contents

- Environment, replay, and constrained-SAC helpers remain directly in this family because they are not used by other closing families.
- `demonstrations/` contains the final-LQR terminal state bank and its collection record.
- `models/`, `logs/`, and `results/` preserve direct, reference, HER, stratified, and terminal experiment artifacts.
- `results/residual_lqr_baseline/` contains the canonical zero-residual validation.
- `results/retained_figures/` contains the retained residual-versus-LQR comparison.
- `plotters/` contains family-specific plotting code.
- `tests/` verifies environment equivalence, replay behavior, pilot configurations, terminal-bank provenance, and baseline validation.

Historical manifests retain their original source paths. Validation resolves relocated residual sources while continuing to verify the stored numeric controller fingerprint.
