# Residual LQR Experiments

This directory contains SAC controllers that learn corrections around the closing LQR controller or its reference trajectory.

## Experiment groups

- Direct residual environments add bounded SAC actuator commands to the LQR command.
- Reference-residual environments modify the position, velocity, attitude, and rate references tracked by the LQR controller.
- HER variants relabel achieved closing goals during replay.
- Non-HER variants use ordinary replay for controlled comparisons.
- Terminal-reference experiments use a curated near-P3 state bank and protected replay partitions.
- Fixed launchers retain the annealed-actor-prior, lateral-reward, and terminal-reference configurations used for successive reference-residual evaluations.
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

## Principal entry points

- `train_sac_closing_v15_source_warmstarted_her.py` distills and replays the fixed V15 source policy before HER training.
- `train_sac_closing_v15_baseline_residual_her.py` trains HER residual actions around the frozen V15 controller.
- `train_sac_closing_exploratory_residual_her.py` adds exploratory residual probes to the V15-baseline HER setup.
- `train_sac_closing_her_residual_lqr.py` trains configurable direct residual SAC around the closing LQR.
- `train_sac_closing_her_lateral_balanced_residual_200k.py` applies the fixed balanced-prefill lateral-progress configuration.
- `train_sac_closing_her_terminal_fade_residual_200k.py` applies the fixed terminal-fade and action-cost configuration.
- `train_sac_closing_her_reference_residual_lqr.py` trains goal-conditioned reference residuals around the LQR.
- `train_sac_closing_reference_residual_lqr_annealed_actor_prior_1hz.py` retains the 1 Hz annealed zero-residual actor-prior configuration.
- `train_sac_closing_reference_residual_lqr_annealed_actor_prior_dt01.py` provides its physically matched 0.1 s configuration.
- `train_sac_closing_reference_residual_lqr_lateral_reward_v2_dt01.py` applies the fixed 0.1 s lateral-progress reward V2 ablation.
- `evaluate_sac_closing_her.py` evaluates goal-conditioned HER policies.
- `evaluate_sac_closing_reference_residual_no_her.py` evaluates fixed-goal reference-residual policies.
- `evaluate_terminal_reference_residual_models.py` compares trained and zero-residual policies on paired terminal resets.
