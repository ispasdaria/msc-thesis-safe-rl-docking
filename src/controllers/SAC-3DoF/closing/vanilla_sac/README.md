# Vanilla SAC

This family contains closing-phase SAC experiments that do not use demonstrations, residual LQR control, curriculum resets, or specialized replay sampling.

## Experiment groups

- `reward_shaping_trials` contains single-seed reward versions and the retained reward-V15 Optuna trial-16 policy.
- `multiseed` contains independent training runs used to measure seed sensitivity. The phase-balanced runs remain separate and move to the corresponding experiment family.
- `observation_comparison` compares center-of-mass and docking-port-relative observations using the same controller configuration.
- `optuna` contains hyperparameter-search outputs.
- Action-hold presets compare one-second control, fixed two-second holds, and distance-dependent holds.
- The `dt=0.1` scripts reproduce the standard controller with timestep-consistent physical durations.

## Entry points

- `train_sac_closing.py` trains the configurable vanilla baseline.
- `train_sac_closing_dt01.py` trains the configurable 0.1 s V15-dt formulation.
- `train_sac_closing_v15dt_full_budget_dt01.py` applies the fixed full-budget V15-dt preset.
- `evaluate_sac_closing.py` evaluates one configured closing policy.
- `evaluate_sac_closing_multiseed.py` evaluates retained multiseed checkpoints.
- `evaluate_sac_closing_v15_baseline.py` evaluates the fixed pre-Optuna V15 baseline.
- `tune_sac_closing_optuna.py` runs the retained closing-phase hyperparameter search.

## Layout

- Training, evaluation, and tuning entry points are stored in this directory.
- `plotters/` contains figure-generation scripts.
- `tests/` contains the 0.1 s training and evaluation contract checks.
- `models/`, `logs/`, and `results/` use matching experiment-group names.
- `results/retained_figures/horizon_comparison` contains the extended-horizon trajectory comparison.
- `results/terminal_state_convergence` contains the seed-15 state-convergence plot.
- `closing_port_observation_env.py` defines the six-component docking-port observation variant used only by this family.

Run scripts from the repository root, for example:

```powershell
python src/controllers/SAC-3DoF/closing/vanilla_sac/evaluate_sac_closing.py
```

Saved Stable-Baselines3 policies require the corresponding `VecNormalize` state when one was used during training.
