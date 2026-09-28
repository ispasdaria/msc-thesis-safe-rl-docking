# Vanilla SAC

This family contains closing-phase SAC experiments that do not use demonstrations, residual LQR control, curriculum resets, or specialized replay sampling.

## Experiment groups

- `reward_shaping_trials` contains single-seed reward versions and the retained reward-V15 Optuna trial-16 policy.
- `multiseed` contains independent training runs used to measure seed sensitivity. The phase-balanced runs remain separate and move to the corresponding experiment family.
- `observation_comparison` compares center-of-mass and docking-port-relative observations using the same controller configuration.
- `optuna` contains hyperparameter-search outputs.
- Action-hold presets compare one-second control, fixed two-second holds, and distance-dependent holds.
- The `dt=0.1` scripts reproduce the standard controller with timestep-consistent physical durations.

## Layout

- Training, evaluation, and tuning entry points are stored in this directory.
- `plotters/` contains figure-generation scripts.
- `models/`, `logs/`, and `results/` use matching experiment-group names.
- `results/retained_figures/horizon_comparison` contains the extended-horizon trajectory comparison.
- `results/terminal_state_convergence` contains the seed-15 state-convergence plot.

Run scripts from the repository root, for example:

```powershell
python src/controllers/SAC-3DoF/closing/vanilla_sac/evaluate_sac_closing.py
```

Saved Stable-Baselines3 policies require the corresponding `VecNormalize` state when one was used during training.
