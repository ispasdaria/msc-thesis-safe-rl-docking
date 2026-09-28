# SAC 3DoF Final Approach

This directory contains the SAC training, tuning, evaluation, and plotting workflows for the final-approach docking phase. Executable paths are resolved from the script location, so commands can be launched from the repository root.

## Retained policies

Two policy campaigns are retained for final comparisons:

- The seed-12 100k policy is `sac_final_approach_portobs_rewardv9_100ksteps_seed12_optuna12` under `models/retained_policies/optuna12_100ksteps_buffer100k`.
- The seed-11 600k policy is `sac_final_approach_portobs_rewardv9_600ksteps_seed11_optuna12_buffer600k` under `models/retained_policies/optuna12_600ksteps_buffer600k`.

Each Stable-Baselines3 model file must be loaded with the VecNormalize file that has the same filename stem. The `.zip` and `_vecnormalize.pkl` files form one policy artifact.

## Directory structure

- Root `evaluate_*.py` files evaluate saved policies or rollout artifacts.
- Root `train_*.py` files train SAC policies.
- `tune_sac_final_optuna.py` performs the Optuna hyperparameter study.
- `observation_space_env.py` implements the observation variants used by the comparison experiment.
- `plotters` contains plotting-only scripts.
- `tests` contains fast interface, path, reproducibility, and artifact checks.
- `logs` contains retained training logs and development logs.
- `models` contains retained policy campaigns and development checkpoints.
- `results` contains retained evaluations, robustness studies, retained figures, and development results.

The `development_experiments` directories preserve unsuccessful, intermediate, and superseded work. These artifacts document the development process and are separate from the retained results.

## Evaluation

The evaluation scripts are separated by purpose:

- General evaluation: `evaluate_sac_final_many_seeds.py`, `evaluate_sac_final_rollouts.py`, and `evaluate_saved_rollout_summary.py`.
- Retained-policy evidence: `evaluate_retained_sac_on_lqr_seeds_0_49.py`, `evaluate_selected_sac_policies_50_rollouts.py`, and `evaluate_selected_sac_robustness_expanded_bounds.py`.
- Training-seed comparison: `evaluate_sac_final_training_seeds.py`.
- Development experiments: `evaluate_observation_space_comparison.py`, `evaluate_sac_final_near_target.py`, and `evaluate_round3_seed19_failures.py`.

The scripts share rollout concepts but produce different experiment-specific artifacts. They remain separate to preserve their established outputs and evaluation criteria.

Evaluate the retained seed-12 policy on the 50 LQR final-approach seeds:

```powershell
.\.venv\Scripts\python.exe src/controllers/SAC-3DoF/final_approach/evaluate_retained_sac_on_lqr_seeds_0_49.py
```

Evaluate selected policies under expanded initial-state bounds:

```powershell
.\.venv\Scripts\python.exe src/controllers/SAC-3DoF/final_approach/evaluate_selected_sac_robustness_expanded_bounds.py --policy all
```

The evaluation environment uses `dt = 1.0 s` and an explicit `max_time = 5,000 s` horizon.

## Plotting

Plotters read committed result or training artifacts and write PNG files. For example:

```powershell
.\.venv\Scripts\python.exe src/controllers/SAC-3DoF/final_approach/plotters/plot_retained_sac_training_results.py
```

See `plotters/README.md` for the input and purpose of each plotter.

## Verification

Run the final-approach test suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest src/controllers/SAC-3DoF/final_approach/tests -q
```
