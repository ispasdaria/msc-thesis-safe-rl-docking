# Contributing and Git Workflow

## Branching strategy

The `main` branch stores the stable version of the project. New work should be developed in separate feature branches created from `main`.

Branch names should follow these conventions:

- `feature/<task-name>` for new functionality
- `fix/<issue-name>` for bug fixes
- `docs/<topic>` for documentation updates
- `testing/<topic>` for experimental work on study cases

Examples:
- `feature/project-structure`
- `feature/3dof-simulator`
- `feature/lqr-baseline`
- `fix/reset-function-found-bug`
- `docs/readme-update`

## Development workflow

1. Start from the latest version of `main`.
2. Create a new branch for the task.
3. Implement and test the changes in that branch.
4. Commit changes with clear and concise commit messages.
5. Open a pull request to merge the branch into `main`.
6. Merge only after the changes are reviewed and the branch is in a stable condition.

## Pull requests

Pull requests are used to keep a clear history of development versions and completed tasks. Each pull request should correspond to one main task or one logically grouped set of changes.

Pull request titles should clearly describe the implemented change.

Examples:
- `Add initial project folder structure`
- `Implement 3-DoF simulator skeleton`
- `Add LQR baseline controller`

## Commit message conventions

Commit messages should be short, clear, and action-oriented.

Examples:
- `Create initial repository structure`
- `Add simulator reset function`
- `Implement state propagation`
- `Update README with setup instructions`

## General rules

- Keep `main` as the stable branch.
- Do not commit incomplete experimental code directly to `main`.
- Use pull requests to document the development history.
- Keep branches focused on one task whenever possible.

## Environment 

This project is developed using Python 3.12.2.

## Repository structure

- `configs/` — Configuration files for simulation settings, controller parameters, reward weights, and training hyperparameters.
- `docs/` — Project notes, implementation documentation, and supporting written material for the repository.
- `src/` — Main source code of the project.
- `src/envs/` — Simulation environments for the docking problem, including the 3-DoF model and later the 6-DoF extension.
- `src/controllers/` — Controller implementations, including the LQR baseline, SAC controller, and CBF-based safety filter.
- `src/utils/` — Shared utility functions such as configuration loading, logging helpers, state conversions, and plotting support.
- `src/evaluation/` — Code for performance metrics, controller comparisons, Monte Carlo runs, and experiment evaluation.
- `scripts/` — Runnable scripts for training, evaluation, verification checks, and experiment execution.
- `tests/` — Unit tests, integration checks, and verification scripts for the simulator, controllers, and safety filter.
- `logs/` — Stored logs from simulations, training runs, and safety-filter interventions.
- `models/` — Saved trained models and other reusable serialized outputs.
- `results/` — Numerical experiment outputs, processed metrics, and saved trajectories.
- `figures/` — Generated plots and figures used for analysis and inclusion in the thesis.

## Coding conventions

- Use 4 spaces for indentation.
- Use `snake_case` for variables, functions, and file names.
- Use `PascalCase` for class names.
- Use `UPPER_CASE` for constants.
- Keep file names short, descriptive, and focused on one clear purpose.
- Prefer meaningful variable and function names over short or unclear names.
- Add concise docstrings to important functions and classes.
- Use comments only when they improve clarity, especially for assumptions, formulas, or non-obvious logic.
- Group imports into standard library imports, third-party imports, and local project imports.
- Avoid hardcoding important parameters such as simulation settings, reward weights, controller gains, and training hyperparameters; store them in configuration files or dedicated parameter definitions instead.
- Keep functions reasonably short and focused on a single task whenever possible.
- Use a consistent code layout and structure across the repository.

### General naming conventions

Examples of preferred naming styles:

- Function name: `compute_reward()`
- Variable name: `state_vector`
- Constant name: `MAX_EPISODE_STEPS`
- Class name: `DockingEnv`
- File name: `lqr_controller.py`

Examples of file names to avoid:

- `stuff.py`
- `test2.py`
- `new_code.py`

### Run and saved-file naming conventions

- Use lowercase letters and underscores for run names and saved file names.
- Include the main method, scenario, and purpose of the file whenever relevant.
- Avoid vague names such as `final_run`, `test_new`, or `output1`.
- Prefer names that make the content understandable without opening the file.
- Use dates only when they add useful version information.
- Keep naming consistent across logs, saved models, result tables, and figures.

Examples of preferred run names:

- `lqr_nominal_case`
- `sac_baseline_training`
- `sac_cbf_eval_case1`
- `sac_warmstart_training`
- `monte_carlo_nominal`

Examples of preferred saved file names:

- `lqr_nominal_trajectory.csv`
- `sac_baseline_metrics.json`
- `sac_cbf_model.pt`
- `reward_training_curve.png`
- `monte_carlo_summary.csv`

A recommended file naming pattern is:

`method_scenario_content`

For example:

- `sac_nominal_metrics.csv`
- `sac_cbf_episode_log.csv`
- `lqr_docking_trajectory.csv`

If multiple versions of the same file are needed, use a clear suffix such as:

- `sac_nominal_metrics_v2.csv`
- `sac_cbf_model_2026_03_27.pt`