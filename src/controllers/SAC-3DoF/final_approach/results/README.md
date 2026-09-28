# Evaluation and Plotting Results

- `retained_policy_evaluations` contains direct evaluations, policy-selection records, reconstructed 50-rollout datasets, and terminal-state convergence data for retained candidates.
- `robustness_evaluations` contains tests outside or beyond the nominal evaluation bounds.
- `retained_figures` contains stored figures and their source data used for reporting.
- `development_experiments` contains intermediate reward, observation, curriculum, replay-buffer, and diagnostic results.

Absolute paths embedded in archived JSON files record the workstation location used by the original run. Executable scripts resolve current repository paths independently and do not use those archived absolute paths as configuration.
