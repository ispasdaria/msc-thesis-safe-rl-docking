# Trained Models

Each SAC model requires the VecNormalize file with the same filename stem. Model and normalization files must be copied and loaded as a pair unless a family README explicitly identifies an archival or incomplete checkpoint.

## Retained policy campaigns

- [`optuna12_100ksteps_buffer100k`](retained_policies/optuna12_100ksteps_buffer100k/README.md) contains the retained round-1 seed-12 policy and the associated 20-seed campaign.
- [`optuna12_600ksteps_buffer600k`](retained_policies/optuna12_600ksteps_buffer600k/README.md) contains the retained round-5 seed-11 policy and the associated 20-seed campaign.

## Development experiment families

- [`com_relative_states_first_input`](development_experiments/com_relative_states_first_input/README.md) records early center-of-mass observation and reward development.
- [`curriculum_experiment`](development_experiments/curriculum_experiment/README.md) contains near-to-far curriculum policies.
- [`near_target_initialization`](development_experiments/near_target_initialization/README.md) contains near-field initialization and warm-start experiments.
- [`observation_comparison`](development_experiments/observation_comparison/README.md) compares center-of-mass and combined observation contracts.
- [`optuna12_200ksteps_buffer200k`](development_experiments/optuna12_200ksteps_buffer200k/README.md) contains the 200k training-budget experiment.
- [`optuna12_300ksteps_buffer100k`](development_experiments/optuna12_300ksteps_buffer100k/README.md) contains the 300k-step, 100k-buffer campaign.
- [`optuna12_300ksteps_buffer300k`](development_experiments/optuna12_300ksteps_buffer300k/README.md) contains the 300k-step, 300k-buffer campaign.
- [`optuna12_600ksteps_buffer100k`](development_experiments/optuna12_600ksteps_buffer100k/README.md) contains the 600k-step, 100k-buffer campaign.
- [`reward_shaping_trials`](development_experiments/reward_shaping_trials/README.md) contains docking-port reward versions 1 through 10.

Other checkpoints within retained campaign directories are preserved because they support policy-selection comparisons.
