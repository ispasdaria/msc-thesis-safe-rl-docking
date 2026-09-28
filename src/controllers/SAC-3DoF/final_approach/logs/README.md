# Training Logs

The log tree separates training records associated with retained policies from development experiments.

- `retained_policy_training` contains the campaigns that produced the retained seed-12 100k policy and seed-11 600k policy.
- `development_experiments` contains reward, observation, curriculum, initialization, replay-buffer, and intermediate Optuna investigations.

Logs record training progress and do not replace the corresponding model and VecNormalize files.
