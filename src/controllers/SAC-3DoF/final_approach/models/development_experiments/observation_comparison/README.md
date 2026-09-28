# Observation-Space Comparison

This family compares two additional final-approach observation contracts:

- `com_observation` uses center-of-mass relative states.
- `combined_observation` combines center-of-mass and docking-port information.

The directory contains final and best checkpoints for both variants. Each model has a matching VecNormalize file. The original docking-port observation controller is stored in the retained policy family and serves as the comparison baseline.

Training is implemented by `train_sac_final_observation_comparison.py`, and evaluation is implemented by `evaluate_observation_space_comparison.py`.
