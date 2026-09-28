# SAC Closing Experiments

This directory contains SAC controllers and experiments for the closing phase of the 3-DoF docking scenario. Experiments are organized by method so that training scripts, evaluations, stored artifacts, tests, and plotting code can be traced together.

## Experiment families

- `vanilla_sac` - standard SAC baselines, reward shaping, observation comparisons, multiseed studies, action-hold variants, and Optuna tuning.
- `curriculum_and_p3_reach` - curriculum learning, near-P3 resets, braking studies, and P3-reach experiments.
- `phase_balanced_replay` - phase-balanced replay, guarded training, and CBF variants.
- `lqr_demonstration_learning` - LQR demonstrations, replay prefill, behavior cloning, and mixed replay.
- `residual_lqr` - direct and reference-residual SAC, HER variants, terminal pilots, and baseline validation.
- `timestep_transfer` - comparisons and training across 1 Hz and 10 Hz execution.
- `two_stage_capture` - systematic and learned two-stage capture experiments.

Python modules that support more than one family remain in this directory. Each family README documents its scripts, experiment variants, and artifacts.
