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

## Shared support modules

- `train_sac_closing_multiseed.py` provides common training configuration, callbacks, metric writers, and checkpoint selection used by the vanilla, curriculum, phase-balanced, demonstration-learning, and timestep-transfer families.
- `near_p3_reset_wrapper.py` provides the fixed near-P3 reset distribution used by curriculum, phase-balanced, and timestep-transfer experiments.
- `behavior_cloned_warmup_sac.py` provides the behavior-cloning warm-up implementation used by demonstration-learning and residual-LQR experiments.

Each family README documents its entry points, experiment variants, and artifacts. Family-specific tests and support modules are stored inside the corresponding family directory.
