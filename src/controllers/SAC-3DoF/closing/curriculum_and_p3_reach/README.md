# Curriculum and P3-Reach Experiments

This family contains closing-phase SAC experiments that modify reset distributions, reward shaping, safety termination, or control timing to reach the complete P3 handover set. Names retain the reward version and state the implemented distinction.

## Experiment groups

- `reset_distribution_curriculum` - staged reset distributions that expand from near-P3 states toward the full P2 closing distribution, including gamma and bootstrap comparisons.
- `nonterminal_safety_continuation` - reward-V15 and P3-reach-V1 continuations from the retained Trial-16 policy with safety violations penalized but nonterminal.
- `from_scratch_reward_progression` - fresh SAC training with P3-reach reward versions V1-V19 and nonterminal safety handling.
- `safety_terminal_braking_experiments` - V19-V21 from-scratch experiments with hard safety termination and LQR-inspired braking or port-level tracking envelopes.
- `ten_hz_adaptive_terminal_continuation` - V15-dt continuation at `dt = 0.1 s` with distance-dependent action cadence and terminal-focused replay warm-up.
- `p3_reach_v12_optuna` - V12 hyperparameter optimization with freshly initialized policies.

Each group has a detailed README under `results/`. Matching model and log directories use the same run names.

## Reward progression

| Version | Implemented distinction |
|---|---|
| V1 | Terminal-error shaping with nonterminal safety violations |
| V2 | Physically scaled terminal-capture shaping |
| V3 | Global position, velocity, attitude, and rate progress |
| V4 | Log-scaled component balancing |
| V5 | Distance-gated terminal-velocity capture |
| V6 | Tolerance-normalized smooth worst-component objective |
| V7 | Balanced mean and smooth worst-component objective |
| V8 | Symmetric goal progress with unsafe-state occupancy cost |
| V9 | Global guidance with gated terminal-velocity matching |
| V10 | Dense terminal capture and departure deterrence |
| V11 | Bounded near-target capture progress |
| V12 | Balanced capture progress and failed-approach resets |
| V13 | Worst-unsatisfied-component capture progress |
| V14 | Alignment and angular-rate soft constraint |
| V15 | Lateral docking-cone occupancy constraint |
| V16 | Lateral docking-cone progress shaping |
| V17 | Cone-feasibility gating of positive task progress |
| V18 | Position- and velocity-weighted capture progress |
| V19 | LQR-inspired axial braking envelope |
| V20 | V19 with calibrated terminal success and failure rewards |
| V21 | Coupled docking-port velocity and alignment envelope |

## Entry points

- `train_p3_reach_reset_distribution_curriculum.py` trains staged reset curricula.
- `train_p3_reach_nonterminal_safety_continuation.py` continues a retained policy without terminating on safety violations.
- `train_p3_reach_reward_from_scratch.py` provides the common from-scratch training pipeline.
- `train_p3_reach_v19_lqr_axial_braking_envelope.py`, `train_p3_reach_v20_calibrated_safety_terminal.py`, and `train_p3_reach_v21_port_velocity_alignment_envelope.py` select distinct safety-terminal environments.
- `train_p3_reach_v15dt_ten_hz_adaptive_cadence.py` runs the ten-hertz continuation experiment.
- `tune_p3_reach_v12_from_scratch_optuna.py` runs the V12 Optuna study.
- `evaluate_p3_reach_checkpoints.py` evaluates stored from-scratch checkpoints on selected seeds.

The launchers for V19-V21 are retained because they select different environment classes, configuration metadata, and checkpoint-ranking rules. Shared source checkpoints remain in their owning vanilla or phase-balanced families.

## Artifact layout

- `models/` contains Stable-Baselines3 checkpoints and associated normalization states.
- `logs/` contains TensorBoard event files and tuning logs.
- `results/` contains configurations, checkpoint metrics, rollout diagnostics, and retained figures.
- `plotters/` contains figure-generation entry points tied to named experiments.
- `tests/` verifies evaluation diagnostics and the ten-hertz continuation contract.

Run entry points from the repository root. Paths are derived from each script location.
