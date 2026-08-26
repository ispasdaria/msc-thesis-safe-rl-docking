# Closing SAC source audit for CBF-aware fine-tuning

## Decision

Use the non-LQR guarded phase-balanced checkpoint at 100,000 additional steps:

```text
src/controllers/SAC-3DoF/closing/models/multiseed/
  sac_closing_v15_trial16_terminal_phase_balanced_guarded600k_seed0/
  seed_0/model_0100000_steps.zip
```

with its matching normalization state:

```text
src/controllers/SAC-3DoF/closing/models/multiseed/
  sac_closing_v15_trial16_terminal_phase_balanced_guarded600k_seed0/
  seed_0/model_0100000_steps_vecnormalize.pkl
```

This is the strongest directly compatible closing SAC checkpoint after
excluding every experiment that used LQR actions, LQR transitions, behavior
cloning, or an LQR-derived constraint.

## Audit scope

The repository contains 1,487 closing-model/statistics artifacts. The source
selection inspected the recorded task evaluations rather than ranking model
filenames or training returns:

- 59 `checkpoint_metrics.csv` files containing 515 checkpoint rows
- 127 rows of independent `per_policy_summary.csv` results
- 14 curriculum evaluation files containing 206 rows
- the Optuna P3-reach evaluation table
- every recorded closing CBF evaluation manifest and aggregate table
- the HER/residual results, which were retained for comparison but are not a
  drop-in standard-SAC source

Episode return was not compared across reward versions because those values
do not share a common scale. The ranking used full-P2 closing evaluations and
prioritized:

1. P3 success
2. absence of hard safety failures
3. low final simultaneous six-component target ratio
4. low minimum and final P3 position error
5. direct compatibility with the standard 12-observation SAC/VecNormalize
   continuation pipeline

Curriculum `current_stage` success rates were not treated as full-task
success. Their recorded `full_test_probe` evaluations failed from the normal
P2 distribution.

## Relevant non-LQR comparison

| Candidate | Evaluation | Failure | Timeout | Best/final max target ratio | Minimum P3 position error | Final P3 position error |
|---|---:|---:|---:|---:|---:|---:|
| **Selected guarded 100k checkpoint** | seeds 200--209 | 0% | 100% | **6.230 final** | 18.459 m | **38.989 m** |
| Selected checkpoint | validation seeds 100--109 | 0% | 100% | **4.659 final** | 7.808 m | **24.658 m** |
| Guarded 600k run's saved `best_model` | seeds 200--209 | 0% | 100% | 15.46 final | **4.970 m** | 44.72 m |
| V2.1 best model | seeds 200--209 | 0% | 100% | 5.85 best / 38.09 final | 89.89 m | 119.5 m |
| Original trial-16 400k | seeds 200--209 | 0% | 100% | 30.30 final | 42.39 m | 66.60 m |
| Vanilla V15 multiseed seed-0 300k | seeds 1000--1099 | 0% | 100% | 12.91 final | 100.5 m | 102.0 m |
| P3-reach V12 best | seeds 200--209 | 0% | 100% | 22.54 final | 219.75 m | 225.53 m |
| HER V4 residual best | seeds 200--209 | 0% | 100% | 18.10 best / 30.30 final | 42.39 m | 66.60 m |

The guarded run's saved `best_model` was selected by closest position before
the newer simultaneous-target diagnostics existed. It flies closer to P3 but
finishes with a substantially worse six-component target ratio. The 100k
checkpoint is therefore the better continuation source: it retains safe
approach behavior and ends much nearer the complete handoff set.

## CBF compatibility check

A no-write rollout of the selected checkpoint on external seed 200 with the
hard `all` filter produced:

- 12,000 steps, timeout, and no safety failure
- zero material CBF interventions
- zero QP failures
- final maximum target ratio: 5.319
- final/minimum position error: 44.867 m
- maximum normalized CBF correction norm: `1.25e-9`

The filter therefore leaves the established deterministic behavior unchanged
on this checked trajectory. During SAC fine-tuning it can still constrain the
stochastic exploratory actions that depart from this behavior.

## Provenance

The selected policy was initialized from the vanilla V15 trial-16 checkpoint
and then continued with online phase-balanced replay. Its experiment config
explicitly records:

```text
lqr_policy_or_transitions_used: false
cbf_filter_used_during_training: false
training_environment: standard Docking3DoFGymEnv closing V15
```

The CBF-aware pilot preserves this checkpoint as an immutable source and uses
the same matching frozen VecNormalize statistics.

## CBF-aware pilot outcome

The seed-0 pilot completed 50,000 additional gradient steps with the hard
`all` CBF in both training and held-out validation. All four checkpoints had
zero safety failures and zero CBF QP failures, but terminal closing quality
regressed as fine-tuning progressed:

| Additional steps | Final max target ratio | Final P3 error | Best max target ratio | CBF intervention fraction |
|---:|---:|---:|---:|---:|
| **0 (selected)** | **4.657** | **25.17 m** | **2.978** | 2.84% |
| 10,000 | 6.775 | 76.10 m | 4.653 | 4.74% |
| 25,000 | 9.301 | 100.57 m | 6.918 | 5.29% |
| 50,000 | 11.644 | 178.78 m | 10.857 | 6.34% |

The checkpoint selector therefore retained additional step 0. This trial
supports deploying the hard filter with the existing guarded 100k policy, but
does not support further training with the current intervention penalty and
phase-balanced replay configuration. The trained 10k, 25k, and 50k files are
diagnostic checkpoints rather than upgrades.
