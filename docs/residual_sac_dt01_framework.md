# Residual SAC framework for the dt=0.1 s closing LQR

All new residual-SAC closing experiments use the sampled-data-verified closing
controller at `dt=0.1 s` (10 Hz). The older 1 Hz residual experiments and their
checkpoints are historical artifacts; they are not valid baselines for new
comparisons and must not be mixed into dt=0.1 s result tables.

## Canonical timing contract

- simulator integration interval: 0.1 s;
- LQR feedback update interval: 0.1 s;
- default SAC residual decision interval: 0.1 s;
- a 12,000 s rollout contains 120,000 simulator/SAC steps;
- `build_Q_closing(dt=0.1)` selects the retuned closing-LQR weights;
- discount factors are converted from a one-second reference as
  `gamma_dt = gamma_1s ** 0.1`;
- persistent per-step rewards are scaled by `0.1 / 1.0`; terminal bonuses and
  penalties remain one-shot events.

These values and conversions live in
`src/controllers/SAC-3DoF/closing/residual_lqr_config.py`. Residual environment
construction through `make_her_env` defaults to 0.1 s and rejects an explicit
1.0 s period. Non-LQR HER environments retain their existing 1.0 s default.

## Supported residual interfaces

The direct action residual executes

```text
u = clip(u_LQR_clipped + scale * gate * a_SAC, -1, 1)
```

and is trained by `train_sac_closing_her_residual_lqr.py`.

The reference residual modifies the LQR's `vx`, `vy`, and attitude references
before the unchanged LQR gain computes the actuator command. Its common parser
and environment configuration are rooted in
`train_sac_closing_reference_residual_lqr_no_her.py`; the stratified and
terminal-reset entry points inherit the same fixed 0.1 s timing contract.

For both interfaces, zero SAC action must exactly reproduce the dt=0.1 s LQR.
Every new experiment configuration records `dt_seconds=0.1`,
`control_frequency_hz=10`, and both SAC/LQR decision intervals.

## Historical compatibility

Saved 1 Hz models remain in their original model/result directories for audit
and thesis provenance. They should be evaluated only with their recorded
historical configuration. New training entry points intentionally do not offer
`dt=1.0` as an option.

## Replay-data audit

The repository contains a 600-rollout held-out campaign for seeds 300--899 at
`closing_dt=0.1 s`, including the canonical 12,000 s campaign and a separate
13,000 s campaign. The main CSV/JSON files contain aggregate rollout metrics,
not every state/action transition. Only a small set of representative full
histories was saved.

The large `lqr_closing_training_rewardv15_*` NPZ archives are legacy 1 Hz
datasets. They also store the normalized LQR actuator command as `action`,
whereas direct residual SAC requires the replay action to be the residual
command (zero for an unmodified LQR transition). They are therefore not valid
prefill sources for the dt=0.1 residual action space.

If cached prefill is needed, regenerate it deterministically with the canonical
controller in `training` mode. Keep seeds 300--899 held out: they are evaluation
rollouts and must not enter training replay. For a first single-seed pilot, the
training entry point's on-demand complete-episode collector is sufficient; a
persistent multi-episode dataset becomes useful for repeated or multi-seed
training.

## Balanced direct-residual lateral pilot

The completed 20,000-step direct-residual pilot retained safety but did not
improve the 4/10 LQR success rate. At 10 Hz, 20,000 steps represent only 2,000 s,
and its 120,000-transition prefill came from one exploratory timeout episode.

The successor preset is
`train_sac_closing_her_residual_lqr_pilot_lateral_balanced_200k.py`. It changes
the experiment contract as follows:

- one accepted pure-LQR prefill episode is required;
- one accepted exploratory-residual prefill episode is required;
- required roles are assigned deterministically before seeded random roles;
- source seeds start at 10, while online training uses seed 0;
- exploratory actions are held for 1.0 s to avoid 10 Hz white-noise probing;
- replay capacity is 500,000, enough for approximately 240,000 prefill plus
  200,000 online transitions;
- the reward uses potential-based six-state progress, explicit lateral P3-set
  progress, and timeout severity based on remaining constraint violation;
- each transition stores its previous achieved goal so HER can recompute the
  progress reward after future-goal relabeling;
- the zero-residual actor prior anneals from 1.0 to 0.1 after critic warm-up.

The old absolute-cost reward and completed 20,000-step artifacts remain valid
historical controls. The new reward is selected explicitly as
`lqr_residual_lateral`; it does not change the validated LQR action law.

## Terminal-fade direct-residual successor

The balanced lateral 200,000-step run remained safe but stayed at 4/10
validation successes. Its actor produced near-boundary residual commands while
the axial-distance gate removed all residual authority inside 100 m, where five
of the six remaining timeouts were lateral-limited.

The successor preset is
`train_sac_closing_her_residual_lqr_pilot_terminal_fade_200k.py`. It keeps the
same source seeds, online seed, replay composition, validation seeds, and
200,000-step interaction budget for a controlled comparison. It changes:

- residual authority is full above a worst P3 ratio of 1.2, smoothly fades
  between ratios 1.2 and 1.0, and is zero only inside the exact success set;
- the critic-visible reward includes residual magnitude and action-rate costs;
- the actor prior anneals from 1.0 to 0.5 rather than 1.0 to 0.1;
- training diagnostics are flushed every 10,000 steps, independent of the
  120,000-step episode length.

Zero residual still reproduces the manifest-validated LQR exactly. The new
interface is selected explicitly as `lqr_residual_terminal_fade`, leaving both
earlier reward versions available for artifact reconstruction.
