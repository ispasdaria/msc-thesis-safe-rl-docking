# Pilot H terminal-reset reference-residual SAC model card

Run: `sac_closing_lqr_reference_residual_pilot_h_dt0p1_lateral_curriculum_500k_seed0`  
Status: **complete**  
Selected best checkpoint: **250,000 SAC decisions**

## Experimental change

Pilot H changes Pilot G's replay
prefill and online start-state distribution. Every reset is an empirical state
where lateral position is the worst P3 component. The maximum reset ratio
expands through [1.25, 1.5, 2.0] at training fractions
[0.25, 0.6]. Episodes last at most
600 physical seconds.

The bank contains 1,885 eligible snapshots from
100 source episodes. Snapshot provenance is
success/failure/timeout = 1,110/
0/
775.

The 100,000-transition replay prefill
uses `terminal curriculum`. Replay remains 50% global,
25% ratio <= 5, and 25% ratio <= 2. Model selection and all reported
checkpoint results below use the original full-closing reset.

## Oracle controllability diagnostic

The pre-training paired diagnostic used 20 states. Zero residual succeeded on 10; the finite-grid one-step search succeeded on 9, with 2 rescues and 3 losses. This demonstrates some action authority but does not claim that the myopic oracle is a better controller.

## Controller and exclusions

```text
delta_reference = [
    0.020 * a[0] m/s,
    0.010 * a[1] m/s,
    1.000 * a[2] deg,
]
u = LQR(state, scheduled_P3_reference + delta_reference)
```

SAC and LQR update every 0.1 seconds. Zero SAC action
exactly reproduces LQR. There is no action hold, distance gate, HER, CBF, or safety-filter signal. Full-closing validation remains unchanged.

## Reward

```text
reward = 2.000 * weighted_P3_progress
         + 4.000 * lateral_violation_progress
         + 5.000 * exact_P3_success
         - 25.000 * hard_failure
         - 2.000 * severity_scaled_timeout
         - dt * 0.002 * ||a||^2
         - (1 / dt) * 0.001 * ||a-a_previous||^2
```

`J` equally combines the mean and maximum of the six log-transformed P3
tolerance ratios.

## SAC configuration

| Parameter | Value |
|---|---:|
| Online interactions | 500,000 |
| Full-closing replay prefill | 100,000 |
| Critic-only warm-up | 100,000 |
| Learning rate | 0.0001 |
| Effective actor learning rate | 2.5e-05 |
| Zero-residual actor weight | 1 |
| Batch size | 256 |
| Train frequency / gradient steps | 40 / 1 |
| Discount factor | 0.99999000 |
| Target smoothing | 0.0050 |
| Entropy coefficient | `auto_0.01` |
| Target entropy | -3.0 |

## Full-closing validation

Validation seeds: 100--109.

| Checkpoint | Success | Hard failure | Timeout | Mean minimum position error [m] | Mean best worst ratio | Mean seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 40.0% | 0.0% | 60.0% | 15.006 | 1.133 | 117819.7 |
| 100,000 | 40.0% | 0.0% | 60.0% | 15.006 | 1.133 | 117819.7 |
| 250,000 | 60.0% | 0.0% | 40.0% | 17.800 | 1.131 | 117011.6 |
| 500,000 | 50.0% | 0.0% | 50.0% | 16.530 | 1.145 | 117908.2 |

## Interpretation

Checkpoint 250,000 ranked best on full-closing validation and is stored as `best_model.zip`.

## Artifacts

- `checkpoint_0_model.zip`: exact zero-residual LQR baseline;
- `checkpoint_<N>_model.zip`: trained checkpoints;
- `best_model.zip`: validation-selected controller;
- `final_model.zip`: final trained controller.

Machine-readable configuration and rollout tables are in
`../../../results/lqr_reference_residual_terminal/sac_closing_lqr_reference_residual_pilot_h_dt0p1_lateral_curriculum_500k_seed0/`.
