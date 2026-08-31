# Pilot I terminal-reset reference-residual SAC model card

Run: `sac_closing_lqr_reference_residual_pilot_i_dt0p1_terminal_fade_500k_seed0`  
Status: **in progress**  
Selected best checkpoint: **0 SAC decisions**

## Experimental change

Pilot I retains Pilot H's 10 Hz
lateral-limited reset curriculum, reward, replay, and SAC configuration. It
adds a terminal residual-authority fade based on the worst simultaneous P3
tolerance ratio: full authority at ratio >=
1.2, linear fade through the
terminal band, and exact zero residual at ratio <=
1. The established LQR
therefore performs the final tolerance crossing.

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
g(r) = clip((r - 1.000) /
            (1.200 - 1.000), 0, 1)
delta_reference = g(max_P3_ratio) * [
    0.020 * a[0] m/s,
    0.010 * a[1] m/s,
    1.000 * a[2] deg,
]
u = LQR(state, scheduled_P3_reference + delta_reference)
```

SAC and LQR update every 0.1 seconds. Zero SAC action
exactly reproduces LQR. There is no action hold, distance gate, HER, CBF, or safety-filter signal. Full-closing validation uses the same residual fade.

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

## Interpretation

Training is in progress; selection is provisional.

## Artifacts

- `checkpoint_0_model.zip`: exact zero-residual LQR baseline;
- `checkpoint_<N>_model.zip`: trained checkpoints;
- `best_model.zip`: validation-selected controller;
- `final_model.zip`: final trained controller.

Machine-readable configuration and rollout tables are in
`../../../results/lqr_reference_residual_terminal/sac_closing_lqr_reference_residual_pilot_i_dt0p1_terminal_fade_500k_seed0/`.
