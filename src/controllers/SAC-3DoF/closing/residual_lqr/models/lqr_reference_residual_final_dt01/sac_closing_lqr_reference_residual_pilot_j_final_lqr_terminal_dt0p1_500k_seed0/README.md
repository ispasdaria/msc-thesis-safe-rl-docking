# Pilot J terminal-reset reference-residual SAC model card

Run: `sac_closing_lqr_reference_residual_pilot_j_final_lqr_terminal_dt0p1_500k_seed0`  
Status: **complete**  
Selected best checkpoint: **0 SAC decisions**

## Experimental change

Pilot J uses a terminal
residual-authority fade based on the worst simultaneous P3 tolerance ratio:
full authority at ratio >=
1.2, linear fade through the
terminal band, and exact zero residual at ratio <=
1. The established LQR
therefore performs the final tolerance crossing.

The bank contains 3,958 eligible snapshots from
17 source episodes. Snapshot provenance is
success/failure/timeout = 1,400/
0/
2,558.

The 100,000-transition replay prefill
uses `terminal curriculum`. Replay remains 50% global,
25% ratio <= 5, and 25% ratio <= 2. Model selection and all reported
checkpoint results below use the original full-closing reset.

## Oracle controllability diagnostic

No oracle summary was available when the run began.

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
| Replay prefill transitions | 100,000 |
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
| 250,000 | 40.0% | 0.0% | 60.0% | 15.701 | 1.140 | 118014.5 |
| 500,000 | 30.0% | 10.0% | 60.0% | 153.574 | 8.067 | 106299.9 |

## Interpretation

No trained checkpoint outranked exact zero-residual LQR on the full-closing validation seeds; `best_model.zip` remains baseline.

## Independent acceptance

The trained candidate did not pass the independent success and safety gate. `accepted_model.zip` is therefore checkpoint 0, the exact final LQR baseline.

## Artifacts

- `checkpoint_0_model.zip`: exact zero-residual LQR baseline;
- `checkpoint_<N>_model.zip`: trained checkpoints;
- `best_model.zip`: validation-selected controller;
- `accepted_model.zip`: controller selected by the independent acceptance gate;
- `final_model.zip`: final trained controller.

Machine-readable configuration and rollout tables are in
`../../../results/lqr_reference_residual_final_dt01/sac_closing_lqr_reference_residual_pilot_j_final_lqr_terminal_dt0p1_500k_seed0/`.
