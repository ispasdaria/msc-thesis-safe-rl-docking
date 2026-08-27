# Pilot F terminal-reset reference-residual SAC model card

Run: `sac_closing_lqr_reference_residual_pilot_f_terminal_r2_h600_1hz_50k_seed0`  
Status: **complete**  
Selected best checkpoint: **25,000 SAC decisions**

## Experimental change

Pilot F changes Pilot D's **online episode start distribution**. Training
episodes begin at empirical states recorded from ordinary zero-residual LQR
closing trajectories with worst P3 tolerance ratio in
`[1.05, 2]`. Their horizon
is 600 seconds. This is a fixed terminal
distribution, not a curriculum.

The bank contains 1,885 eligible snapshots from
100 source episodes. Snapshot provenance is
success/failure/timeout = 1,110/
0/
775.

Full LQR trajectories still provide the 60,000
transition replay prefill. Replay remains 50% global, 25% ratio <= 5, and 25%
ratio <= 2. Model selection and all reported checkpoint results below use the
original full closing reset and 12,000-second horizon.

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

SAC and LQR update every second. Zero SAC action exactly reproduces LQR. There
is no action hold, distance gate, HER, CBF, safety-filter signal, or curriculum.
The actor is initialized at zero and retains Pilot D's constant zero-residual
penalty.

## Reward (unchanged)

```text
reward = 1.000 * (J(previous) - gamma * J(next))
         + 1.000 * exact_P3_success
         - 10.000 * hard_failure
         - 1.000 * unsuccessful_timeout
         - 0.010 * ||a||^2
         - 0.010 * ||a-a_previous||^2
```

`J` equally combines the mean and maximum of the six log-transformed P3
tolerance ratios.

## SAC configuration

| Parameter | Value |
|---|---:|
| Online interactions | 50,000 |
| Full-closing replay prefill | 60,000 |
| Critic-only warm-up | 10,000 |
| Learning rate | 0.0001 |
| Effective actor learning rate | 2.5e-05 |
| Zero-residual actor weight | 1 |
| Batch size | 256 |
| Train frequency / gradient steps | 4 / 1 |
| Discount factor | 0.9999 |
| Target smoothing | 0.0050 |
| Entropy coefficient | `auto_0.01` |
| Target entropy | -3.0 |

## Full-closing validation

Validation seeds: 100--109.

| Checkpoint | Success | Hard failure | Timeout | Mean minimum position error [m] | Mean best worst ratio | Mean seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 80.0% | 0.0% | 20.0% | 16.531 | 1.005 | 11361.6 |
| 10,000 | 80.0% | 0.0% | 20.0% | 16.531 | 1.005 | 11361.6 |
| 25,000 | 90.0% | 0.0% | 10.0% | 17.321 | 0.998 | 11294.4 |
| 50,000 | 90.0% | 0.0% | 10.0% | 18.342 | 1.004 | 11276.7 |

## Interpretation

Checkpoint 25,000 ranked best on full-closing validation and is stored as `best_model.zip`.

## Held-out results and recommendation

On full-closing testing seeds 300--399, exact zero-residual LQR succeeded on
75/100 episodes and this selected checkpoint succeeded on 72/100. Paired
changes were one rescue and four losses (two-sided exact p = 0.375). Neither
controller had a hard failure.

On 50 held-out empirical terminal resets with the 600 s training horizon, LQR
succeeded on 18/50 and this checkpoint on 19/50, with two rescues and one loss
(two-sided exact p = 1.0).

**Recommendation: retain the zero-residual LQR.** The validation improvement
did not generalize, and terminal-task improvement was too small and uncertain
to justify using the learned residual. Detailed analysis is in
`../../../results/lqr_reference_residual_terminal/sac_closing_lqr_reference_residual_pilot_f_terminal_r2_h600_1hz_50k_seed0/EXPERIMENT_ANALYSIS.md`.

## Artifacts

- `checkpoint_0_model.zip`: exact zero-residual LQR baseline;
- `checkpoint_<N>_model.zip`: trained checkpoints;
- `best_model.zip`: validation-selected controller;
- `final_model.zip`: final trained controller.

Machine-readable configuration and rollout tables are in
`../../../results/lqr_reference_residual_terminal/sac_closing_lqr_reference_residual_pilot_f_terminal_r2_h600_1hz_50k_seed0/`.
