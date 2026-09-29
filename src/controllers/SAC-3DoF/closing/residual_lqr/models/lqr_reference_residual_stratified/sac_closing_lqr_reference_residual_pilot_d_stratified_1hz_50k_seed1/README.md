# Pilot D P3-stratified reference-residual SAC model card

Run: `sac_closing_lqr_reference_residual_pilot_d_stratified_1hz_50k_seed1`  
Status: **complete**  
Selected best checkpoint: **0 SAC decisions**

## What this experiment changes

Pilot D changes only Pilot C's replay sampling
distribution. The controller, fixed P3 goal, reward, one-second SAC/LQR
cadence, zero-residual actor prior, and no-HER setup are unchanged.

No CBF is used during training.

Each batch is constructed from:

- **50% uniform:** all stored closing transitions;
- **25% near-P3 pool:**
  `worst ratio <= 5`;
- **25% near-P3 inner band:**
  `worst ratio <= 2`.

The targeted pools are deliberately nested: every `ratio <= 2` transition is
eligible for both targeted quarters, giving the closest states extra replay
weight exactly as implied by the two requested thresholds. Uniform replay also
samples from the complete buffer and may naturally include near-P3 transitions.
If a targeted ring is empty early in collection, its missing batch slots fall
back to uniform replay.

Final replay audit: Global/outer/inner stored: 110,000/16,623/3,160. Realized sampled fractions: 50.0%/25.0%/25.0%.

## Controller

```text
delta_reference = [
    0.020 * a[0] m/s,
    0.010 * a[1] m/s,
    1.000 * a[2] deg,
]
u_nominal = LQR(state, scheduled_P3_reference + delta_reference)
```

Zero SAC action exactly reproduces the original LQR. There is no action hold,
distance gate, CBF signal, HER relabeling, or safety filtering in training.

## Reward

```text
reward = 1.000 * (J(previous) - gamma * J(next))
         + 1.000 * exact_P3_success
         - 10.000 * hard_failure
         - 1.000 * unsuccessful_timeout
         - 0.010 * ||a||^2
         - 0.010 * ||a - a_previous||^2
```

`J` equally combines the mean and maximum of the six log-transformed,
normalized P3 tolerance ratios. The replay strata use the maximum untransformed
ratio from the actual next state and actual fixed P3 target.

## SAC configuration

| Parameter | Value |
|---|---:|
| Online interactions | 50,000 |
| Replay prefill | 60,000 |
| Global/outer/inner capacities | 300,000/100,000/100,000 |
| Critic-only warm-up | 10,000 interactions |
| Learning rate | 0.0001 |
| Effective actor learning rate | 2.5e-05 |
| Actor zero-reference weight | 1 |
| Train frequency / gradient steps | 4 / 1 |
| Batch size | 256 |
| Discount factor | 0.9999 |
| Target smoothing | 0.0050 |
| Entropy coefficient | `auto_0.01` |
| Target entropy | -3.0 |

## Validation results

Validation seeds: 100--109.

| Checkpoint | Success | Hard failure | Timeout | Mean minimum position error [m] | Mean best worst ratio | Mean seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 80.0% | 0.0% | 20.0% | 16.531 | 1.005 | 11361.6 |
| 10,000 | 80.0% | 0.0% | 20.0% | 16.531 | 1.005 | 11361.6 |
| 25,000 | 80.0% | 0.0% | 20.0% | 15.648 | 1.005 | 11383.1 |
| 50,000 | 80.0% | 0.0% | 20.0% | 16.650 | 1.007 | 11369.6 |

## Interpretation

No trained checkpoint outranked the exact zero-residual LQR baseline on the validation seeds.

## Held-out interpretation

Because validation selected checkpoint 0, `best_model.zip` is exactly the
zero-residual LQR policy. Its held-out behavior is therefore the already
verified LQR result: 14/20 successes on seeds 200--219, 75/100 on seeds
300--399, and 89/120 combined, with zero hard failures. These identical
rollouts were not recomputed.

This replication did not produce a learned residual policy that passed the
validation ranking. Checkpoint 25k had mean best worst ratio 1.00525 and
checkpoint 50k had 1.00745, both worse than checkpoint 0 at 1.00473.

## Artifacts

- `checkpoint_0_model.zip`: exact zero-residual LQR baseline;
- `checkpoint_<N>_model.zip`: evaluated SAC checkpoints;
- `best_model.zip`: validation-ranked checkpoint;
- `final_model.zip`: end-of-training policy.

Configuration, rollout tables, and `replay_statistics.json` are in
`../../../results/lqr_reference_residual_stratified/sac_closing_lqr_reference_residual_pilot_d_stratified_1hz_50k_seed1/`.
