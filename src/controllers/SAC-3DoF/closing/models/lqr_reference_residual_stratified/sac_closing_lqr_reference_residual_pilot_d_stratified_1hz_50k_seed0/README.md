# Pilot D P3-stratified reference-residual SAC model card

Run: `sac_closing_lqr_reference_residual_pilot_d_stratified_1hz_50k_seed0`  
Status: **complete**  
Selected best checkpoint: **25,000 SAC decisions**

## What this experiment changes

Pilot D changes only Pilot C's replay sampling distribution. The controller,
fixed P3 goal, reward, one-second SAC/LQR cadence, zero-residual actor prior,
and no-HER setup are unchanged. No CBF is used during training.

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

Final replay audit: Global/outer/inner stored: 110,000/18,629/3,854. Realized sampled fractions: 50.0%/25.0%/25.0%.

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
| 25,000 | 80.0% | 0.0% | 20.0% | 16.422 | 1.002 | 11354.6 |
| 50,000 | 80.0% | 0.0% | 20.0% | 16.480 | 1.005 | 11369.0 |

## Interpretation

Checkpoint 25,000 ranked best on the validation seeds and is stored as `best_model.zip`.

## Initial held-out check

Seeds 200--219 were evaluated only after checkpoint selection.

| Controller | Success | Hard failure | Timeout | Mean best worst ratio | Mean final worst ratio |
|---|---:|---:|---:|---:|---:|
| Exact zero-residual LQR | 14/20 (70%) | 0/20 | 6/20 | 1.0392 | 1.1564 |
| Pilot D best (25k) | 15/20 (75%) | 0/20 | 5/20 | 1.0349 | 1.1371 |

Pilot D rescued seed 219 (`best worst ratio 1.0020 -> 0.9981`) and did not
lose a seed on this set. This is a promising one-seed gain, not strong
statistical evidence: there is only one discordant pair, so a larger held-out
set is required before claiming reliable superiority to LQR.

## Artifacts

- `checkpoint_0_model.zip`: exact zero-residual LQR baseline;
- `checkpoint_<N>_model.zip`: evaluated SAC checkpoints;
- `best_model.zip`: validation-ranked checkpoint;
- `final_model.zip`: end-of-training policy.

Configuration, rollout tables, and `replay_statistics.json` are in
`../../../results/lqr_reference_residual_stratified/sac_closing_lqr_reference_residual_pilot_d_stratified_1hz_50k_seed0/`.
