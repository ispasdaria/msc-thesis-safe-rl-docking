# Pilot E P3-stratified reference-residual SAC model card

Run: `sac_closing_lqr_reference_residual_pilot_e_actor_prior_annealed_1hz_50k_seed0`  
Status: **complete**  
Selected best checkpoint: **25,000 SAC decisions**

## What this experiment changes

Pilot E changes exactly one Pilot D setting: the zero-residual actor penalty
is annealed linearly from 1 at interaction 10,000 to 0.1 at interaction
30,000, then held constant. The controller, reward, replay distribution,
seeds, initialization, and all other SAC settings are unchanged.

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

Final replay audit: Global/outer/inner stored: 110,000/18,735/3,511. Realized sampled fractions: 50.0%/25.0%/25.0%.

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
| Actor zero-reference weight | 1 at 10,000; linear to 0.1 at 30,000 |
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
| 25,000 | 80.0% | 0.0% | 20.0% | 15.939 | 1.005 | 11360.0 |
| 50,000 | 70.0% | 0.0% | 30.0% | 15.429 | 1.017 | 11406.8 |

## Interpretation

Checkpoint 25,000 ranked best on the validation seeds and is stored as `best_model.zip`.

The 50,000-step policy dropped from 8/10 to 7/10 validation successes after
the prior had remained at 0.1 for 20,000 interactions. This is evidence that
the weaker constraint allowed harmful actor drift, so `final_model.zip` should
not be used as the selected controller.

## Held-out evaluation

All policies were evaluated deterministically with no CBF. Seeds 200--219 and
300--399 were not used for checkpoint selection.

| Seed set | Exact LQR | Pilot D best | Pilot E best | Pilot E hard failures |
|---|---:|---:|---:|---:|
| 200--219 | 14/20 | 15/20 | 14/20 | 0 |
| 300--399 | 75/100 | 75/100 | 76/100 | 0 |
| Combined held-out | 89/120 | 90/120 | 90/120 | 0 |

On seeds 300--399, Pilot E rescued LQR seeds 355, 367, and 371, but lost LQR
seeds 302 and 358. Across all 120 seeds it rescued four LQR timeouts and lost
three LQR successes. The exact paired McNemar test is `p = 1.0`; the one-net
success gain is not evidence of a reliable improvement.

Pilot E and Pilot D are exactly tied at 90/120 but disagree on six seeds: Pilot
E rescues 333, 355, and 367 relative to Pilot D while losing 208, 302, and 358.
Their paired McNemar result is also `p = 1.0`.

On the 100-seed set, Pilot E's mean best worst ratio is 1.0472 versus 1.0463
for LQR and 1.0454 for Pilot D. Its mean final velocity error is also higher
(0.00920 versus 0.00879 for LQR). These secondary results do not support the
actor-annealing hypothesis. Pilot D remains the more conservative residual-SAC
candidate, but neither residual policy has demonstrated superiority to LQR.

## Artifacts

- `checkpoint_0_model.zip`: exact zero-residual LQR baseline;
- `checkpoint_<N>_model.zip`: evaluated SAC checkpoints;
- `best_model.zip`: validation-ranked checkpoint;
- `final_model.zip`: end-of-training policy.

Configuration, rollout tables, and `replay_statistics.json` are in
`../../../results/lqr_reference_residual_stratified/sac_closing_lqr_reference_residual_pilot_e_actor_prior_annealed_1hz_50k_seed0/`.
