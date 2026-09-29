# Pilot G P3-stratified reference-residual SAC model card

Run: `sac_closing_lqr_reference_residual_pilot_g_dt0p1_reward_v2_500k_seed0`  
Status: **complete**  
Selected best checkpoint: **250,000 SAC decisions**

## What this experiment changes

Pilot G uses a zero-residual actor penalty annealed linearly from 1 at interaction 100,000 to 0.1 at interaction 300,000, then held constant. Pilot G changes only the reward from the 10 Hz Pilot E ablation. The retuned LQR, replay distribution, seeds, interaction counts, and SAC optimizer are unchanged.

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

Final replay audit: Global/outer/inner stored: 620,000/100,770/10,506. Realized sampled fractions: 61.5%/25.0%/13.5%.

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
reward = 2.000 * (Jw(previous) - gamma * Jw(next))
         + 4.000 * (L(previous) - gamma * L(next))
         + 5.000 * exact_P3_success
         - 25.000 * hard_failure
         - 2.000 * severity_scaled_timeout
         - dt * 0.002 * ||a||^2
         - (1 / dt) * 0.001 * ||a - a_previous||^2
```

`Jw` uses component weights [1.0, 4.0, 1.0, 2.0, 1.0, 1.0] and gives the worst component weight 0.75. `L` is the log-scaled excess of the lateral-position ratio above one. Timeout severity uses the remaining worst P3 violation. The replay strata use the maximum untransformed ratio from
the actual next state and actual fixed P3 target.

## SAC configuration

| Parameter | Value |
|---|---:|
| Online interactions | 500,000 |
| Replay prefill | 100,000 |
| Global/outer/inner capacities | 650,000/150,000/50,000 |
| Critic-only warm-up | 100,000 interactions |
| Learning rate | 0.0001 |
| Effective actor learning rate | 2.5e-05 |
| Actor zero-reference weight | 1 at 100,000; linear to 0.1 at 300,000 |
| Train frequency / gradient steps | 40 / 1 |
| Batch size | 256 |
| Discount factor | 0.99999000 |
| Target smoothing | 0.0050 |
| Entropy coefficient | `auto_0.01` |
| Target entropy | -3.0 |

## Validation results

Validation seeds: 100--109.

| Checkpoint | Success | Hard failure | Timeout | Mean minimum position error [m] | Mean best worst ratio | Mean seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 40.0% | 0.0% | 60.0% | 15.006 | 1.133 | 11782.0 |
| 100,000 | 40.0% | 0.0% | 60.0% | 15.006 | 1.133 | 11782.0 |
| 250,000 | 50.0% | 0.0% | 50.0% | 15.614 | 1.138 | 11752.5 |
| 500,000 | 50.0% | 0.0% | 50.0% | 15.912 | 1.146 | 11760.0 |

## Interpretation

Checkpoint 250,000 ranked best on the validation seeds and is stored as `best_model.zip`.

## Artifacts

- `checkpoint_0_model.zip`: exact zero-residual LQR baseline;
- `checkpoint_<N>_model.zip`: evaluated SAC checkpoints;
- `best_model.zip`: validation-ranked checkpoint;
- `final_model.zip`: end-of-training policy.

Configuration, rollout tables, and `replay_statistics.json` are in
`../../../results/lqr_reference_residual_stratified/sac_closing_lqr_reference_residual_pilot_g_dt0p1_reward_v2_500k_seed0/`.
