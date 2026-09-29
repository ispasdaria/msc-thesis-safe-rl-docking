# Direct residual SAC model card

Run: `sac_closing_goal_her_direct_residual_dt0p1_terminal_fade_200k_seed0`  
Status: **complete**  
Selected best checkpoint: **120,000 residual decisions**

## Purpose

This experiment leaves the closing LQR law unchanged. SAC selects one
normalized residual action every 0.1 s (1
simulator step(s)), while the LQR action is recomputed from the current physical
state every 0.1 s. With one hold step, both feedback decisions
operate at 10 Hz.

```text
u_executed = clip(u_LQR_clipped + 0.05 * terminal_fade(P3_ratio) * a_residual, -1, 1)
```

Residual authority is full while the worst P3 ratio is at or above 1.2, uses a smoothstep terminal fade, and becomes zero only at or below ratio 1.0.

## SAC configuration

| Parameter | Value |
|---|---:|
| Online residual decisions | 200,000 |
| Residual decision interval | 0.1 s |
| LQR update interval | 0.1 s |
| Replay prefill transitions | 200,000 |
| Critic warm-up | 30,000 residual decisions/updates |
| Critic learning rate | 0.0001 |
| Actor learning-rate scale | 0.25 |
| Effective actor learning rate | 2.5e-05 |
| Train frequency | 1 |
| Gradient steps | 1 |
| Update-to-data ratio | 1 |
| Discount factor | 0.999900 per residual decision |
| Target smoothing coefficient | 0.0050 |
| Entropy coefficient | `auto_0.001` |
| Target entropy | -6.0 |
| Zero-residual actor weight | 1.0 |
| Final zero-residual actor weight | 0.5 |
| Batch size | 512 |
| HER future goals per transition | 4 |

The first 30,000 online residual decisions use
the deterministic zero actor and update only the critic. With `train_freq=1`
and `gradient_steps=1`, this produces one critic update per collected residual
transition during warm-up.

## Reward

The HER-compatible reward is potential-based:

```text
reward = 2.0 * (J(previous) - gamma * J(next))
       + 4.0
         * (L(previous) - gamma * L(next))
       + 5.0 * exact_P3_success
       - 25.0 * hard_failure
       - 2.0
         * (1 + 2.0 * worst_violation)
         * unsuccessful_timeout
```

`J` weights lateral and lateral-velocity errors more heavily, while `L`
measures only lateral P3-set violation. Previous achieved goals are retained in
replay so HER can recompute both progress terms for relabeled goals.

The critic-visible reward also subtracts time-consistent action costs:

```text
0.1
  * 0.0020 * ||a_effective||^2
+ 10.0
  * 0.0010
  * ||a_effective - a_previous||^2
```

The constrained actor additionally optimizes
`1.0 * ||a_residual||^2`. This is an
actor-only regularizer and is not part of the environment reward or critic
target. The reward returned for a held residual decision is the final
0.1-second substep reward so HER can recompute it consistently; the sum across
substeps is retained only as an `info` diagnostic.

## Validation results

Validation uses seeds 100--109.
`Mean decisions` counts SAC residual decisions; `Approx. seconds` multiplies
that count by 0.1 s and may overestimate an episode whose final held
action ended early on success or termination.

| Checkpoint | Success | Hard failure | Timeout | Mean minimum position error [m] | Mean best worst ratio | Mean decisions | Approx. seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 40.0% | 0.0% | 60.0% | 15.006 | 1.133 | 117819.7 | 11782.0 |
| 120,000 | 40.0% | 0.0% | 60.0% | 14.926 | 1.132 | 117797.0 | 11779.7 |
| 200,000 | 40.0% | 0.0% | 60.0% | 15.057 | 1.133 | 117830.3 | 11783.0 |

## Artifacts

- `checkpoint_0_model.zip`: exact zero-residual LQR baseline;
- `checkpoint_<N>_model.zip`: evaluated model at residual decision `N`;
- `best_model.zip`: best validation-ranked checkpoint;
- `final_model.zip`: policy at the end of training.

The matching metrics and exact configuration are under
`../../../results/her_lqr_residual/sac_closing_goal_her_direct_residual_dt0p1_terminal_fade_200k_seed0/`.
