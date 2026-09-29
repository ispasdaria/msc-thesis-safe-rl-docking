# Initial alignment-CBF integration results

## Scope

The alignment extension adds four affine QP rows to the historical cone and
port-speed filter:

- upper and lower attitude HOCBF rows protecting `|psi_error| <= 3 deg`;
- upper and lower angular-rate CBF rows protecting
  `|omega| <= 0.15 deg/s`.

The integration checks below use the previously selected cone/speed gains and
initial alignment gains:

```text
cone_k1 = cone_k2 = 0.3
speed_alpha = 0.5
attitude_k1 = attitude_k2 = 0.3
angular_rate_alpha = 0.5
```

All runs use hard constraints, OSQP, RK4, and `dt = 1 s`.

## Alignment-failure controller

SAC training seed 0 was evaluated on scenario seeds 0--9. With cone and speed
protection only, all ten episodes failed on the simulator's alignment limit
after a mean of 94.8 steps. With all seven CBF rows enabled:

- no episode crossed the simulator's alignment failure limits;
- no attitude or angular-rate protected barrier became negative;
- no QP solve failed and no slack was available;
- all ten episodes reached the 5000-step evaluation limit without docking.

This demonstrates safety-failure prevention, not recovery of a poor nominal
policy. The filter cannot guarantee task completion when the SAC controller
does not provide a useful docking command under the added constraint.

The complete artifacts are in:

```text
results/final_approach/evaluations/cbf_development_and_holdout/sac_seed00_cone_tuned_alignment_initial_hard_translational_vs_all_cbf_eval_seeds0000_0009
```

## Non-degradation controller

SAC training seed 10 was evaluated with all seven rows on scenario seeds
0--9. It retained 10/10 successful dockings, with zero QP failures and no
negative cone, speed, attitude, or angular-rate barrier samples. Mean
intervention frequency was approximately 0.288%.

The complete artifacts are in:

```text
results/final_approach/evaluations/cbf_development_and_holdout/sac_seed10_cone_tuned_alignment_initial_hard_all_cbf_eval_seeds0000_0009
```

## Sampled-data caveat and next step

For SAC seed 0, the alignment-protected trajectories slightly exceeded the
conservative 0.15 m/s port-speed envelope. The largest observed speed was
approximately 0.150133 m/s. This remained well below the simulator's 0.2 m/s
unsafe threshold, but it is a protected-set crossing and must not be reported
as hard sampled-time invariance.

Trying `speed_alpha = 0.2` on the worst observed scenario did not eliminate
the small crossing. The complete probe is retained under:

```text
results/final_approach/evaluations/cbf_development_and_holdout/sac_seed00_cone_tuned_alignment_initial_hard_all_cbf_probe_eval_seed0008_speed_alpha0p2
```

Before freezing the seven-row filter, its response gains should be tuned
jointly with the updated Optuna script and checked on held-out controllers and
scenario seeds. A smaller controller timestep or a sampled-data/tightened
speed formulation should be considered if zero 0.15 m/s crossings cannot be
obtained through gain selection.
