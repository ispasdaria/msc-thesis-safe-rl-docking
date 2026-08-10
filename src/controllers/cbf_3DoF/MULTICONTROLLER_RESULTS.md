# Final-approach multi-controller CBF evaluation

## Method

Six independently trained SAC checkpoints were selected from the 600k-step,
600k-replay-buffer experiment. They cover velocity failures, low/intermediate/
high geometry success rates, and an already-perfect controller.

Each controller was evaluated on testing seeds 0--9. Baseline results come from
the existing best-checkpoint multi-seed evaluation. Filtered results use:

- the exact-cone HOCBF and total-port-speed CBF together;
- hard CBF inequalities (no slack);
- OSQP;
- the same test seeds and model-specific VecNormalize statistics;
- the initial untuned gains in `FinalApproachCBFConfig`.

## Results

| SAC training seed | Dominant baseline failure | Baseline success | CBF success | Change | Mean intervention fraction | QP failures |
|---:|---|---:|---:|---:|---:|---:|
| 0 | velocity | 0/10 | 0/10 | 0 | 4.15% | 0 |
| 1 | geometry | 1/10 | 10/10 | +9 | 2.18% | 0 |
| 10 | geometry | 7/10 | 10/10 | +3 | 1.69% | 0 |
| 14 | geometry | 3/10 | 10/10 | +7 | 1.81% | 0 |
| 15 | geometry | 9/10 | 10/10 | +1 | 2.34% | 0 |
| 16 | none | 10/10 | 10/10 | 0 | 1.43% | 0 |

Across this deliberately stratified panel, success increased from 30/60 (50%)
to 50/60 (83.3%). For the four geometry-failure controllers, it increased from
20/40 (50%) to 40/40 (100%). The already-perfect seed-16 controller remained
at 10/10, providing an initial non-degradation check.

The seed-0 result is important. Its original total-speed violations were
prevented: filtered terminal port speeds remained about 0.148--0.149 m/s,
inside the 0.15 m/s CBF envelope. However, all ten episodes then crossed the
unprotected angular-rate alignment boundary. Thus the filter changed the
failure mode from velocity to alignment but did not produce docking success.
This is expected from a partial filter that enforces only two of the final-
approach safety properties.

## Interpretation

The evidence supports the claim that the exact-cone CBF materially improves
SAC controllers whose recurrent failure is geometric cone escape. It does not
support a claim that the two-CBF filter can repair every poor policy or enforce
the full simulator safety definition. In particular, attitude and angular-rate
alignment are not constrained by the current QP.

The panel was selected to cover known performance regimes rather than sampled
randomly from all checkpoints. It should therefore be presented as a
controlled failure-mode experiment. A final report should additionally give
results over all selected controllers and a larger held-out evaluation-seed
set after the CBF gains have been frozen.

## Result directories

Per-step actions, barrier values, residuals, and per-rollout metrics are stored
locally under:

```text
src/controllers/cbf_3DoF/results/final_approach/sac_seed00_untuned_hard_combined_cbf_eval_seeds0000_0009
src/controllers/cbf_3DoF/results/final_approach/sac_seed01_untuned_hard_combined_cbf_eval_seeds0000_0009
src/controllers/cbf_3DoF/results/final_approach/sac_seed10_untuned_hard_combined_cbf_eval_seeds0000_0009
src/controllers/cbf_3DoF/results/final_approach/sac_seed14_untuned_hard_combined_cbf_eval_seeds0000_0009
src/controllers/cbf_3DoF/results/final_approach/sac_seed15_untuned_hard_combined_cbf_eval_seeds0000_0009
src/controllers/cbf_3DoF/results/final_approach/sac_seed16_untuned_hard_combined_cbf_eval_seeds0000_0009
```
