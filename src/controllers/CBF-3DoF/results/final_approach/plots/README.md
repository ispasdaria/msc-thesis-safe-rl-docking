# Representative final-approach SAC + CBF plots

These figures were generated directly from the schema-2 trajectory artifacts
of the 300k-timestep, 300k-replay-buffer SAC evaluation. Each case contains
the same five comparisons in PNG and vector PDF form:

- spatial docking trajectory and protected cone
- protected speed, attitude, and angular-rate envelopes
- safety-filter intervention and action correction
- hard-QP feasibility diagnostics
- sampled cone-boundary tolerance zoom

The directory names state why each trajectory was selected:

1. `alignment_constraint_rescue_training_seed03`: unfiltered SAC fails the
   cone, cone+speed fails alignment, and the full CBF succeeds.
2. `full_cbf_non_degradation_training_seed16`: cone+speed and full CBF both
   succeed, demonstrating that alignment protection does not spoil an already
   successful filtered trajectory.
3. `cone_hard_qp_infeasibility_training_seed00`: the hard QP is infeasible due
   to the cone constraint family.
4. `cone_crossing_within_numerical_tolerance_training_seed10`: the strict cone
   crossing is smaller than the 1e-6 reporting tolerance.
5. `cone_crossing_beyond_numerical_tolerance_training_seed10`: the cone
   crossing is larger than the 1e-6 reporting tolerance.
6. `qp_nominal_fallback_limitation_training_seed17`: docking succeeds despite
   failed QP steps and therefore illustrates that the nominal-action fallback
   is not continuously CBF-certified.

`training_seedXX` identifies the SAC checkpoint. The nested `eval_seedXXXX`
identifies the paired simulator initial condition, not another trained model.
The source NPZ, CSV, and configuration artifacts remain in their corresponding
directories under `../evaluations/`.
