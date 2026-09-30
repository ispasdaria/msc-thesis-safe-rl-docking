# Round 1 Seed 12 Final-Approach CBF Campaign

This directory contains the retained 50-seed evaluation of the round-1,
training-seed-12 final-approach SAC policy at its final 100,000-step
checkpoint. The machine-readable settings used to generate these artifacts
are preserved in `run_configuration.json`.

## CBF gain provenance

This campaign used the historical evaluator defaults that were active when
the rollouts were generated:

| Gain | Campaign value |
| --- | ---: |
| Cone gain, `cone_k1 = cone_k2` | 0.1 |
| Speed gain, `speed_alpha` | 0.5 |
| Attitude gain, `attitude_k1 = attitude_k2` | 0.3 |
| Angular-rate gain, `angular_rate_alpha` | 0.5 |

These values differ from the later gain-tuning selection reported in the
thesis: cone gain 0.3, speed gain 0.5, attitude gain 0.5, and angular-rate
gain 1.0. The later selection is preserved separately at
`../../../tuning/alignment_cbf_competent_controllers_grid/selected_cbf_gain_configuration.json`.

The campaign artifacts were not regenerated or relabeled with the later
gains. Results in this directory must therefore be interpreted using the
historical values recorded above and in `run_configuration.json`.
