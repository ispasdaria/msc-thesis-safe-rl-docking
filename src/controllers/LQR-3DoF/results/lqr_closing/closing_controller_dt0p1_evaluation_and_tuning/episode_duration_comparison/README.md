# Episode Duration Comparison

This directory contains supporting material for the episode-limit study.

- `closing_duration_comparison_seed0041_seed0036.png` compares a representative
  successful rollout with a representative timeout.
- `histories_seed0036_t14000` contains the seed-36 trajectory generated with a
  `14,000 s` episode limit.

The comparison determines whether additional simulation time resolves the
remaining closing failures. The longer horizon does not resolve the selected
timeout, supporting the conclusion that the dominant failure is terminal
convergence rather than insufficient episode duration.
