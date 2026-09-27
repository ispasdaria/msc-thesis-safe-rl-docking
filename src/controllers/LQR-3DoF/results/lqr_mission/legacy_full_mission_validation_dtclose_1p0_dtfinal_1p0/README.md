# Legacy Full-Mission Validation with 1 s Control Intervals

This directory contains the held-out full-mission evaluation of the earlier LQR
configuration. Both the closing and final-approach controllers use a `1 s`
control interval. The closing limit is `12,000 s`, and the final-approach limit
is `5,000 s`.

The evaluation was executed in two 300-seed batches and then combined:

- Seeds 300 to 599 produced 239 successful missions.
- Seeds 600 to 899 produced 251 successful missions.
- The combined evaluation produced 490 successes from 600 missions.
- Every mission that reached final approach completed successfully.
- The only failures occurred during closing: 109 timeouts and one safety
  violation.

The closing controller in this legacy evaluation exhibits near-continuous
actuator saturation. These results should not be interpreted as the performance
of the later closing controller with a `0.1 s` control interval.

The two seed-batch directories preserve the independently generated results and
logs. `combined_seeds_300_899` contains their exact concatenation and the
600-seed convergence analysis.
