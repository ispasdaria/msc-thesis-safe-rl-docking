# Held-Out Validation over 300 Seeds

This directory contains the independent evaluation over seeds 1000-1299. The
campaign uses `dt=0.1 s`, a `12,000 s` episode limit, and the selected baseline
controller configuration. These seeds were not used by the 50-seed tuning
studies.

The evaluation reports:

- 156 successes (`52%`);
- 143 timeouts;
- one hard velocity safety violation;
- 122 lateral-position tolerance failures.

The validation confirms that the controller succeeds in approximately half of
the tested initial conditions and that lateral terminal position is the
dominant remaining limitation. The CSV contains per-rollout measurements; the
JSON file contains aggregate statistics and the evaluation configuration.
