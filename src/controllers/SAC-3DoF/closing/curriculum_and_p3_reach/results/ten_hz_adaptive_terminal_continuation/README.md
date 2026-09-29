# Ten-Hertz Adaptive Terminal Continuation

This V15-dt experiment continued the selected phase-balanced controller with `dt = 0.1 s`. It used one-second action holds in the far field, ten-hertz action recomputation near the terminal region, deterministic replay warm-up, and a mixture of standard and fixed near-P3 resets.

The retained output is a 200,000-step seed-0 continuation identified as run 2. Its source checkpoint remains in the phase-balanced family.

Matching checkpoints and logs are under `../../models/ten_hz_adaptive_terminal_continuation/` and `../../logs/ten_hz_adaptive_terminal_continuation/`.
