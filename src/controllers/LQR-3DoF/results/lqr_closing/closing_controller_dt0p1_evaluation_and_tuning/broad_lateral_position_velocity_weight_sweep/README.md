# Broad Lateral Position and Velocity Weight Sweep

This sweep evaluates 12 combinations of multipliers applied to the LQR lateral
position weight (`my`) and lateral velocity weight (`mvy`). The controller
architecture, reference construction, `R` matrix, `0.1 s` time step, and
`12,000 s` episode limit remain unchanged.

`gain_and_stability_precheck.json` records the gains and sampled-system
stability checks. All 12 candidates passed the stability check. Each candidate
directory contains results for seeds 0-49, while the aggregate CSV and JSON
files provide the safety-first ranking.

Reducing the lateral velocity weight increased the number of successes but also
introduced velocity safety violations. The unchanged `my=1, mvy=1` candidate
ranked first under the safety-first rule with 23 successes and no hard safety
violations. The `my=1, mvy=0.75` candidate produced 27 successes but two hard
velocity violations. The broad sweep therefore did not identify a clearly
superior safety-preserving configuration and motivated the narrower follow-up
sweep.
