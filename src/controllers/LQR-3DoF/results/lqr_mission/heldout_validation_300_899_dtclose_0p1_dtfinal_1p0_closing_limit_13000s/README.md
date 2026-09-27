# Held-Out Full-Mission Validation with a 13,000 s Closing Limit

This evaluation contains 600 missions over seeds 300 to 899 using the selected
LQR phase controllers.

Configuration:

- Closing control interval: `0.1 s`
- Final-approach control interval: `1 s`
- Closing duration limit: `13,000 s`
- Final-approach duration limit: `5,000 s`
- Mode: `testing`
- Handover: the actual closing terminal state becomes the final-approach initial
  state

Results:

- 344 successful missions, or 57.33 percent
- 250 closing timeouts
- Six closing safety violations
- 100 percent final-approach success among the 344 missions that reached it

The CSV contains per-mission outcomes, timing, safety, terminal-error,
saturation, and control-effort measurements. The JSON contains the controller
matrices, configuration, aggregate results, confidence intervals, and mean
metrics.

This is distinct from the existing 600-mission evaluation with a `12,000 s`
closing limit, which produced 328 successful missions.
