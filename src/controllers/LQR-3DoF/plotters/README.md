# LQR result plotters

The plotters are grouped by the phase represented in the output.

- `closing` contains plots of commands, actuator saturation, terminal requirements, sampled-data stability, schedule behavior, and representative trajectories.
- `final_approach` contains terminal-state convergence and reference-schedule comparisons.
- `full_mission` contains campaign-level figures and plots of saved two-phase mission histories.

Default input and output paths refer to the retained files under `../results`. Plotters that require missing rollout histories identify the evaluation script needed to generate them.
