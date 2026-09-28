# Representative full-mission histories

These compressed histories reproduce the retained mixed-controller full-mission figures.

- Seeds 306, 403, and 803 use LQR closing and the round-1 final SAC policy trained with seed 12 for 100k steps.
- The additional seed-803 history uses the same LQR closing trajectory and the round-5 final SAC policy trained with seed 11 for 600k steps.

The files store complete closing and final-approach state histories together with handover, terminal-state, timing, and geometry metadata. They are consumed by `plotters/plot_representative_mixed_full_mission_trajectories.py`.
