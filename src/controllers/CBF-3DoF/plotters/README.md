# CBF plotters

- `plot_final_approach_cbf_trajectories.py` plots saved final-approach evaluations.
- `plot_constraint_and_control_diagnostics.py` plots retained constraint histories and CBF control interventions.
- `plot_lqr_cbf_evaluation.py` compares retained LQR trajectories with their CBF-filtered counterparts.
- `plot_sac_cbf_trajectory_comparisons.py` compares representative SAC and SAC-CBF trajectories across closing and final approach.

These scripts consume existing CSV, JSON, and NPZ artifacts. They do not rerun training or evaluation.
