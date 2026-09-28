# SAC plotters

This directory contains plots that combine or summarize retained SAC experiment data.

- `plot_sac_evaluation_summary.py` creates an empirical distribution of final docking-port distance and compares normalized closing terminal errors between the retained vanilla and phase-balanced policies. It reads committed evaluation CSV and JSON files without loading a policy or rerunning an environment.
- `plot_pre_post_optuna_learning_curves.py` compares raw episode returns and identically computed trailing averages before and after Optuna tuning. It writes two PNG figures and a JSON metadata manifest under `../results/pre_post_optuna_comparison`.
- `plot_representative_mixed_full_mission_trajectories.py` plots representative missions using the LQR closing controller and a retained final-approach SAC policy. Existing compact histories are reused; missing histories can be regenerated through the retained mixed-mission evaluation configuration. Figures are written under `../figures/representative_full_mission_figures`.

Default figure outputs are stored under `../figures`. Result-derived learning-curve figures remain under `../results/pre_post_optuna_comparison` with their metadata manifest.
