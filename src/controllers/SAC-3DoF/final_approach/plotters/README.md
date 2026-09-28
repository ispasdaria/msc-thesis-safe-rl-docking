# SAC Final-Approach Plotters

This directory contains plotting-only scripts for final-approach SAC experiments. The scripts read stored training or evaluation artifacts and do not train policies.

- `plot_episode_returns.py` plots episode-return histories from development training runs.
- `plot_final_geometry_trajectory.py` plots final-approach geometry for a stored rollout.
- `plot_retained_policy_episode_duration.py` plots episode duration for the retained seed-12 100k policy.
- `plot_retained_sac_final_approach_trajectories.py` plots representative retained-policy trajectories.
- `plot_retained_sac_training_results.py` plots training metrics for retained policies.
- `plot_sac_final_rollouts.py` plots state and control histories from saved evaluation rollouts.
- `plot_seed3_actor_critic_losses.py` compares actor and critic losses for the seed-3 development experiment.
- `plot_seed3_buffer_comparison.py` compares seed-3 replay-buffer development experiments.
- `plot_selected_policy_analysis.py` plots stored selected-policy evaluations.

Input and output paths are resolved relative to `final_approach`, independent of the process working directory.
