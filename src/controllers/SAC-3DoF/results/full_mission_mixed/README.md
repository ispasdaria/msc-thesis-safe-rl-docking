# Mixed-controller full-mission results

These results evaluate complete missions in which the closing and final-approach phases may use different controllers. The retained configuration uses the LQR closing controller with `dt = 0.1 s` and a final-approach SAC policy with `dt = 1 s`.

- `closing_controller_comparison_seeds_300_349` is the initial 50-mission comparison between LQR closing and two closing SAC policies.
- `retained_lqr_closing_sac_final_seeds_350_899` extends the two successful LQR-closing configurations by 550 missions.
- `representative_full_mission_histories` contains complete histories used by the retained full-mission figures.
- `unsuccessful_closing_sac_cbf_trial_seeds_0_49` records an unsuccessful development experiment using closing SAC with a CBF.

The CSV files contain one record per mission. Each configuration summary contains aggregate statistics for one controller combination. A `campaign_summary.json` file combines the configuration summaries evaluated in that seed block.

The campaign runner is `src/controllers/run_mixed_full_mission_evaluations.py`. The numerical campaign audit is `src/controllers/analyze_mixed_full_mission_results.py`.
