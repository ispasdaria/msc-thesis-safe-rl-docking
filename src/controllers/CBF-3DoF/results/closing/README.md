# Closing-phase CBF results

- `evaluations/controller_campaigns/` contains complete controller campaigns and selected-policy evaluations.
- `evaluations/reward_progression_evaluations/` contains P3-reaching reward-version comparisons.
- `evaluations/cbf_development_ablations/` contains development runs, Trial-16 studies, and V12 CBF ablations.
- `tuning/` contains the closing gain-search study.

The speed-CBF development runs use descriptive names:

- `trial16_400k_speed_cbf_ablation_seeds0000_0009/` is the ten-seed development ablation;
- `trial16_400k_speed_cbf_ablation_seeds0200_0299/` is the corresponding 100-seed held-out evaluation;
- `p3reach_v7_300k_baseline_vs_speed_cbfs_seeds0200_0204/` compares the P3-reach v7 policy with and without both speed barriers.

Two abandoned 20,000-second folders containing only partial trajectory files were removed. Complete evaluation folders retain their configuration, per-episode metrics, aggregate metrics, and available step histories.

`evaluations/controller_campaigns/campaign_phase_balanced_100k_seeds_0_49/`
is the representative closing campaign because it contains the configuration,
episode table, aggregate table, and retained step histories used by the
cross-controller trajectory figures.
