# CBF evaluations

- `evaluate_closing_cbf.py` compares closing-phase SAC policies with closing CBF variants.
- `evaluate_final_approach_cbf.py` compares final-approach SAC policies with cone, speed, and alignment filters.
- `evaluate_lqr_cbf.py` evaluates the closing and final-approach LQR controllers with the corresponding hard CBF-QP filters.

All evaluators use paired seeds and write beneath the matching phase directory in `../results/`. Run them from the repository root so external controller and environment paths remain explicit.

Each evaluation writes its resolved configuration, per-episode metrics,
aggregate metrics, and, where enabled, step-level NPZ histories. The
command-line interface for each entry point is available with `--help`.

Representative retained outputs are:

- `../results/closing/evaluations/controller_campaigns/campaign_phase_balanced_100k_seeds_0_49/` for closing-phase SAC;
- `../results/final_approach/evaluations/controller_campaigns/campaign_round5_seed10_best490k_seeds_0_49/` for final-approach SAC;
- `../results/lqr/closing_dt0p1_full_cbf_family_breakdown_seeds_0_49/` for LQR and per-family intervention fields.
