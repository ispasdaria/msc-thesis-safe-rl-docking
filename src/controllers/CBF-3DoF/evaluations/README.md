# CBF evaluations

- `evaluate_closing_cbf.py` compares closing-phase SAC policies with closing CBF variants.
- `evaluate_final_approach_cbf.py` compares final-approach SAC policies with cone, speed, and alignment filters.
- `evaluate_lqr_cbf.py` evaluates the closing and final-approach LQR controllers with the corresponding hard CBF-QP filters.

All evaluators use paired seeds and write beneath the matching phase directory in `../results/`. Run them from the repository root so external controller and environment paths remain explicit.
