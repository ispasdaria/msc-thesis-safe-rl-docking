# LQR result analysis

- `analyze_lqr_weight_scaling_robustness.py` recomputes the continuous-time LQR gains under moderate Q and R scaling and evaluates continuous and sampled-data closed-loop stability.
- `analyze_rollout_convergence.py` estimates how saved mission metrics converge as the number of evaluated rollouts increases. Random permutations remove dependence on the original seed ordering.

The scripts read retained evaluation data or controller definitions and write analysis tables under the corresponding LQR result directory.
