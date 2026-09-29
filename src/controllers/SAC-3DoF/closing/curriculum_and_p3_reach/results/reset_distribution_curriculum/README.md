# Reset-Distribution Curriculum

These runs tested staged reset distributions that begin near P3 and expand toward the full P2 closing distribution after fixed validation thresholds are met. The group includes Trial-16 warm starts, a from-scratch comparison, stage-0 retries, discount-factor and bootstrap comparisons, and V19/V20 safety-terminal variants.

Run names state the reward version, initialization, gamma when varied, reset scope, retry identifier, and seed. Stored configurations define the stages and promotion thresholds. The retained diagnostics show that strong local near-P3 performance did not by itself establish reliable full-distribution performance.

Matching checkpoints and TensorBoard logs are under `../../models/reset_distribution_curriculum/` and `../../logs/reset_distribution_curriculum/`.
