# Nonterminal-Safety Continuation

These runs continued the retained reward-V15 Trial-16 policy for 300,000 additional steps while safety violations remained penalized but did not terminate the episode. One run retained V15; the other used P3-reach V1.

The experiment tested whether longer unsafe trajectories provided useful recovery data. Checkpoint metrics, training safety diagnostics, and validation rollouts record task outcome and unsafe-state occupancy. These are diagnostic continuations rather than selected controllers.

Matching checkpoints and logs are under `../../models/nonterminal_safety_continuation/` and `../../logs/nonterminal_safety_continuation/`.
