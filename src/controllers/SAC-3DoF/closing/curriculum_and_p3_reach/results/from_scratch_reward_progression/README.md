# From-Scratch Reward Progression

This group records fresh SAC training used to develop P3-reach rewards V1-V19. Every run starts with a new policy, critics, replay buffer, and observation-normalization state. The run names identify the implemented reward distinction, training budget, gamma override when applicable, and seed.

The retained checkpoint and rollout diagnostics document successive failure modes involving terminal-component convergence, premature slowing, target departure, docking-cone containment, and braking-envelope tracking.

`cross_family_phase_balanced_replay_reference/` preserves the V12 phase-balanced run historically produced through this pipeline; the method itself is documented under the phase-balanced family.

Matching checkpoints and logs are under `../../models/from_scratch_reward_progression/` and `../../logs/from_scratch_reward_progression/`.
