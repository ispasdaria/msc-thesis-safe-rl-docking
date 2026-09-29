# Safety-Terminal Braking Experiments

These from-scratch experiments terminate immediately on hard closing-safety violations.

- V19 combines the V18 capture objective with an LQR-inspired axial braking envelope.
- V20 removes persistent goal-state and time costs and balances the hard-failure penalty against the success reward.
- V21 replaces scalar axial tracking with a coupled docking-port velocity, attitude, and angular-rate envelope.

V19 used a 600,000-step budget. V20 and V21 were 100,000-step diagnostic pilots. Stored checkpoint tables and rollout metrics are the authoritative outcomes.

Matching checkpoints and logs are under `../../models/safety_terminal_braking_experiments/` and `../../logs/safety_terminal_braking_experiments/`.
