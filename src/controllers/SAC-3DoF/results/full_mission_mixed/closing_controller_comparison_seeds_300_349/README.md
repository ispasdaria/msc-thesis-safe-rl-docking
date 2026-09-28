# Closing-controller comparison, seeds 300 to 349

This 50-mission campaign compares four mixed-controller configurations.

- `lqr_final_r1_s12_100k` uses LQR closing at `dt = 0.1 s` and the round-1 final SAC policy trained with seed 12 for 100k steps. It completed 29 of 50 missions.
- `lqr_final_r5_s11_600k` uses the same LQR closing controller and the round-5 final SAC policy trained with seed 11 for 600k steps. It completed 29 of 50 missions.
- `balanced100k_final_r1_s12_100k` uses the phase-balanced closing SAC checkpoint at 100k steps and the round-1 final SAC policy. It produced no closing successes.
- `vanilla400k_final_r1_s12_100k` uses the vanilla reward-15 closing SAC policy at 400k steps and the round-1 final SAC policy. It produced no closing successes.

Both closing SAC policies use `dt = 1 s`. Their missions timed out during closing, so the final-approach policy was not activated. This comparison supports the selection of LQR for the retained closing phase.
