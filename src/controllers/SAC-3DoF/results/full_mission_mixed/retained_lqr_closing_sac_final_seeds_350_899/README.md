# Retained LQR-closing and SAC-final campaign, seeds 350 to 899

This folder extends the successful mixed-controller configurations from the initial seed block by 550 missions.

- `lqr_final_r1_s12_100k` completed 299 of 550 missions.
- `lqr_final_r5_s11_600k` completed 299 of 550 missions.

Both configurations use LQR closing with `dt = 0.1 s` and final-approach SAC with `dt = 1 s`. Every mission that reached the final phase completed docking. The unsuccessful missions ended during closing.

Combined with seeds 300 to 349, each configuration was evaluated over 600 missions and completed 328 missions, corresponding to approximately 54.7% success.
