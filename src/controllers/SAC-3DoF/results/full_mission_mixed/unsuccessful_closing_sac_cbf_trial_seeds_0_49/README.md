# Unsuccessful closing SAC and CBF trial, seeds 0 to 49

This development campaign combines the vanilla reward-15 closing SAC policy at 400k steps with the full closing CBF and the round-1 final SAC policy trained with seed 12 for 100k steps.

Both phases were configured with `dt = 1 s`. All 50 missions timed out during closing, so the final-approach SAC policy was never activated. The CBF intervened on approximately 62% of closing steps on average and reported no quadratic-program failures.

The folder is retained as evidence that this closing SAC and CBF configuration did not provide a viable full-mission handover. It is not part of the retained mixed-controller performance campaign.
