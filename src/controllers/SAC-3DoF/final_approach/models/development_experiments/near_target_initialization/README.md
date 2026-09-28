# Near-Target Initialization Experiments

This family tests whether SAC can learn the final docking segment from initial states closer to the target than the standard final-approach distribution. The stored policies cover several Gym reward revisions, 100,000-step and 200,000-step budgets, and a warm-start experiment that expands from 35-70 m to 60-120 m. All runs use training seed 0.

Final and best checkpoints are stored at the family root. The `checkpoints` directory contains intermediate training snapshots. One 10,000-step checkpoint does not have a same-stem VecNormalize file and should not be loaded independently. Other models must be paired with their matching normalization file.

The experiment is developmental and was not selected as the final controller.
