# Near-to-Far Curriculum Experiment

This family contains the near-to-far curriculum experiment implemented by `train_sac_final_curriculum.py`. Training progresses through close, middle, and standard final-approach initial-state ranges while using the curriculum basis reward.

The directory contains 200,000-step policies for training seeds 0, 1, and 2. Each policy has a matching VecNormalize file. The experiment was retained as development history and was not selected as the final controller.

Related logs and learning curves are stored under the matching `curriculum_experiment` development directories.
