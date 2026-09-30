"""Compatibility exports for goal-conditioned closing environments."""

from envs.docking3dof.rewards.goal_conditioned import (
    GoalConditionedClosingEnv,
    GoalConditionedClosingV2Env,
    GoalConditionedClosingV3Env,
)


__all__ = [
    "GoalConditionedClosingEnv",
    "GoalConditionedClosingV2Env",
    "GoalConditionedClosingV3Env",
]
