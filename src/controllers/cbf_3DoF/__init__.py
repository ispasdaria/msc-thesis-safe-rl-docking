"""Control-barrier-function safety filters for the 3-DoF docking model."""

from .barriers import FinalApproachCBFConfig, build_final_approach_constraints
from .closing_barriers import ClosingCBFConfig, build_closing_constraints
from .diagnostics import (
    attitude_hocbf_admissibility,
    cone_hocbf_admissibility,
    hard_cbf_feasibility_margin,
)
from .qp_filter import ClosingCBFFilter, FinalApproachCBFFilter

__all__ = [
    "ClosingCBFConfig",
    "ClosingCBFFilter",
    "FinalApproachCBFConfig",
    "FinalApproachCBFFilter",
    "build_closing_constraints",
    "build_final_approach_constraints",
    "attitude_hocbf_admissibility",
    "cone_hocbf_admissibility",
    "hard_cbf_feasibility_margin",
]
