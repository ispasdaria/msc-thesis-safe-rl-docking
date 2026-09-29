"""Controller adapter for applying a CBF-QP to classical controllers.

The Gym shields convert normalized policy actions.  Classical controllers in
this repository instead implement ``compute_action(env, state)`` and return a
physical body-frame command.  This adapter keeps that controller contract and
projects the returned command through the phase-appropriate CBF-QP.
"""

from __future__ import annotations

import numpy as np

from .barriers import FinalApproachCBFConfig
from .closing_barriers import ClosingCBFConfig
from .qp_filter import ClosingCBFFilter, FinalApproachCBFFilter


class CBFQPController:
    """Wrap a physical-action controller with the appropriate CBF-QP filter.

    Parameters
    ----------
    nominal_controller:
        An object implementing ``compute_action(env, state)`` (including both
        repository LQR controllers), or a callable accepting ``(env, state)``.
    config:
        Optional phase-specific CBF configuration.  When omitted, the default
        configuration for the environment phase is used.

    Notes
    -----
    ``compute_action`` returns the filtered physical body-frame command, so a
    wrapped controller can be registered directly with ``DockingMissionRunner``.
    The most recent :class:`FilterResult` is available as ``last_result`` for
    logging solver status, interventions, residuals, and slack.
    """

    def __init__(self, nominal_controller, config=None):
        if nominal_controller is None:
            raise ValueError("nominal_controller must not be None.")
        if not (
            hasattr(nominal_controller, "compute_action")
            or callable(nominal_controller)
        ):
            raise TypeError(
                "nominal_controller must be callable or implement "
                "compute_action(env, state)."
            )
        if config is not None and not isinstance(
            config, (ClosingCBFConfig, FinalApproachCBFConfig)
        ):
            raise TypeError(
                "config must be ClosingCBFConfig or FinalApproachCBFConfig."
            )

        self.nominal_controller = nominal_controller
        self.config = config
        self.cbf_filter = None
        self.last_result = None

    def _make_filter(self, env):
        phase = getattr(env, "phase", None)
        if phase == "closing":
            if isinstance(self.config, FinalApproachCBFConfig):
                raise ValueError(
                    "FinalApproachCBFConfig cannot be used in the closing phase."
                )
            return ClosingCBFFilter(env, self.config)
        if phase == "final":
            if isinstance(self.config, ClosingCBFConfig):
                raise ValueError(
                    "ClosingCBFConfig cannot be used in the final phase."
                )
            return FinalApproachCBFFilter(env, self.config)
        raise ValueError(
            "CBFQPController requires an environment whose phase is "
            "'closing' or 'final'."
        )

    def _nominal_action(self, env, state):
        if hasattr(self.nominal_controller, "compute_action"):
            action = self.nominal_controller.compute_action(env, state)
        else:
            action = self.nominal_controller(env, state)
        return np.asarray(action, dtype=np.float64)

    def compute_action(self, env, state):
        """Return the CBF-filtered physical body-frame controller action."""
        # Filters retain environment dynamics and limits, so rebuild if this
        # reusable adapter is registered on a different environment instance.
        if self.cbf_filter is None or self.cbf_filter.core_env is not env:
            self.cbf_filter = self._make_filter(env)

        nominal = self._nominal_action(env, state)
        self.last_result = self.cbf_filter.filter(state, nominal)
        return self.last_result.action_filtered.copy()


def with_cbf_qp(controller, config=None):
    """Convenience factory for wrapping an LQR or other physical controller."""
    return CBFQPController(controller, config=config)
