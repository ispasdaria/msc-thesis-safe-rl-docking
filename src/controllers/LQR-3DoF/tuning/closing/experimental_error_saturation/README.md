# Experimental error-saturation sweep

`sweep_closing_error_saturation_controller.py` tunes the alternate controller implemented in `lqr_3dof_closing_error_sat.py`. That controller limits the state error supplied to the feedback law before computing the command.

The experiment is separated from the primary closing-phase sweep because it changes the controller formulation rather than only changing Q and R weights. It is retained as development work and is not identified as the source of the selected closing-controller results.
