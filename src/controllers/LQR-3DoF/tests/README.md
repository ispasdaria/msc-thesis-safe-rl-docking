# LQR Controller Tests

The tests are grouped by mission phase and scope.

- Unit tests verify individual matrices, mappings, transformations, validation,
  and error calculations.
- Integration tests verify interactions between the environment, model,
  controller synthesis, and controller execution.
- Functional tests execute controller rollouts and check deterministic,
  termination, success-rate, and safety properties.
- Plotter tests verify cached-history alignment, reference construction, and
  full-mission geometry transformations.

Run the complete LQR test suite from the repository root:

```text
python -m pytest src/controllers/LQR-3DoF/tests -q
```

Run the faster tests without simulation-heavy rollout campaigns:

```text
python -m pytest src/controllers/LQR-3DoF/tests -m "not slow" -q
```

Run only the simulation-heavy tests:

```text
python -m pytest src/controllers/LQR-3DoF/tests -m slow -q
```
