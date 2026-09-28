# SAC utility tests

`test_figure_output_layout.py` verifies the retained figure destinations and confirms that removed PDF outputs and obsolete state-evolution plotters are not restored.

Phase-specific controller and evaluation tests remain beside their corresponding closing or final-approach implementations.

Run this test from the repository root:

```text
python -m pytest src/controllers/SAC-3DoF/tests -q
```
