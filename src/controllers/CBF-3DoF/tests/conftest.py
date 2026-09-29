import os
import sys


CONTROLLER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.abspath(os.path.join(CONTROLLER_PATH, "..", ".."))
EVALUATIONS_PATH = os.path.join(CONTROLLER_PATH, "evaluations")
TUNING_PATH = os.path.join(CONTROLLER_PATH, "tuning")
for path in (CONTROLLER_PATH, EVALUATIONS_PATH, TUNING_PATH, SRC_PATH):
    if path not in sys.path:
        sys.path.insert(0, path)
