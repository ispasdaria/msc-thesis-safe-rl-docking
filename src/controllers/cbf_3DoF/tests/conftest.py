import os
import sys


SRC_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
