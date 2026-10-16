"""Compatibility entry point for the deterministic evaluator.

Run python win_yuan/regression.py new --json current.json from the repository.
Relative report paths are resolved inside repository reports/.
Photo evaluation requires the selected data directory and its truth TSV.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from win_yuan.evaluation import FINGERS, OLD_TRUTH, load_truth
from win_yuan.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main())
