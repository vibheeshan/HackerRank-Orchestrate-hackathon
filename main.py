#!/usr/bin/env python3
"""
Evaluation Runner (code/evaluation/main.py)
Evaluates generated output.csv against dataset/sample_requests.csv.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from evaluation_harness import main as run_evaluation

if __name__ == '__main__':
    sys.exit(run_evaluation())
