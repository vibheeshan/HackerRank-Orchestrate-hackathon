#!/usr/bin/env python3
"""
Buy or Wait - AI Financial Agent
HackerRank Orchestrate September 2026 Challenge

Root Entry Point (main.py)
Delegates execution to code/main.py pipeline.
"""

import sys
from pathlib import Path

# Add project root to python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from code.main import run_pipeline

if __name__ == '__main__':
    sys.exit(run_pipeline())
