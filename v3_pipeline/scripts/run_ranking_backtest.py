#!/usr/bin/env python3 -u
"""
Run ranking strategy backtest for Stage 3.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import and run the backtest module
from v3_pipeline.backtest.ranking_strategy import main

if __name__ == "__main__":
    main()
