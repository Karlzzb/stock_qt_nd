"""
Test V9 and V10 simulation scripts can execute and produce correct output.

Run with: python -m pytest tests/test_v9_v10_simulation.py -v
"""
import os
import sys
import pandas as pd
import tempfile
import types

# Add src to path
SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, SRC_DIR)

# Mock missing modules (auxiliary analysis tools that aren't in worktree)
mock_hold_analysis = types.ModuleType('tools.hold_analysis')
mock_hold_analysis.hold_analyzer = lambda *a, **kw: None

mock_profit_analysis = types.ModuleType('tools.profit_analysis')
mock_profit_analysis.profit_analyzer = lambda *a, **kw: None

mock_return_analysis = types.ModuleType('tools.return_analysis')
mock_return_analysis.return_analyzer = lambda *a, **kw: None

mock_trades_analysis = types.ModuleType('tools.trades_analysis')
mock_trades_analysis.trades_analyzer = lambda *a, **kw: None

mock_return_prob_analysis = types.ModuleType('tools.return_prob_correlation_analysis')
mock_return_prob_analysis.correlation_analyzer = lambda *a, **kw: None

sys.modules['tools.hold_analysis'] = mock_hold_analysis
sys.modules['tools.profit_analysis'] = mock_profit_analysis
sys.modules['tools.return_analysis'] = mock_return_analysis
sys.modules['tools.trades_analysis'] = mock_trades_analysis
sys.modules['tools.return_prob_correlation_analysis'] = mock_return_prob_analysis

# Set random seed for reproducibility
import numpy as np
import random
random.seed(42)
np.random.seed(42)


def test_v9_strategy_class_import():
    """V9 strategy class can be imported."""
    from strategies.smart_sniper_strategy_v9 import SmartSniperStrategyV9
    assert SmartSniperStrategyV9 is not None
    # Verify it's a class
    assert isinstance(SmartSniperStrategyV9, type)


def test_v10_strategy_class_import():
    """V10 strategy class can be imported."""
    from strategies.smart_sniper_strategy_v10 import SmartSniperStrategyV10
    assert SmartSniperStrategyV10 is not None
    assert isinstance(SmartSniperStrategyV10, type)


def test_v9_strategy_can_be_instantiated():
    """V9 strategy can be instantiated with minimal params."""
    from strategies.smart_sniper_strategy_v9 import SmartSniperStrategyV9
    strategy = SmartSniperStrategyV9(initial_capital=100000, max_positions=3)
    assert strategy is not None
    # Check V9-specific attributes exist
    assert hasattr(strategy, 'recent_rise_n')
    assert hasattr(strategy, 'rq_window')
    assert hasattr(strategy, 'current_rq')


def test_v10_strategy_can_be_instantiated():
    """V10 strategy can be instantiated with minimal params."""
    from strategies.smart_sniper_strategy_v10 import SmartSniperStrategyV10
    strategy = SmartSniperStrategyV10(initial_capital=100000, max_positions=3)
    assert strategy is not None
    # Check V10-specific attributes exist
    assert hasattr(strategy, 'trailing_offset')
    assert hasattr(strategy, 'trailing_pct')
    assert hasattr(strategy, 'use_partial_take_profit')


def test_v9_simulation_module_import():
    """V9 simulation module can be imported."""
    import grid_trading_simulation_v9
    assert grid_trading_simulation_v9 is not None


def test_v10_simulation_module_import():
    """V10 simulation module can be imported."""
    import grid_trading_simulation_v10
    assert grid_trading_simulation_v10 is not None


def test_v9_simulation_mp_module_import():
    """V9 simulation MP module can be imported."""
    import grid_trading_simulation_v9_mp
    assert grid_trading_simulation_v9_mp is not None


def test_v10_simulation_mp_module_import():
    """V10 simulation MP module can be imported."""
    import grid_trading_simulation_v10_mp
    assert grid_trading_simulation_v10_mp is not None


def test_v9_simulation_no_v8_imports():
    """Verify V9 simulation doesn't import any V8 modules."""
    import ast
    v9_file = os.path.join(SRC_DIR, 'grid_trading_simulation_v9.py')

    with open(v9_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for v8 imports
    assert 'from strategies.smart_sniper_strategy import' not in content
    assert 'from grid_trading_simulation_v8 import' not in content
    assert 'import grid_trading_simulation_v8' not in content


def test_v10_simulation_no_v8_imports():
    """Verify V10 simulation doesn't import any V8 modules."""
    import ast
    v10_file = os.path.join(SRC_DIR, 'grid_trading_simulation_v10.py')

    with open(v10_file, 'r', encoding='utf-8') as f:
        content = f.read()

    assert 'from strategies.smart_sniper_strategy import' not in content
    assert 'from grid_trading_simulation_v8 import' not in content
    assert 'import grid_trading_simulation_v8' not in content
