"""Tests for v3_pipeline/src/label_candidates.py.

Anti-leakage invariants asserted here:
1. Window excludes the signal day: changing row t's OHLC must not change
   label at row t.
2. Spike capture: a high spike inside [t+1, t+1+h] is reflected in mfr.
3. Boundary: label at t uses exactly rows t+1..t+1+h, not t+h+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v3_pipeline.src.label_candidates import (
    cur_return,
    max_forward_return,
    open_exec_return,
)


def _make_ohlc(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.01, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_open_exec_basic():
    ohlc = _make_ohlc()
    h = 3
    lbl = open_exec_return(ohlc, h)
    t = 10
    expected = ohlc["close"].iloc[t + 1 + h] / ohlc["open"].iloc[t + 1] - 1
    assert lbl.iloc[t] == pytest.approx(expected)


def test_cur_basic():
    ohlc = _make_ohlc()
    h = 3
    lbl = cur_return(ohlc, h)
    t = 10
    expected = ohlc["close"].iloc[t + 1 + h] / ohlc["close"].iloc[t] - 1
    assert lbl.iloc[t] == pytest.approx(expected)


def test_mfr_window_excludes_signal_day():
    """Mutating row t's high/open/close must not move mfr/open_exec at t."""
    ohlc = _make_ohlc()
    h = 5
    t = 10
    base = max_forward_return(ohlc, h).iloc[t]
    base_oe = open_exec_return(ohlc, h).iloc[t]

    mutated = ohlc.copy()
    mutated.loc[t, "high"] = ohlc["high"].iloc[t] * 3.0
    mutated.loc[t, "open"] = ohlc["open"].iloc[t] * 0.5
    mutated.loc[t, "close"] = ohlc["close"].iloc[t] * 2.0
    assert max_forward_return(mutated, h).iloc[t] == pytest.approx(base)
    assert open_exec_return(mutated, h).iloc[t] == pytest.approx(base_oe)


def test_mfr_captures_spike_inside_window():
    """A high spike at t+3 (inside [t+1, t+1+h]) must appear in mfr."""
    ohlc = _make_ohlc()
    h = 5
    t = 10
    spiked = ohlc.copy()
    spiked.loc[t + 3, "high"] = ohlc["high"].max() * 2
    lbl = max_forward_return(spiked, h)
    expected = spiked["high"].iloc[t + 3] / spiked["open"].iloc[t + 1] - 1
    assert lbl.iloc[t] == pytest.approx(expected)


def test_mfr_ignores_spike_outside_window():
    """A spike at t+1+h+1 (just past the window) must NOT appear in mfr."""
    ohlc = _make_ohlc()
    h = 5
    t = 10
    spiked = ohlc.copy()
    spiked.loc[t + 1 + h + 1, "high"] = ohlc["high"].max() * 2
    lbl_base = max_forward_return(ohlc, h).iloc[t]
    lbl_spiked = max_forward_return(spiked, h).iloc[t]
    assert lbl_spiked == pytest.approx(lbl_base)
