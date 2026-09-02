"""V5 label candidate families (pure functions, no leakage by construction).

All functions take a per-symbol OHLC DataFrame sorted by date ascending and
return a Series aligned to the same index. Row t's label only uses rows > t.

Label families (see memory v5-label-campaign):
- cur:        T close -> T+1+h close      (V4 status quo, control)
- open_exec:  T+1 open -> T+1+h close     (executable entry)
- mfr:        T+1 open -> max high over [t+1 .. t+1+h]  (max forward return,
              sniper-style target: best achievable exit inside the window)
"""

from __future__ import annotations

import pandas as pd


def cur_return(ohlc: pd.DataFrame, h: int) -> pd.Series:
    """Close-to-close: buy at T close, sell at T+1+h close."""
    return ohlc["close"].shift(-(h + 1)) / ohlc["close"] - 1


def open_exec_return(ohlc: pd.DataFrame, h: int) -> pd.Series:
    """Executable: buy at T+1 open, sell at T+1+h close."""
    return ohlc["close"].shift(-(h + 1)) / ohlc["open"].shift(-1) - 1


def max_forward_return(ohlc: pd.DataFrame, h: int) -> pd.Series:
    """Max forward return: buy at T+1 open, best high over [t+1 .. t+1+h].

    Computed with a reversed rolling max so the window never includes the
    signal day t itself:

        reversed_max[t] = max(high[t-h .. t])      (rolling on reversed series)
        window_max[t]   = reversed_max reversed, shifted -1
                        = max(high[t+1 .. t+1+h])
    """
    reversed_max = (
        ohlc["high"].iloc[::-1].rolling(h + 1, min_periods=h + 1).max()
    )
    s = reversed_max.iloc[::-1]
    window_max = s.shift(-1)
    return window_max / ohlc["open"].shift(-1) - 1
