"""
grid_search.py — Exhaustive grid search di In-Sample period ONLY.

ATURAN DATA LEAKAGE KRITIS:
    - Fungsi ini HANYA menerima is_data (bukan full data atau oos_data).
    - Parameter yang dioptimasi di sini akan dikunci sebelum dijalankan ke OOS.
    - Grid search di OOS = data leakage / snooping bias = TERLARANG.
"""

import itertools
from typing import List

import pandas as pd

from src.indicators import add_rsi2_signals, add_bollinger_signals
from src.engine import backtest
from src.metrics import compute_metrics

MIN_TRADES = 5


# ------------------------------------------------------------------ #
# RSI(2) Grid Search
# ------------------------------------------------------------------ #
def grid_search_rsi2(
    is_data: pd.DataFrame,
    capital: float,
    min_trades: int = MIN_TRADES,
) -> List[dict]:
    """
    Grid search parameter RSI(2) di In-Sample period saja.

    Parameter grid (identik dengan prd.md):
      - rsi_entry : [5, 10, 15, 20]
      - rsi_exit  : [50, 60, 70]
      - sma_exit  : [5, 10]

    Total kombinasi: 4 × 3 × 2 = 24

    Returns:
        List of dicts sorted by Sharpe (best first).
        Hanya kombinasi dengan >= min_trades yang dimasukkan.
    """
    results = []

    for entry, exit_, sma_exit in itertools.product(
        [5, 10, 15, 20],
        [50, 60, 70],
        [5, 10]
    ):
        df, em, xm = add_rsi2_signals(is_data, entry, exit_, sma_exit=sma_exit)
        if len(df) < 50:
            continue

        df, trades = backtest(df, em, xm, capital)
        m = compute_metrics(df, trades, capital)

        if m["n_trades"] < min_trades:
            continue

        results.append({
            "params": {
                "rsi_entry": entry,
                "rsi_exit":  exit_,
                "sma_exit":  sma_exit,
            },
            **m
        })

    # Sort by Sharpe ratio descending
    results.sort(key=lambda r: r["sharpe"], reverse=True)
    return results


# ------------------------------------------------------------------ #
# Bollinger Bands Grid Search
# ------------------------------------------------------------------ #
def grid_search_bb(
    is_data: pd.DataFrame,
    capital: float,
    min_trades: int = MIN_TRADES,
) -> List[dict]:
    """
    Grid search parameter Bollinger Bands di In-Sample period saja.

    Parameter grid (identik dengan prd.md):
      - num_std : [1.5, 2.0, 2.5]
      - window  : [10, 20, 30]

    Total kombinasi: 3 × 3 = 9

    Returns:
        List of dicts sorted by Sharpe (best first).
        Hanya kombinasi dengan >= min_trades yang dimasukkan.
    """
    results = []

    for num_std, window in itertools.product(
        [1.5, 2.0, 2.5],
        [10, 20, 30]
    ):
        df, em, xm = add_bollinger_signals(is_data, num_std, window)
        if len(df) < 50:
            continue

        df, trades = backtest(df, em, xm, capital)
        m = compute_metrics(df, trades, capital)

        if m["n_trades"] < min_trades:
            continue

        results.append({
            "params": {
                "num_std": num_std,
                "window":  window,
            },
            **m
        })

    results.sort(key=lambda r: r["sharpe"], reverse=True)
    return results
