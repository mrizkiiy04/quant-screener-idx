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

from src.indicators import add_signals
from src.engine import run_backtest
from src.metrics import compute_metrics

MIN_TRADES = 5


def grid_search_signals(is_data: pd.DataFrame, capital: float, verbose: bool = False) -> tuple:
    """
    Grid search untuk parameter The Golden Strategy (BB %B + ADX + Volume).

    Menggunakan Baseline Grid (Dipersempit untuk mencegah overfitting).
    """
    param_grid = {
        "num_std": [1.5, 2.0, 2.5],
        "window": [15, 20, 25],
        "adx_threshold": [20, 25],
        "vol_ratio_min": [1.0, 1.2, 1.5],
        "atr_mult": [1.5, 2.0]
    }

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[k] for k in keys)))

    best_params = None
    best_sharpe = -999.0
    best_result = None
    all_results = []

    if verbose:
        print(f"[{is_data.index[0].date()} -> {is_data.index[-1].date()}] "
              f"Searching {len(combinations)} parameter combos...")

    for combo in combinations:
        params = dict(zip(keys, combo))
        
        # Pisahkan params untuk sinyal vs engine (trailing stop)
        signal_params = {k: v for k, v in params.items() if k != "atr_mult"}
        
        df, em, xm = add_signals(is_data, **signal_params)
        df_res, trades_df = run_backtest(df, em, xm, capital, atr_mult=params["atr_mult"])
        metrics = compute_metrics(df_res, trades_df, capital)
        
        metrics.update(params)
        all_results.append(metrics)

        if metrics["n_trades"] >= MIN_TRADES:
            if metrics["sharpe"] > best_sharpe:
                best_sharpe = metrics["sharpe"]
                best_params = params
                best_result = metrics

    if best_params is None:
        return None, None, pd.DataFrame(all_results)

    return best_params, best_result, pd.DataFrame(all_results)
