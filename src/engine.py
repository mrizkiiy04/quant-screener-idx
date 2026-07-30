"""
engine.py — Backtest execution engine (Golden Strategy / ATR Trailing Stop).

Fee & cost model:
  - Lot size  : 100 lembar
  - Fee beli  : 0.19% (broker IDX tipikal)
  - Fee jual  : 0.29% (termasuk pajak)
  - Slippage  : 0.05% per sisi (market impact estimasi)

ATR Trailing Stop Logic:
  - Saat entry: stop_level = entry_price - (ATR_at_entry * atr_mult)
  - Setiap bar berikutnya: stop_level = max(stop_level, close - ATR_current * atr_mult)
  - Posisi di-close jika harga < stop_level
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

# ------------------------------------------------------------------ #
# Konstanta biaya — IDX
# ------------------------------------------------------------------ #
LOT_SIZE = 100       # 1 lot = 100 lembar
FEE_BUY  = 0.0019   # 0.19%
FEE_SELL = 0.0029   # 0.29% (termasuk PPh)
SLIPPAGE = 0.0005   # 0.05% per transaksi


# ------------------------------------------------------------------ #
# Leakage detection
# ------------------------------------------------------------------ #
def check_leakage(
    df: pd.DataFrame,
    entry_mask: pd.Series,
    exit_mask: pd.Series,
) -> List[str]:
    flags = []
    if not df.index.equals(entry_mask.index):
        flags.append("[LEAK] entry_mask index mismatch!")
    if not df.index.equals(exit_mask.index):
        flags.append("[LEAK] exit_mask index mismatch!")
    if entry_mask.isna().any():
        flags.append("[WARN] entry_mask contains NaN (potensi ffill leakage).")
    if exit_mask.isna().any():
        flags.append("[WARN] exit_mask contains NaN.")
    return flags


# ------------------------------------------------------------------ #
# Backtest Engine (Trailing Stop)
# ------------------------------------------------------------------ #
def run_backtest(
    df: pd.DataFrame,
    entry_mask: pd.Series,
    exit_mask: pd.Series,
    capital: float,
    atr_mult: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtest engine v2 dengan ATR Trailing Stop Loss.
    """
    df = df.copy()

    if "ATR" not in df.columns:
        raise ValueError("Kolom 'ATR' tidak ditemukan.")

    cash        = capital
    shares      = 0
    entry_price = None
    entry_date  = None
    stop_level  = None
    trades      = []
    equity_curve = []

    for date, row in df.iterrows():
        close = row["Close"]
        atr   = row["ATR"]

        if shares == 0:
            # --- Cek Entry ---
            if entry_mask.loc[date]:
                buy_price    = close * (1 + SLIPPAGE)
                cost_per_lot = buy_price * LOT_SIZE * (1 + FEE_BUY)
                n_lots       = int(cash // cost_per_lot)

                if n_lots > 0:
                    shares      = n_lots * LOT_SIZE
                    entry_price = buy_price
                    cash       -= shares * buy_price * (1 + FEE_BUY)
                    entry_date  = date
                    stop_level  = buy_price - (atr * atr_mult)

        else:
            # --- Update Trailing Stop ---
            new_stop = close - (atr * atr_mult)
            stop_level = max(stop_level, new_stop)

            # --- Cek Exit ---
            hit_stop   = close <= stop_level
            hit_signal = exit_mask.loc[date]

            if hit_stop or hit_signal:
                sell_price = close * (1 - SLIPPAGE)
                revenue    = shares * sell_price * (1 - FEE_SELL)
                
                trade_return = (revenue / (shares * entry_price * (1 + FEE_BUY))) - 1
                cash += revenue

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": sell_price,
                    "shares": shares,
                    "return": trade_return,
                    "exit_reason": "stop_loss" if hit_stop else "signal"
                })

                shares      = 0
                entry_price = None
                entry_date  = None
                stop_level  = None

        # Catat equity
        current_value = cash + (shares * close)
        equity_curve.append(current_value)

    df["Equity"] = equity_curve
    trades_df = pd.DataFrame(trades)

    return df, trades_df
