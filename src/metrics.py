"""
metrics.py — Perhitungan metrik performa dan konsistensi IS/OOS.

Identik dengan logika di prd.md, ditambah:
  - profit_factor (gross profit / gross loss)
  - avg_win / avg_loss per trade
  - calmar_ratio (CAGR / |max_dd|)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

MIN_TRADES = 5   # Minimum trade di OOS agar verdict bisa diberikan


# ------------------------------------------------------------------ #
# Compute metrics
# ------------------------------------------------------------------ #
def compute_metrics(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    capital: float,
) -> dict:
    """
    Hitung semua metrik performa dari hasil backtest.

    Args:
        df: DataFrame dengan kolom 'Equity'.
        trades_df: DataFrame trades (output dari engine.backtest).
        capital: Modal awal (Rp).

    Returns:
        dict berisi semua metrik.
    """
    final_equity = df["Equity"].iloc[-1]
    total_return = (final_equity / capital - 1) * 100

    n_days  = (df.index[-1] - df.index[0]).days
    n_years = max(n_days / 365.25, 0.01)
    cagr    = ((final_equity / capital) ** (1 / n_years) - 1) * 100

    daily_ret = df["Equity"].pct_change().dropna()
    if daily_ret.std() > 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    dd     = (df["Equity"] / df["Equity"].cummax() - 1) * 100
    max_dd = dd.min()

    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    if not trades_df.empty:
        wins   = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        win_rate = len(wins) / len(trades_df) * 100
        avg_win  = float(wins["pnl"].mean())   if not wins.empty   else 0.0
        avg_loss = float(losses["pnl"].mean()) if not losses.empty else 0.0

        gross_profit = wins["pnl"].sum()
        gross_loss   = abs(losses["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_hold_days = float(trades_df["hold_days"].mean()) if "hold_days" in trades_df.columns else 0.0
    else:
        win_rate      = 0.0
        avg_win       = 0.0
        avg_loss      = 0.0
        profit_factor = 0.0
        avg_hold_days = 0.0

    return {
        "final_equity":  final_equity,
        "total_return":  total_return,
        "cagr":          cagr,
        "sharpe":        sharpe,
        "max_dd":        max_dd,
        "calmar":        calmar,
        "win_rate":      win_rate,
        "n_trades":      len(trades_df),
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "profit_factor": profit_factor,
        "avg_hold_days": avg_hold_days,
    }


# ------------------------------------------------------------------ #
# Consistency verdict (IS → OOS degradation analysis)
# ------------------------------------------------------------------ #
def consistency_verdict(
    is_m: dict,
    oos_m: dict,
    min_trades: int = MIN_TRADES,
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Evaluasi konsistensi performa dari In-Sample ke Out-of-Sample.

    Tujuan: Deteksi apakah strategi benar-benar punya edge atau hanya overfit ke data IS.

    Logic (identik dengan prd.md):
      - Jika OOS < min_trades: INCONCLUSIVE (tidak bisa disimpulkan)
      - Degradasi Sharpe dihitung: (IS_sharpe - OOS_sharpe) / IS_sharpe * 100
      - < 20% degradasi → ROBUST
      - 20-50% → WASPADA
      - > 50% tapi OOS masih profit → LEMAH TAPI MASIH PROFIT
      - OOS rugi/flat → GAGAL (kemungkinan overfit)

    Returns:
        Tuple (score [0-100 atau None], degradation [% atau None], verdict [str])
    """
    # Kasus khusus: sinyal OOS terlalu sedikit
    if oos_m["n_trades"] < min_trades:
        verdict = (
            f"INCONCLUSIVE — OOS cuma {oos_m['n_trades']} trade (min {min_trades}). "
            f"Bukan berarti strategi gagal; sinyal entry memang jarang/tidak muncul di "
            f"periode OOS ini. Coba perpanjang periode data atau longgarkan parameter."
        )
        return None, None, verdict

    is_sharpe  = max(is_m["sharpe"], 0.01)   # Hindari div by zero
    degradation = (is_sharpe - oos_m["sharpe"]) / abs(is_sharpe) * 100

    still_profitable = (oos_m["sharpe"] > 0) and (oos_m["total_return"] > 0)
    same_direction   = (is_m["total_return"] > 0) == (oos_m["total_return"] > 0)

    score = max(0, 100 - max(degradation, 0))

    if not still_profitable or not same_direction:
        verdict = (
            "GAGAL — OOS rugi/flat (Sharpe atau return <= 0), "
            "strategi ini kemungkinan besar overfit ke data IS"
        )
    elif degradation < 20:
        verdict = (
            "ROBUST — performa OOS konsisten dgn IS, edge kemungkinan nyata"
        )
    elif degradation < 50:
        verdict = (
            "WASPADA — degradasi cukup besar, tapi masih net profit; "
            "perlu data lebih panjang utk yakin"
        )
    else:
        verdict = (
            "LEMAH TAPI MASIH PROFIT — degradasi besar (kemungkinan winner's curse "
            "dari grid search), tapi OOS tidak rugi. Jangan andalkan ukuran return IS, "
            "tapi strategi belum tentu tidak berguna"
        )

    return score, degradation, verdict
