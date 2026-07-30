"""
robustness.py — Parameter Plateau Test & Sensitivity Analysis.

Tujuan: Deteksi overfitting dengan memeriksa apakah best_params berada di
        "dataran stabil" (plateau) atau di "puncak terisolir" (overfit peak).

Konsep Parameter Plateau (2025/2026 best practice):
  - Overfit     : Profit hanya di num_std=2.0 tapi tidak di 1.9 atau 2.1 → RED FLAG
  - Robust      : Profit stabil di num_std 1.7–2.3 → "plateau" → GREEN FLAG
  - Plateau Score: 0-100, semakin tinggi semakin robust

Cara kerja:
  1. Ambil best_params dari IS grid search
  2. Variasikan setiap parameter ± beberapa step
  3. Run backtest di IS untuk setiap kombinasi variasi
  4. Hitung Sharpe untuk setiap variasi
  5. Plateau score = % dari variasi yang menghasilkan Sharpe > threshold

Output:
  - plateau_score     : float 0-100
  - plateau_data      : DataFrame dengan semua variasi dan Sharpe-nya
  - sensitivity_report: dict per parameter
  - Heatmap PNG       : charts/plateau_heatmap_*.png
"""

import itertools
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.engine_v2 import backtest_v2
from src.metrics import compute_metrics


# ------------------------------------------------------------------ #
# Parameter variation ranges
# ------------------------------------------------------------------ #
BB_V2_VARIATIONS = {
    # param_name : list of relative steps to add to best value
    "num_std":       [-0.5, -0.3, -0.1,  0.0,  0.1,  0.3,  0.5],
    "window":        [-5,   -3,   -1,    0,    1,    3,    5  ],
    "adx_threshold": [-5,   -3,   0,     3,    5              ],
    "vol_ratio_min": [-0.3, -0.1, 0.0,   0.1,  0.3           ],
    "atr_mult":      [-0.5, -0.3, 0.0,   0.3,  0.5           ],
}

# Constraints: nilai minimum yang valid
BB_V2_MIN = {
    "num_std":       0.5,
    "window":        5,
    "adx_threshold": 10,
    "vol_ratio_min": 0.5,
    "atr_mult":      0.5,
}


def _clamp_params(params: dict) -> dict:
    """Pastikan parameter tidak di bawah minimum valid."""
    out = params.copy()
    for k, minval in BB_V2_MIN.items():
        if k in out:
            out[k] = max(out[k], minval)
    return out


def parameter_plateau_test(
    is_data: pd.DataFrame,
    best_params: dict,
    signal_fn: Callable,
    capital: float,
    sharpe_threshold_ratio: float = 0.5,
    min_trades: int = 3,
) -> Dict:
    """
    Uji robustness parameter dengan memvariasikan nilai di sekitar best_params.

    Args:
        is_data              : In-Sample DataFrame.
        best_params          : dict parameter terbaik dari grid search.
        signal_fn            : Fungsi sinyal (add_bb_v2_signals atau sejenisnya).
        capital              : Modal awal.
        sharpe_threshold_ratio: Variasi dianggap "baik" jika Sharpe > best_sharpe * ratio.
                               Default 0.5 = setengah dari best Sharpe IS.
        min_trades           : Minimal trade agar variasi dianggap valid.

    Returns:
        dict dengan keys:
          - plateau_score    : float 0-100
          - best_sharpe      : Sharpe dari best_params
          - n_variations     : Total variasi yang ditest
          - n_above_threshold: Variasi yang Sharpe-nya > threshold
          - sensitivity      : dict per parameter (std dev Sharpe saat divarasi)
          - plateau_df       : DataFrame semua variasi + Sharpe
          - warning          : str atau None
    """
    # 1. Hitung Sharpe best_params sebagai baseline
    ENGINE_ONLY = {"atr_mult"}
    signal_p = {k: v for k, v in best_params.items() if k not in ENGINE_ONLY}
    engine_p = {k: v for k, v in best_params.items() if k in ENGINE_ONLY}

    try:
        df_best, em, xm = signal_fn(is_data, **signal_p)
        atr_mult = best_params.get("atr_mult", 2.0)
        df_best, trades = backtest_v2(df_best, em, xm, capital, atr_mult=atr_mult)
        m_best = compute_metrics(df_best, trades, capital)
        best_sharpe = m_best["sharpe"]
    except Exception as e:
        return {
            "plateau_score": 0,
            "best_sharpe": 0,
            "n_variations": 0,
            "n_above_threshold": 0,
            "sensitivity": {},
            "plateau_df": pd.DataFrame(),
            "warning": f"Error saat run best_params: {e}",
        }

    sharpe_threshold = best_sharpe * sharpe_threshold_ratio

    # 2. Variasikan setiap parameter yang ada di best_params
    records = []

    for param_name in best_params:
        if param_name not in BB_V2_VARIATIONS:
            continue

        variations = BB_V2_VARIATIONS[param_name]
        base_val   = best_params[param_name]

        for step in variations:
            varied_params = best_params.copy()
            varied_params[param_name] = round(base_val + step, 4)
            varied_params = _clamp_params(varied_params)

            # Pisahkan engine params dari signal params
            varied_signal_p = {k: v for k, v in varied_params.items() if k not in ENGINE_ONLY}
            varied_atr_mult = varied_params.get("atr_mult", 2.0)

            try:
                df_v, em_v, xm_v = signal_fn(is_data, **varied_signal_p)
                if len(df_v) < 50:
                    continue

                df_v, trades_v = backtest_v2(df_v, em_v, xm_v, capital, atr_mult=varied_atr_mult)
                m_v = compute_metrics(df_v, trades_v, capital)

                if m_v["n_trades"] < min_trades:
                    sharpe_v = np.nan
                else:
                    sharpe_v = m_v["sharpe"]

            except Exception:
                sharpe_v = np.nan

            records.append({
                "varied_param": param_name,
                "base_value":   base_val,
                "varied_value": varied_params[param_name],
                "step":         step,
                "sharpe":       sharpe_v,
                "is_best":      step == 0,
                "above_threshold": (
                    not np.isnan(sharpe_v) and sharpe_v >= sharpe_threshold
                ),
            })

    plateau_df = pd.DataFrame(records)

    if plateau_df.empty:
        return {
            "plateau_score": 0,
            "best_sharpe": best_sharpe,
            "n_variations": 0,
            "n_above_threshold": 0,
            "sensitivity": {},
            "plateau_df": plateau_df,
            "warning": "Tidak ada variasi yang berhasil dijalankan.",
        }

    # 3. Hitung plateau score
    valid_rows = plateau_df[~plateau_df["sharpe"].isna()]
    n_valid    = len(valid_rows)
    n_above    = valid_rows["above_threshold"].sum()
    plateau_score = (n_above / n_valid * 100) if n_valid > 0 else 0

    # 4. Sensitivity per parameter (std dev Sharpe saat divariasi)
    sensitivity = {}
    for param in plateau_df["varied_param"].unique():
        subset = plateau_df[
            (plateau_df["varied_param"] == param) & (~plateau_df["sharpe"].isna())
        ]["sharpe"]
        sensitivity[param] = {
            "sharpe_std":  float(subset.std()),
            "sharpe_min":  float(subset.min()),
            "sharpe_max":  float(subset.max()),
            "sharpe_mean": float(subset.mean()),
        }

    # 5. Warning jika plateau score rendah
    if plateau_score < 40:
        warning = (
            f"HIGH OVERFITTING RISK — Plateau score {plateau_score:.0f}/100. "
            f"Hanya {n_above}/{n_valid} variasi yang Sharpe-nya > {sharpe_threshold:.2f}. "
            f"Parameter sangat sensitif — strategi mungkin tidak robust."
        )
    elif plateau_score < 70:
        warning = (
            f"MODERATE RISK — Plateau score {plateau_score:.0f}/100. "
            f"Parameter cukup robust tapi perlu lebih banyak data untuk konfirmasi."
        )
    else:
        warning = None

    return {
        "plateau_score":      plateau_score,
        "best_sharpe":        best_sharpe,
        "n_variations":       n_valid,
        "n_above_threshold":  int(n_above),
        "sharpe_threshold":   sharpe_threshold,
        "sensitivity":        sensitivity,
        "plateau_df":         plateau_df,
        "warning":            warning,
    }


def plot_sensitivity_heatmap(
    plateau_data: Dict,
    best_params: dict,
    ticker: str,
    timestamp: str,
    charts_dir: str,
) -> Optional[Path]:
    """
    Plot sensitivitas Sharpe per parameter sebagai bar chart grouped.

    Args:
        plateau_data: Output dari parameter_plateau_test().
        best_params : Parameter terbaik IS.
        ticker      : Nama ticker untuk label.
        timestamp   : Session timestamp untuk nama file unik.
        charts_dir  : Folder output chart.

    Returns:
        Path ke file PNG, atau None jika tidak ada data.
    """
    df = plateau_data.get("plateau_df")
    if df is None or df.empty:
        return None

    charts_path = Path(charts_dir)
    charts_path.mkdir(parents=True, exist_ok=True)

    params_varied = df["varied_param"].unique()
    n_params      = len(params_varied)
    if n_params == 0:
        return None

    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 5), sharey=False)
    if n_params == 1:
        axes = [axes]

    score   = plateau_data["plateau_score"]
    b_sharpe = plateau_data["best_sharpe"]
    thresh  = plateau_data.get("sharpe_threshold", b_sharpe * 0.5)

    for ax, param in zip(axes, params_varied):
        subset = df[df["varied_param"] == param].sort_values("varied_value")
        vals   = subset["varied_value"].values
        sharpes = subset["sharpe"].values
        colors  = []
        for v, s in zip(vals, sharpes):
            if np.isnan(s):
                colors.append("#d1d5db")
            elif s >= thresh:
                colors.append("#16a34a")
            else:
                colors.append("#dc2626")

        bars = ax.bar(range(len(vals)), sharpes, color=colors, alpha=0.85, edgecolor="white")

        # Label nilai parameter di bawah bar
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([f"{v:.2g}" for v in vals], rotation=45, fontsize=8)

        # Mark best param
        best_val = best_params.get(param)
        if best_val is not None:
            for i, v in enumerate(vals):
                if abs(v - best_val) < 1e-6:
                    ax.get_xticklabels()[i].set_fontweight("bold")
                    ax.get_xticklabels()[i].set_color("#2563eb")

        ax.axhline(thresh, color="#f59e0b", linestyle="--", linewidth=1.2,
                   label=f"Threshold ({thresh:.2f})")
        ax.axhline(0, color="#374151", linewidth=0.7, alpha=0.5)
        ax.set_title(f"{param}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Sharpe (IS)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7)

    ticker_slug = ticker.replace(".", "_")
    fig.suptitle(
        f"Parameter Plateau Test — {ticker}\n"
        f"Plateau Score: {score:.0f}/100  |  Best IS Sharpe: {b_sharpe:.2f}  "
        f"(hijau = > threshold {thresh:.2f})",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()

    out_path = charts_path / f"plateau_{ticker_slug}_{timestamp}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path
