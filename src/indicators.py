"""
indicators.py — Perhitungan indikator teknikal dan sinyal entry/exit.

Versi Final: Bollinger Bands (%B) + ADX regime filter + Volume confirmation + ATR.

PENTING (Data Leakage):
    - Semua indikator menggunakan rolling window yang hanya melihat ke belakang (look-back).
    - TIDAK ada shift(-1) atau forward-fill yang bisa menimbulkan look-ahead bias.
    - dropna() dipakai untuk membuang baris tanpa data cukup (bukan ffill/bfill).
"""

import pandas as pd
import numpy as np


# ------------------------------------------------------------------ #
# ADX — Average Directional Index  (regime filter)
# ------------------------------------------------------------------ #
def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Hitung ADX + DI+/DI- dari OHLC data.

    ADX < 20-25 = pasar range-bound (ideal untuk mean reversion)
    ADX > 25    = pasar trending   (hindari fading / mean reversion)

    Args:
        df: DataFrame dengan kolom High, Low, Close.
        period: Smoothing period (default 14, standar Wilder).

    Returns:
        DataFrame dengan kolom tambahan: TR, DM_plus, DM_minus,
        ATR, DI_plus, DI_minus, DX, ADX.
    """
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    prev_c = c.shift(1)

    # True Range
    tr1 = h - l
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    dm_plus  = np.where((h - h.shift(1)) > (l.shift(1) - l), np.maximum(h - h.shift(1), 0), 0)
    dm_minus = np.where((l.shift(1) - l) > (h - h.shift(1)), np.maximum(l.shift(1) - l, 0), 0)

    dm_plus_s  = pd.Series(dm_plus,  index=df.index)
    dm_minus_s = pd.Series(dm_minus, index=df.index)

    # Wilder smoothing
    atr_s    = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    dip_s    = dm_plus_s.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    dim_s    = dm_minus_s.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    di_plus  = (dip_s / atr_s) * 100
    di_minus = (dim_s / atr_s) * 100

    dx  = ((di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    result = df.copy()
    result["ATR"]      = atr_s
    result["DI_plus"]  = di_plus
    result["DI_minus"] = di_minus
    result["ADX"]      = adx
    return result


# ------------------------------------------------------------------ #
# ATR — Average True Range  (untuk trailing stop)
# ------------------------------------------------------------------ #
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Hitung ATR (Average True Range) untuk dynamic stop loss.

    Returns:
        Series ATR (dalam satuan harga, Rp).
    """
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    prev_c = c.shift(1)

    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


# ------------------------------------------------------------------ #
# Bollinger Bands — %B + ADX + Volume + ATR (Golden Strategy)
# ------------------------------------------------------------------ #
def add_signals(
    data: pd.DataFrame,
    num_std: float = 2.0,
    window: int = 20,
    adx_threshold: float = 25.0,
    vol_ratio_min: float = 1.2,
    percent_b_entry: float = 0.05,
    percent_b_exit: float = 0.50,
    adx_period: int = 14,
    atr_period: int = 14,
    vol_ma_period: int = 20,
) -> tuple:
    """
    Bollinger Bands (Golden Strategy) — Strategi mean reversion tangguh.

    Logika Sinyal:
    1. %B (Percent Bollinger) sebagai entry signal — masuk saat sangat oversold
    2. ADX regime filter — hanya masuk saat pasar range-bound (ADX < threshold)
    3. Volume ratio confirmation — masuk hanya saat ada tekanan jual BERVOLUME
    4. BandWidth — deteksi Bollinger Squeeze (hindari entry saat squeeze)
    5. ATR trailing stop — ditangani oleh engine eksekusi
    """
    required = ["High", "Low", "Close", "Volume"]
    missing  = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Dibutuhkan kolom: {missing}")

    df = data.copy()

    # --- Bollinger Bands dasar ---
    df["BB_SMA"]   = df["Close"].rolling(window).mean()
    df["BB_STD"]   = df["Close"].rolling(window).std()
    df["BB_Upper"] = df["BB_SMA"] + num_std * df["BB_STD"]
    df["BB_Lower"] = df["BB_SMA"] - num_std * df["BB_STD"]

    # --- %B (Percent Bollinger) ---
    band_width_price = df["BB_Upper"] - df["BB_Lower"]
    df["Pct_B"]      = (df["Close"] - df["BB_Lower"]) / band_width_price.replace(0, np.nan)

    # --- BandWidth (deteksi Bollinger Squeeze) ---
    df["BandWidth"] = band_width_price / df["BB_SMA"] * 100
    bw_ma = df["BandWidth"].rolling(window).mean()
    df["BW_Squeeze"] = df["BandWidth"] < (bw_ma * 0.75)

    # --- ADX regime filter ---
    df_adx    = compute_adx(df, period=adx_period)
    df["ADX"]  = df_adx["ADX"]
    df["ATR"]  = df_adx["ATR"]

    # --- Volume Ratio (VR20) ---
    df["Vol_MA"]  = df["Volume"].rolling(vol_ma_period).mean()
    df["Vol_Ratio"] = df["Volume"] / df["Vol_MA"]

    # --- Drop NaN ---
    df = df.dropna(subset=[
        "BB_SMA", "BB_STD", "Pct_B", "BandWidth",
        "ADX", "ATR", "Vol_MA", "Vol_Ratio"
    ]).copy()

    # ================================================================
    # ENTRY CONDITIONS
    # ================================================================
    cond_oversold  = df["Pct_B"] < percent_b_entry
    cond_regime    = df["ADX"] < adx_threshold
    cond_volume    = df["Vol_Ratio"] >= vol_ratio_min
    cond_no_squeeze = ~df["BW_Squeeze"]

    entry_mask = cond_oversold & cond_regime & cond_volume & cond_no_squeeze

    # ================================================================
    # EXIT CONDITIONS
    # ================================================================
    exit_mask = df["Pct_B"] >= percent_b_exit

    return df, entry_mask, exit_mask
