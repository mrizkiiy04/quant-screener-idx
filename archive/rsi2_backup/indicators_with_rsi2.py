"""
indicators.py — Perhitungan indikator teknikal dan sinyal entry/exit.

PENTING (Data Leakage):
    - Semua indikator menggunakan rolling window yang hanya melihat ke belakang (look-back).
    - TIDAK ada shift(-1) atau forward-fill yang bisa menimbulkan look-ahead bias.
    - dropna() dipakai untuk membuang baris tanpa data cukup (bukan ffill/bfill).
    - Fungsi ini identik dengan logika di prd.md, hanya direfaktor menjadi modul terpisah.
"""

import pandas as pd
import numpy as np


# ------------------------------------------------------------------ #
# RSI (EMA-smoothed, Wilder method)
# ------------------------------------------------------------------ #
def compute_rsi(close: pd.Series, length: int = 2) -> pd.Series:
    """
    Hitung RSI menggunakan EMA (Wilder smoothing), identik dengan prd.md.

    Args:
        close: Series harga penutupan.
        length: Periode RSI (default 2 untuk RSI(2) mean reversion).

    Returns:
        Series RSI bernilai 0-100.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Ketika avg_loss == 0 (harga naik terus), RSI = 100
    rsi[avg_loss == 0] = 100

    return rsi


# ------------------------------------------------------------------ #
# RSI(2) Mean Reversion Signals
# ------------------------------------------------------------------ #
def add_rsi2_signals(
    data: pd.DataFrame,
    rsi_entry: float,
    rsi_exit: float,
    sma_trend: int = 200,
    sma_exit: int = 5,
    rsi_len: int = 2,
) -> tuple:
    """
    Tambahkan sinyal entry/exit berbasis RSI(2) mean reversion.

    Entry (BUY): Harga > SMA_trend (uptrend) DAN RSI(2) < rsi_entry (oversold jangka pendek).
    Exit (SELL): RSI(2) > rsi_exit (overbought) ATAU Harga > SMA_exit.

    Args:
        data: DataFrame OHLCV.
        rsi_entry: Threshold RSI untuk entry (misal 10 = beli jika RSI < 10).
        rsi_exit: Threshold RSI untuk exit (misal 70 = jual jika RSI > 70).
        sma_trend: Periode SMA untuk filter trend (default 200).
        sma_exit: Periode SMA untuk exit cepat (default 5).
        rsi_len: Periode RSI (default 2).

    Returns:
        Tuple (df_with_indicators, entry_mask, exit_mask)
    """
    df = data.copy()
    df["SMA_trend"] = df["Close"].rolling(sma_trend).mean()
    df["SMA_exit"] = df["Close"].rolling(sma_exit).mean()
    df["RSI2"] = compute_rsi(df["Close"], rsi_len)

    # Drop baris awal yang belum punya nilai indikator (NaN dari rolling window)
    # Ini BUKAN look-ahead — kita hanya membuang baris yang tidak bisa dihitung
    df = df.dropna(subset=["SMA_trend", "SMA_exit", "RSI2"]).copy()

    entry_mask = (df["Close"] > df["SMA_trend"]) & (df["RSI2"] < rsi_entry)
    exit_mask = (df["RSI2"] > rsi_exit) | (df["Close"] > df["SMA_exit"])

    return df, entry_mask, exit_mask


# ------------------------------------------------------------------ #
# Bollinger Bands Mean Reversion Signals
# ------------------------------------------------------------------ #
def add_bollinger_signals(
    data: pd.DataFrame,
    num_std: float,
    window: int = 20,
) -> tuple:
    """
    Tambahkan sinyal entry/exit berbasis Bollinger Bands mean reversion.

    Entry (BUY): ZScore < -num_std (harga di bawah lower band → oversold).
    Exit (SELL): ZScore >= 0 (harga kembali ke mean / SMA).

    Args:
        data: DataFrame OHLCV.
        num_std: Jumlah standar deviasi untuk lebar band (misal 2.0).
        window: Periode rolling untuk SMA dan STD (default 20).

    Returns:
        Tuple (df_with_indicators, entry_mask, exit_mask)
    """
    df = data.copy()
    df["SMA"] = df["Close"].rolling(window).mean()
    df["STD"] = df["Close"].rolling(window).std()
    df["Upper"] = df["SMA"] + num_std * df["STD"]
    df["Lower"] = df["SMA"] - num_std * df["STD"]
    df["ZScore"] = (df["Close"] - df["SMA"]) / df["STD"]

    df = df.dropna(subset=["SMA", "STD", "ZScore"]).copy()

    # Entry saat ZScore < -num_std: harga ekstrem di bawah rata-rata
    entry_mask = df["ZScore"] < -num_std
    # Exit saat ZScore >= 0: harga kembali ke rata-rata
    exit_mask = df["ZScore"] >= 0

    return df, entry_mask, exit_mask
