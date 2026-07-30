"""
data.py — Download dan validasi data OHLCV via yfinance.
"""

import yfinance as yf
import pandas as pd
from typing import Tuple


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download data OHLCV dari yfinance.

    Returns:
        DataFrame dengan kolom flat (Open, High, Low, Close, Volume).

    Raises:
        ValueError: jika data < 250 hari (tidak cukup untuk SMA-200 + IS/OOS split).
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # Flatten MultiIndex jika ada (yfinance kadang return MultiIndex)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if raw.empty:
        raise ValueError(
            f"Tidak ada data untuk ticker '{ticker}'. "
            f"Pastikan format benar (contoh: BMRI.JK) dan koneksi internet aktif."
        )

    if len(raw) < 250:
        raise ValueError(
            f"Data terlalu sedikit: {len(raw)} hari untuk '{ticker}'. "
            f"Minimal 250 hari diperlukan (SMA-200 + IS/OOS split). "
            f"Coba perpanjang rentang tanggal."
        )

    # Pastikan kolom Close ada
    if "Close" not in raw.columns:
        raise ValueError(f"Kolom 'Close' tidak ditemukan di data {ticker}. Kolom: {list(raw.columns)}")

    # Drop baris dengan Close NaN
    raw = raw.dropna(subset=["Close"])

    return raw


def split_data(
    raw: pd.DataFrame, is_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.core.indexes.datetimes.DatetimeTZDtype, int]:
    """
    Split data menjadi In-Sample (kalibrasi) dan Out-of-Sample (validasi).

    Args:
        raw: DataFrame OHLCV lengkap.
        is_ratio: Float 0-1, proporsi data untuk IS (misalnya 0.7 = 70% pertama).

    Returns:
        Tuple: (is_data, oos_data, split_date, split_idx)

    Catatan Leakage:
        - Grid search HANYA boleh dijalankan di is_data.
        - oos_data tidak boleh disentuh sampai parameter dari IS sudah terkunci.
        - split_date digunakan sebagai garis pemisah di chart.
    """
    if not (0 < is_ratio < 1):
        raise ValueError(f"is_ratio harus antara 0 dan 1 (eksklusif), dapat: {is_ratio}")

    split_idx = int(len(raw) * is_ratio)

    # Pastikan minimal IS dan OOS punya cukup data
    if split_idx < 200:
        raise ValueError(
            f"IS period terlalu pendek ({split_idx} hari). "
            f"Kurangi is_ratio atau perpanjang data."
        )
    if len(raw) - split_idx < 30:
        raise ValueError(
            f"OOS period terlalu pendek ({len(raw) - split_idx} hari). "
            f"Naikkan is_ratio atau perpanjang data."
        )

    split_date = raw.index[split_idx]
    is_data = raw.iloc[:split_idx].copy()
    oos_data = raw.iloc[split_idx:].copy()

    return is_data, oos_data, split_date, split_idx
