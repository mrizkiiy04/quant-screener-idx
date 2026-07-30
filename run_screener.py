#!/usr/bin/env python3
"""
run_screener.py — Daily Signal Screener (Golden Strategy)

Skrip ini ditujukan untuk dijalankan setiap pukul 15:50 WIB (via GitHub Actions).
Ia akan mengunduh data hari ini, menghitung indikator, dan meludahkan sinyal
dalam format JSON untuk dibaca oleh frontend Netlify.
"""

import json
import datetime
import yfinance as yf
import pandas as pd
from pathlib import Path

from src.indicators import add_signals
from src.engine import run_backtest

# Daftar 45 Saham Paling Likuid di Indonesia (Indeks LQ45)
UNIVERSE = [
    "ACES.JK", "ADRO.JK", "AKRA.JK", "AMMN.JK", "AMRT.JK", 
    "ANTM.JK", "ARTO.JK", "ASII.JK", "BBCA.JK", "BBNI.JK", 
    "BBRI.JK", "BBTN.JK", "BMRI.JK", "BRIS.JK", "BRPT.JK", 
    "BUKA.JK", "CPIN.JK", "ESSA.JK", "EXCL.JK", "GGRM.JK", 
    "GOTO.JK", "HRUM.JK", "ICBP.JK", "INCO.JK", "INDF.JK", 
    "INKP.JK", "INTP.JK", "ITMG.JK", "KLBF.JK", "MAPI.JK", 
    "MBMA.JK", "MDKA.JK", "MEDC.JK", "MTEL.JK", "PGAS.JK", 
    "PGEO.JK", "PTBA.JK", "SIDO.JK", "SMGR.JK", "SRTG.JK", 
    "TLKM.JK", "TOWR.JK", "TPIA.JK", "UNTR.JK", "UNVR.JK"
]

# The Golden Ratio (Locked from TLKM Robust Test)
GOLDEN_PARAMS = {
    "num_std": 2.0,
    "window": 20,
    "adx_threshold": 20,
    "vol_ratio_min": 1.0,
    "atr_mult": 1.5
}


def fetch_recent_data(ticker: str, days: int = 150) -> pd.DataFrame:
    """Unduh data 150 hari terakhir agar MA dan ADX cukup ruang pemanasan."""
    end = datetime.date.today() + datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=days)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df.dropna()


def main():
    public_dir = Path("public")
    public_dir.mkdir(exist_ok=True)
    
    results = []
    
    # Supaya frontend tau kapan terakhir update
    last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M WIB")

    for ticker in UNIVERSE:
        try:
            raw = fetch_recent_data(ticker)
            if raw.empty or len(raw) < 50:
                continue
                
            # Pisahkan atr_mult karena tidak dipakai di add_signals
            signal_params = {k: v for k, v in GOLDEN_PARAMS.items() if k != "atr_mult"}
            atr_mult = GOLDEN_PARAMS["atr_mult"]
            
            df, entry_mask, exit_mask = add_signals(raw, **signal_params)
            
            if df.empty:
                continue
                
            # Ambil data hari terakhir (hari ini) untuk sinyal utama
            last_date = df.index[-1]
            last_row = df.iloc[-1]
            is_buy = bool(entry_mask.iloc[-1])
            is_sell = bool(exit_mask.iloc[-1])
            
            close_price = float(last_row["Close"])
            atr = float(last_row["ATR"])
            
            # Hitung Trailing Stop jika hari ini kita memegang barang
            stop_level = close_price - (atr * atr_mult)
            
            if is_buy:
                signal = "BUY"
            elif is_sell:
                signal = "SELL"
            else:
                signal = "HOLD / WAIT"
                
            # Ambil data historis 90 hari terakhir untuk chart
            df_chart = df.tail(90)
            chart_data = []
            for idx, row in df_chart.iterrows():
                chart_data.append({
                    "date": idx.strftime("%Y-%m-%d"),
                    "close": float(row["Close"]),
                    "upper": float(row["BB_Upper"]),
                    "lower": float(row["BB_Lower"]),
                    "sma": float(row["BB_SMA"])
                })
                
            # Rincian kondisi rules pada hari terakhir
            rule_details = {
                "oversold": bool(last_row["Pct_B"] < GOLDEN_PARAMS["percent_b_entry"]) if "percent_b_entry" in GOLDEN_PARAMS else bool(last_row["Pct_B"] < 0.05),
                "regime": bool(last_row["ADX"] < GOLDEN_PARAMS["adx_threshold"]),
                "volume": bool(last_row["Vol_Ratio"] >= GOLDEN_PARAMS["vol_ratio_min"]),
                "squeeze": bool(last_row["BW_Squeeze"])
            }
            
            # Simulasi riwayat trading 150 hari ke belakang
            _, trades_df = run_backtest(df, entry_mask, exit_mask, capital=10000000, atr_mult=atr_mult)
            trade_history = []
            if not trades_df.empty:
                for _, tr in trades_df.iterrows():
                    trade_history.append({
                        "entry_date": tr["Entry Date"].strftime("%Y-%m-%d"),
                        "exit_date": tr["Exit Date"].strftime("%Y-%m-%d"),
                        "entry_price": float(tr["Entry Price"]),
                        "exit_price": float(tr["Exit Price"]),
                        "return_pct": float(tr["Return (%)"])
                    })
            # Urutkan trades dari terbaru ke terlama
            trade_history = trade_history[::-1]
                
            results.append({
                "ticker": ticker,
                "date": last_date.strftime("%Y-%m-%d"),
                "close": close_price,
                "signal": signal,
                "atr": atr,
                "stop_loss": stop_level,
                "pct_b": float(last_row["Pct_B"]),
                "adx": float(last_row["ADX"]),
                "vol_ratio": float(last_row["Vol_Ratio"]),
                "rules": rule_details,
                "chart": chart_data,
                "history": trade_history
            })
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            
    output_data = {
        "last_updated": last_updated,
        "signals": results
    }
    
    out_file = public_dir / "signals.json"
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Selesai! Hasil tersimpan di {out_file}")


if __name__ == "__main__":
    main()
