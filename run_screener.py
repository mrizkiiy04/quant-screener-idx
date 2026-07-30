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
from src.indicators import add_signals, compute_supertrend, compute_echo_forecast
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


def fetch_recent_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Unduh data 2 tahun agar Echo Forecast punya cukup histori."""
    df = yf.download(ticker, period=period, progress=False)
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
            ticker_obj = yf.Ticker(ticker)
            raw = fetch_recent_data(ticker)
            
            if raw.empty or len(raw) < 50:
                continue
                
            # Fetch fundamentals
            eps = None
            bv = None
            try:
                info = ticker_obj.info
                eps = info.get("trailingEps")
                bv = info.get("bookValue")
            except Exception as e:
                print(f"Fundamentals fetch failed for {ticker}: {e}")
                
            fair_value = None
            valuation_status = "N/A"
            margin = None
            
            if eps and eps > 0:
                # Menggunakan standar valuasi P/E 15x untuk mature blue chips (jauh lebih akurat dari Graham Number lama untuk sektor perbankan LQ45)
                fair_value = eps * 15
                
            # Pisahkan atr_mult karena tidak dipakai di add_signals
            signal_params = {k: v for k, v in GOLDEN_PARAMS.items() if k != "atr_mult"}
            atr_mult = GOLDEN_PARAMS["atr_mult"]
            
            # --- Tambahan Supertrend untuk Chart ---
            df = compute_supertrend(raw, period=10, multiplier=3.0)

            # 2. Hitung indikator BB & Dapatkan mask
            df, entry_mask, exit_mask = add_signals(df, **signal_params)
            
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
                    "sma": float(row["BB_SMA"]),
                    "st": float(row["ST"]) if pd.notnull(row["ST"]) else None,
                    "st_dir": int(row["ST_DIR"]) if pd.notnull(row["ST_DIR"]) else 0,
                    "volume": int(row["Volume"])
                })
                
            # Rincian kondisi rules pada hari terakhir
            rule_details = {
                "oversold": bool(last_row["Pct_B"] < GOLDEN_PARAMS["percent_b_entry"]) if "percent_b_entry" in GOLDEN_PARAMS else bool(last_row["Pct_B"] < 0.05),
                "regime": bool(last_row["ADX"] < GOLDEN_PARAMS["adx_threshold"]),
                "volume": bool(last_row["Vol_Ratio"] >= GOLDEN_PARAMS["vol_ratio_min"]),
                "squeeze": bool(last_row["BW_Squeeze"])
            }
            
            # Tentukan status valuasi
            if fair_value:
                margin = (fair_value - close_price) / fair_value
                if margin > 0.10:
                    valuation_status = "Harga Murah"
                elif -0.10 <= margin <= 0.10:
                    valuation_status = "Entry Wajar"
                else:
                    valuation_status = "Overvalued"
                    
            # --- Echo Forecast ---
            forecast_values = compute_echo_forecast(df, eval_window=60, forecast_window=20)
            forecast_dates = []
            if forecast_values:
                last_dt = df.index[-1]
                future_dts = pd.bdate_range(start=last_dt + pd.Timedelta(days=1), periods=len(forecast_values))
                forecast_dates = [dt.strftime("%Y-%m-%d") for dt in future_dts]
            
            # Simulasi riwayat trading 150 hari ke belakang
            _, trades_df = run_backtest(df, entry_mask, exit_mask, capital=10000000, atr_mult=atr_mult)
            trade_history = []
            if not trades_df.empty:
                for _, tr in trades_df.iterrows():
                    trade_history.append({
                        "entry_date": tr["entry_date"].strftime("%Y-%m-%d"),
                        "exit_date": tr["exit_date"].strftime("%Y-%m-%d"),
                        "entry_price": float(tr["entry_price"]),
                        "exit_price": float(tr["exit_price"]),
                        "return_pct": float(tr["return"]) * 100
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
                "history": trade_history,
                "forecast": {
                    "dates": forecast_dates,
                    "values": forecast_values
                },
                "fundamentals": {
                    "eps": eps,
                    "bv": bv,
                    "fair_value": fair_value,
                    "margin": margin,
                    "status": valuation_status
                }
            })
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            
    output_data = {
        "last_updated": last_updated,
        "params": GOLDEN_PARAMS,
        "signals": results
    }
    
    out_file = public_dir / "signals.json"
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Selesai! Hasil tersimpan di {out_file}")


if __name__ == "__main__":
    main()
