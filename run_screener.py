#!/usr/bin/env python3
"""
run_screener.py — Daily Signal Screener (Golden Strategy)

Skrip ini ditujukan untuk dijalankan setiap pukul 15:50 WIB (via GitHub Actions).
Ia akan mengunduh data hari ini, menghitung indikator, dan meludahkan sinyal
dalam format JSON untuk dibaca oleh frontend Netlify.
"""

import os
import json
import datetime
import time
import yfinance as yf
import pandas as pd
import feedparser
from google import genai
from google.genai import types
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


def fetch_rss_news(ticker: str) -> list:
    """Fetch recent news for a ticker from Google News RSS (Max 3 days old)."""
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+saham+when:3d&hl=id&gl=ID&ceid=ID:id"
        feed = feedparser.parse(url)
        news = []
        for entry in feed.entries:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                published_time = time.mktime(entry.published_parsed)
                current_time = time.time()
                days_old = (current_time - published_time) / (24 * 3600)
                if days_old > 3:
                    continue
            
            title = entry.title
            if " - " in title:
                title = title.rsplit(" - ", 1)[0]
            news.append({
                "title": title,
                "link": entry.link,
                "published": entry.get("published", "")
            })
            
            if len(news) >= 3:
                break
        return news
    except Exception as e:
        print(f"Error fetching RSS for {ticker}: {e}")
        return []


def generate_ai_sentiment_batch(items: list) -> dict:
    """Generate AI Sentiment in bulk using one API call."""
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # Fallback deterministic generator
    def fallback_generator():
        out = {}
        for item in items:
            ticker = item["ticker"]
            rule_details = item["rules"]
            margin = item.get("fundamentals", {}).get("margin")
            signal = item["signal"]
            bullets = []
            if rule_details.get("oversold"): bullets.append("Harga mendekati area oversold (Pct B < 0.05)")
            else: bullets.append("Harga berada di area wajar/atas")
            
            if rule_details.get("volume"): bullets.append("Terjadi lonjakan volume (Vol Ratio > 1.0x)")
            else: bullets.append("Volume perdagangan saat ini masih normal")
            
            if rule_details.get("regime"): bullets.append("Tren cenderung sideways atau mulai berbalik (ADX < 20)")
            else: bullets.append("Tren pergerakan harga sedang menguat (Trending)")
            
            if margin and margin > 0.1: bullets.append(f"Valuasi diskon {margin*100:.1f}% dari harga wajar")
            
            score = 80 if signal == "BUY" else (20 if signal == "SELL" else 50)
            action = "ACCUMULATE BUY" if signal == "BUY" else ("STRONG SELL" if signal == "SELL" else "HOLD / WAIT")
            out[ticker] = {"score": score, "action": action, "bullets": bullets[:4]}
        return out

    if not api_key:
        return fallback_generator()
    
    if not items:
        return {}

    try:
        client = genai.Client(api_key=api_key)
        
        prompt = "Anda adalah analis saham kuantitatif. Berikan analisis sentimen singkat untuk SETIAP saham di bawah ini.\n"
        for item in items:
            ticker = item['ticker']
            close = item['close']
            signal = item['signal']
            rule_details = item['rules']
            margin = item.get("fundamentals", {}).get("margin")
            win_rate = item.get("backtest_summary", {}).get("win_rate", 0)
            
            prompt += f"""
Saham: {ticker}
- Harga: {close}
- Sinyal: {signal}
- Oversold (Pct B < 0.05): {rule_details.get('oversold')}
- Volume Ratio > 1.0: {rule_details.get('volume')}
- ADX < 20 (Sideways): {rule_details.get('regime')}
- Diskon: {f"{margin*100:.1f}%" if margin else "N/A"}
- Win Rate Historis: {f"{win_rate:.1f}%"}
"""
        prompt += """
Tugas Anda:
Kembalikan JSON array dimana setiap elemen adalah object dengan format:
[
  {
      "ticker": "<Kode Saham>",
      "score": <angka 0-100, representasi keyakinan bullish, misal 75>,
      "action": "<ACCUMULATE BUY / HOLD / WAIT / STRONG SELL>",
      "bullets": ["<alasan 1 maksimal 8 kata>", "<alasan 2 maksimal 8 kata>", "<alasan 3 maksimal 8 kata>"]
  }
]
Format alasan dengan gaya analis teknikal/fundamental ringkas. Hanya kembalikan valid JSON array murni.
"""
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
                ai_data = json.loads(response.text)
                
                out = {}
                for obj in ai_data:
                    if "ticker" in obj:
                        out[obj["ticker"]] = {
                            "score": obj.get("score", 50),
                            "action": obj.get("action", "HOLD / WAIT"),
                            "bullets": obj.get("bullets", [])
                        }
                
                # Ensure all tickers have a fallback if Gemini skipped them
                fallback_data = fallback_generator()
                for item in items:
                    if item["ticker"] not in out:
                        out[item["ticker"]] = fallback_data[item["ticker"]]
                        
                return out
            except Exception as e:
                print(f"Gemini API attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    print("All 3 attempts failed. Using fallback generator.")
                    return fallback_generator()
                time.sleep(2)
    except Exception as e:
        print(f"Error calling Gemini in batch: {e}")
        return fallback_generator()


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

    # Limit to first 3 tickers for local testing if running manually, otherwise full
    universe_list = UNIVERSE
    if os.environ.get("QUICK_TEST"):
        universe_list = UNIVERSE[:3]

    for ticker in universe_list:
        try:
            ticker_obj = yf.Ticker(ticker)
            raw = fetch_recent_data(ticker)
            
            if raw.empty or len(raw) < 50:
                continue
                
            # Fetch fundamentals
            eps = None
            bv = None
            long_name = ticker
            try:
                info = ticker_obj.info
                eps = info.get("trailingEps")
                bv = info.get("bookValue")
                long_name = info.get("longName", info.get("shortName", ticker))
            except Exception as e:
                print(f"Fundamentals fetch failed for {ticker}: {e}")
                
            fair_value = None
            valuation_status = "N/A"
            margin = None
            
            if eps and eps > 0:
                fair_value = eps * 15
                
            signal_params = {k: v for k, v in GOLDEN_PARAMS.items() if k != "atr_mult"}
            atr_mult = GOLDEN_PARAMS["atr_mult"]
            
            df = compute_supertrend(raw, period=10, multiplier=3.0)
            df, entry_mask, exit_mask = add_signals(df, **signal_params)
            
            if df.empty:
                continue
                
            last_date = df.index[-1]
            last_row = df.iloc[-1]
            is_buy = bool(entry_mask.iloc[-1])
            is_sell = bool(exit_mask.iloc[-1])
            
            close_price = float(last_row["Close"])
            atr = float(last_row["ATR"])
            stop_level = close_price - (atr * atr_mult)
            
            if is_buy:
                signal = "BUY"
            elif is_sell:
                signal = "SELL"
            else:
                signal = "HOLD / WAIT"
                
            df_chart = df.tail(90)
            chart_data = []
            for idx, row in df_chart.iterrows():
                chart_data.append({
                    "date": idx.strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "upper": float(row["BB_Upper"]),
                    "lower": float(row["BB_Lower"]),
                    "sma": float(row["BB_SMA"]),
                    "st": float(row["ST"]) if pd.notnull(row["ST"]) else None,
                    "st_dir": int(row["ST_DIR"]) if pd.notnull(row["ST_DIR"]) else 0,
                    "volume": int(row["Volume"])
                })
                
            rule_details = {
                "oversold": bool(last_row["Pct_B"] < GOLDEN_PARAMS.get("percent_b_entry", 0.05)),
                "regime": bool(last_row["ADX"] < GOLDEN_PARAMS["adx_threshold"]),
                "volume": bool(last_row["Vol_Ratio"] >= GOLDEN_PARAMS["vol_ratio_min"]),
                "squeeze": bool(last_row["BW_Squeeze"])
            }
            
            if fair_value:
                margin = (fair_value - close_price) / fair_value
                if margin > 0.10:
                    valuation_status = "Harga Murah"
                elif -0.10 <= margin <= 0.10:
                    valuation_status = "Entry Wajar"
                else:
                    valuation_status = "Overvalued"
                    
            forecast_values = compute_echo_forecast(df, eval_window=60, forecast_window=20)
            forecast_dates = []
            if forecast_values:
                last_dt = df.index[-1]
                future_dts = pd.bdate_range(start=last_dt + pd.Timedelta(days=1), periods=len(forecast_values))
                forecast_dates = [dt.strftime("%Y-%m-%d") for dt in future_dts]
            
            _, trades_df = run_backtest(df, entry_mask, exit_mask, capital=10000000, atr_mult=atr_mult)
            trade_history = []
            win_rate = 0
            avg_profit = 0
            if not trades_df.empty:
                for _, tr in trades_df.iterrows():
                    trade_history.append({
                        "entry_date": tr["entry_date"].strftime("%Y-%m-%d"),
                        "exit_date": tr["exit_date"].strftime("%Y-%m-%d"),
                        "entry_price": float(tr["entry_price"]),
                        "exit_price": float(tr["exit_price"]),
                        "return_pct": float(tr["return"]) * 100
                    })
                wins = len(trades_df[trades_df["return"] > 0])
                win_rate = (wins / len(trades_df)) * 100
                avg_profit = trades_df["return"].mean() * 100
            
            trade_history = trade_history[::-1]
            
            # Fetch RSS
            news = fetch_rss_news(ticker)
                
            results.append({
                "ticker": ticker,
                "long_name": long_name,
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
                "backtest_summary": {
                    "win_rate": win_rate,
                    "avg_profit": avg_profit,
                    "trades_count": len(trade_history)
                },
                "news": news,
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
            
            print(f"Processed {ticker}")
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            
    # --- BATCH AI SENTIMENT GENERATION ---
    print("Generating AI Sentiment in batch...")
    ai_sentiments = generate_ai_sentiment_batch(results)
    
    for res in results:
        res["ai_sentiment"] = ai_sentiments.get(res["ticker"], {
            "score": 50,
            "action": "HOLD / WAIT",
            "bullets": ["Sentimen AI tidak tersedia"]
        })
            
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
