#!/usr/bin/env python3
"""
run_backtest.py — Entry point CLI untuk Walk-Forward Backtest IS/OOS

Strategi  : Bollinger Bands v2 (Golden Strategy)
Universe  : Saham IDX (.JK) via yfinance
Framework : Pure pandas/numpy (tanpa Backtrader)
Logging   : SQLite (results/backtest_results.db) + .log file per session

Usage:
    python run_backtest.py                             # Default 5 saham, mode static
    python run_backtest.py --mode rolling              # Anchored WFO
    python run_backtest.py --mode blind --tickers ICBP.JK # Uji lurus tanpa optimasi
"""

import argparse
import sys
import traceback
from pathlib import Path

from src.logger import BacktestLogger
from src.data import download_data, split_data
from src.grid_search import grid_search_signals
from src.indicators import add_signals
from src.engine import run_backtest
from src.walkforward import run_walkforward
from src.rolling import run_rolling_walkforward
from src.blind import run_blind_test
from src.robustness import parameter_plateau_test, plot_sensitivity_heatmap


# ------------------------------------------------------------------ #
# Default ticker universe (Top-5 IDX blue chips berdasarkan market cap)
# ------------------------------------------------------------------ #
DEFAULT_TICKERS = [
    "BMRI.JK",   # Bank Mandiri
    "BBCA.JK",   # BCA
    "TLKM.JK",   # Telkom Indonesia
    "ASII.JK",   # Astra International
    "BBRI.JK",   # BRI
]


# ------------------------------------------------------------------ #
# Argument parser
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_backtest",
        description=(
            "Walk-Forward Backtest IS/OOS — IDX Stocks\n"
            "Strategi: Bollinger Bands v2 (Golden Strategy)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python run_backtest.py
  python run_backtest.py --mode rolling
  python run_backtest.py --mode blind --tickers CPIN.JK
        """
    )

    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS, metavar="TICKER",
        help=f"Daftar ticker IDX (default: {' '.join(DEFAULT_TICKERS)})"
    )
    parser.add_argument(
        "--start", default="2021-01-01", metavar="YYYY-MM-DD",
        help="Tanggal mulai data (default: 2021-01-01)"
    )
    parser.add_argument(
        "--end", default="2026-07-01", metavar="YYYY-MM-DD",
        help="Tanggal akhir data (default: 2026-07-01)"
    )
    parser.add_argument(
        "--capital", type=float, default=10_000_000, metavar="IDR",
        help="Modal awal dalam Rupiah (default: 10000000)"
    )
    parser.add_argument(
        "--is-ratio", type=float, default=0.7, metavar="FLOAT",
        help="Proporsi In-Sample 0–1 (default: 0.7 = 70%% awal untuk kalibrasi)"
    )
    parser.add_argument(
        "--mode", choices=["static", "rolling", "blind"], default="static",
        help="Mode optimasi: static (split tunggal), rolling (anchored WFO), atau blind (tanpa optimasi) (default: static)"
    )
    parser.add_argument(
        "--train-days", type=int, default=750,
        help="Jumlah hari untuk initial In-Sample window di mode 'rolling' (default: 750)"
    )
    parser.add_argument(
        "--test-days", type=int, default=250,
        help="Jumlah hari untuk tiap Out-of-Sample window di mode 'rolling' (default: 250)"
    )
    parser.add_argument(
        "--results-dir", default="results", metavar="DIR",
        help="Direktori output untuk logs, db, csv, charts (default: results)"
    )

    return parser.parse_args()


# ------------------------------------------------------------------ #
# Main execution
# ------------------------------------------------------------------ #
def main():
    args = parse_args()

    # Inisialisasi sistem logging ke folder yang diminta
    logger = BacktestLogger(args.results_dir)

    logger.info(f"======================================================================")
    logger.info(f"  WALK-FORWARD BACKTEST — Terminal Edition")
    logger.info(f"  Session ID : {logger.run_id}")
    logger.info(f"  Tickers    : {', '.join(args.tickers)}")
    logger.info(f"  Periode    : {args.start} s/d {args.end}")
    logger.info(f"  Modal      : Rp {args.capital:,.0f}")
    if args.mode == "static":
        logger.info(f"  IS/OOS     : {args.is_ratio:.0%} / {1 - args.is_ratio:.0%}")
    elif args.mode == "rolling":
        logger.info(f"  WFO Mode   : ROLLING (Train {args.train_days} / Test {args.test_days})")
    elif args.mode == "blind":
        logger.info(f"  WFO Mode   : BLIND TEST (Locked Params, Unseen Data)")
    logger.info(f"  Strategi   : BOLLINGER BANDS V2 (Golden Strategy)")
    logger.info(f"  Output dir : {Path(args.results_dir).resolve()}")
    logger.info(f"======================================================================")
    logger.info("")

    summary = []

    for ticker in args.tickers:
        logger.info(f"######################################################################")
        logger.info(f"  TICKER: {ticker}")
        logger.info(f"######################################################################")
        try:
            # 1. Download data
            raw = download_data(ticker, args.start, args.end)
            if raw.empty or len(raw) < 50:
                logger.error(f"[{ticker}] Data kosong atau terlalu pendek. Skip.")
                continue

            if args.mode == "static":
                # Data split untuk mode static
                is_data, oos_data, _, _ = split_data(raw, is_ratio=args.is_ratio)
                logger.info(f"  Data    : {len(raw)} hari trading")
                logger.info(f"  IS      : {is_data.index[0].date()} → {is_data.index[-1].date()} ({len(is_data)} hari)")
                logger.info(f"  OOS     : {oos_data.index[0].date()} → {oos_data.index[-1].date()} ({len(oos_data)} hari)")
                logger.info("")
            else:
                logger.info(f"  Data    : {len(raw)} hari trading")
                logger.info("")

            if args.mode == "rolling":
                # --- ROLLING WFO PATH ---
                result = run_rolling_walkforward(
                    name="Bollinger Bands v2 (%B+ADX+Vol+ATR)",
                    grid_fn=grid_search_signals,
                    signal_fn=add_signals,
                    engine_fn=run_backtest,
                    raw_data=raw,
                    train_days=args.train_days,
                    test_days=args.test_days,
                    capital=args.capital,
                    ticker=ticker,
                    timestamp=logger.timestamp,
                    logger=logger,
                    results_dir=args.results_dir,
                )
                
                if result:
                    logger.save_run(
                        ticker=ticker, strategy="bb_v2_rolling",
                        start=args.start, end=args.end,
                        is_ratio=0.0, capital=args.capital, 
                        best_params=result["params"], 
                        is_m=result["is"], oos_m=result["oos"],
                        score=result["score"], degradation=result["degradation"],
                        verdict=result["verdict"],
                        leakage_flags=result["leakage_flags"]
                    )
                    summary.append(result)

            elif args.mode == "blind":
                # --- BLIND TEST PATH ---
                locked_params = {
                    "num_std": 2.0,
                    "window": 20,
                    "adx_threshold": 20,
                    "vol_ratio_min": 1.0,
                    "atr_mult": 1.5
                }
                result = run_blind_test(
                    name="Bollinger Bands v2 (%B+ADX+Vol+ATR)",
                    signal_fn=add_signals,
                    engine_fn=run_backtest,
                    raw_data=raw,
                    params=locked_params,
                    capital=args.capital,
                    ticker=ticker,
                    timestamp=logger.timestamp,
                    logger=logger,
                    results_dir=args.results_dir,
                )
                if result:
                    result["name"] = "Bollinger Bands v2"
                    result["ticker"] = ticker
                    logger.save_run(
                        ticker=ticker, strategy="bb_v2_blind",
                        start=args.start, end=args.end,
                        is_ratio=0.0, capital=args.capital, 
                        best_params=result["params"], 
                        is_m=result["is"], oos_m=result["oos"],
                        score=result["score"], degradation=result["degradation"],
                        verdict=result["verdict"],
                        leakage_flags=result["leakage_flags"]
                    )
                    summary.append(result)
                    
            else:
                # --- STATIC WFO PATH ---
                result = run_walkforward(
                    name="Bollinger Bands v2 (%B+ADX+Vol+ATR)",
                    grid_fn=grid_search_signals,
                    signal_fn=add_signals,
                    engine_fn=run_backtest,
                    is_data=is_data,
                    oos_data=oos_data,
                    capital=args.capital,
                    ticker=ticker,
                    timestamp=logger.timestamp,
                    logger=logger,
                    results_dir=args.results_dir,
                )
                if result:
                    result["name"] = "Bollinger Bands v2"
                    result["ticker"] = ticker
                    # Sensitivitas
                    try:
                        plateau_data = parameter_plateau_test(
                            is_data=is_data,
                            best_params=result["params"],
                            signal_fn=add_signals,
                            capital=args.capital
                        )
                        score = plateau_data["plateau_score"]
                        plot_sensitivity_heatmap(
                            plateau_data=plateau_data,
                            best_params=result["params"],
                            ticker=ticker,
                            timestamp=logger.timestamp,
                            charts_dir=str(Path(args.results_dir) / "charts")
                        )
                    except Exception as e:
                        logger.error(f"Gagal generate heatmap untuk {ticker}: {e}")
                        score = None

                    # Log ke SQLite
                    logger.save_run(
                        ticker=ticker, strategy="bb_v2",
                        start=args.start, end=args.end,
                        is_ratio=args.is_ratio, capital=args.capital,
                        best_params=result["params"],
                        is_m=result["is"], oos_m=result["oos"],
                        score=score, degradation=result["degradation"],
                        verdict=result["verdict"],
                        leakage_flags=result["leakage_flags"]
                    )
                    summary.append(result)

            logger.info("")

        except Exception as e:
            logger.error(f"[{ticker}] Fatal error: {e}")
            logger.error(traceback.format_exc())

    # --- Selesai loop ticker ---
    logger.print_comparison_table(summary)

    logger.info(f"======================================================================")
    logger.info(f"  SELESAI")
    logger.info(f"  Log file : {logger.log_path}")
    logger.info(f"  Database : {logger.db_path}")
    logger.info(f"  Charts   : {Path(args.results_dir)/'charts'}/")
    logger.info(f"  CSV      : {Path(args.results_dir)/'csv'}/")
    logger.info(f"======================================================================")


if __name__ == "__main__":
    main()
