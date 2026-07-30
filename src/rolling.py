"""
rolling.py — Modul Rolling Walk-Forward Optimization (Anchored).

Menjalankan proses sliding/expanding window WFO:
1. Melatih parameter (grid search) di masa lalu (In-Sample).
2. Mengeksekusi (trading) pada periode singkat (OOS).
3. Menggeser jendela dan melatih ulang.
4. Menyambungkan kurva equity OOS.
5. Mencatat log stabilitas parameter (agar bisa dicek jika parameter melompat liar).
"""

from pathlib import Path
from typing import Callable, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.engine import check_leakage
from src.metrics import compute_metrics


def run_rolling_walkforward(
    name: str,
    grid_fn: Callable,
    signal_fn: Callable,
    engine_fn: Callable,
    raw_data: pd.DataFrame,
    train_days: int,
    test_days: int,
    capital: float,
    ticker: str,
    timestamp: str,
    logger,
    results_dir: str = "results",
) -> Optional[dict]:
    """
    Jalankan siklus Anchored Rolling Walk-Forward.

    Args:
        name        : Nama strategi (untuk chart/log).
        grid_fn     : Fungsi optimasi grid search.
        signal_fn   : Fungsi generator sinyal.
        engine_fn   : Fungsi eksekusi backtest (engine_v2/backtest_v2 disarankan).
        raw_data    : Full dataframe dari data download (sorted by index).
        train_days  : Panjang Initial Train Window (hari trading, misal 750).
        test_days   : Panjang Test Window (OOS) / Step size (hari trading, misal 250).
        capital     : Modal awal simulasi.
        ticker      : Simbol ticker.
        timestamp   : ID unik run.
        logger      : Logger class.
        results_dir : Output directory.
    """
    charts_dir = Path(results_dir) / "charts"
    csv_dir    = Path(results_dir) / "csv"
    charts_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    n_days = len(raw_data)
    if n_days <= train_days + test_days:
        logger.error(f"Data terlalu pendek ({n_days} hari). Butuh > {train_days + test_days}.")
        return None

    # Struktur penyimpanan untuk gabungan OOS
    all_oos_trades = []
    daily_returns_oos = pd.Series(dtype=float)

    # Logging parameter
    parameter_log = []
    leakage_flags = []

    # Iterasi Anchored WFO
    current_train_end = train_days
    fold = 1

    logger.info(f"\n{'='*70}")
    logger.info(f"  ROLLING WFO (Anchored) — {name} [{ticker}]")
    logger.info(f"  Total Data: {n_days} hari | Train={train_days} | Test={test_days}")
    logger.info(f"{'='*70}")

    while current_train_end < n_days:
        train_start_idx = 0  # Anchored: selalu mulai dari awal
        test_end_idx = min(current_train_end + test_days, n_days)

        is_data = raw_data.iloc[train_start_idx:current_train_end].copy()
        oos_data = raw_data.iloc[current_train_end:test_end_idx].copy()

        is_start_dt = is_data.index[0].strftime('%Y-%m-%d')
        is_end_dt   = is_data.index[-1].strftime('%Y-%m-%d')
        oos_start_dt= oos_data.index[0].strftime('%Y-%m-%d')
        oos_end_dt  = oos_data.index[-1].strftime('%Y-%m-%d')

        logger.info(f"\n--- FOLD {fold} ---")
        logger.info(f"IS : {is_start_dt} → {is_end_dt} ({len(is_data)} hari)")
        logger.info(f"OOS: {oos_start_dt} → {oos_end_dt} ({len(oos_data)} hari)")

        # 1. Optimasi IS
        results = grid_fn(is_data, capital)
        if not results:
            logger.warning(f"Fold {fold}: Tidak ada parameter valid di In-Sample.")
            current_train_end += test_days
            fold += 1
            continue

        best_params = results[0]["params"]
        logger.info(f"Fold {fold} Best Params: {best_params}")
        logger.info(f"Fold {fold} IS Sharpe: {results[0]['sharpe']:.2f} | Trades: {results[0]['n_trades']}")

        # Catat parameter log
        parameter_log.append({
            "fold": fold,
            "oos_start": oos_start_dt,
            "oos_end": oos_end_dt,
            "oos_days": len(oos_data),
            **best_params
        })

        # Pisahkan engine params
        ENGINE_ONLY_PARAMS = {"atr_mult"}
        signal_params = {k: v for k, v in best_params.items() if k not in ENGINE_ONLY_PARAMS}
        engine_kwargs = {k: v for k, v in best_params.items() if k in ENGINE_ONLY_PARAMS}

        # Jalankan OOS 
        df_oos, em_oos, xm_oos = signal_fn(oos_data, **signal_params)
        leakages = check_leakage(df_oos, em_oos, xm_oos)
        if leakages:
            leakage_flags.extend(leakages)
        
        df_oos_result, trades_oos = engine_fn(df_oos, em_oos, xm_oos, capital, **engine_kwargs)
        
        if not trades_oos.empty:
            trades_oos["fold"] = fold
            all_oos_trades.append(trades_oos)
            logger.info(f"Fold {fold} OOS Trades: {len(trades_oos)}")
        else:
            logger.info(f"Fold {fold} OOS Trades: 0")

        daily_ret = df_oos_result["Equity"].pct_change().fillna(0)
        daily_returns_oos = pd.concat([daily_returns_oos, daily_ret])

        current_train_end += test_days
        fold += 1

    if daily_returns_oos.empty:
        logger.warning(f"[{ticker}] Tidak ada simulasi OOS yang sukses.")
        return None

    continuous_equity = capital * (1 + daily_returns_oos).cumprod()
    
    # Menangani duplicated indices karena pd.concat dari daily returns (yang bisa tumpang tindih 1 hari jika ada)
    # Hapus indeks duplikat:
    daily_returns_oos = daily_returns_oos[~daily_returns_oos.index.duplicated(keep='last')]
    continuous_equity = capital * (1 + daily_returns_oos).cumprod()

    df_continuous = pd.DataFrame({
        "Close": raw_data.loc[continuous_equity.index, "Close"],
        "Equity": continuous_equity
    })

    if all_oos_trades:
        df_trades_all = pd.concat(all_oos_trades, ignore_index=True)
    else:
        df_trades_all = pd.DataFrame()

    total_oos_m = compute_metrics(df_continuous, df_trades_all, capital)

    # Parameter Stability
    df_params = pd.DataFrame(parameter_log)
    logger.info(f"\n--- PARAMETER STABILITY TABLE [{ticker}] ---")
    
    stability_metrics = {}
    for col in df_params.columns:
        if col not in ["fold", "oos_start", "oos_end", "oos_days"]:
            stability_metrics[col] = df_params[col].nunique()
    
    logger.info(f"\n{df_params.to_string(index=False)}")
    
    unstable_params = [p for p, count in stability_metrics.items() if count > len(parameter_log) * 0.5]
    if unstable_params:
        logger.warning(f"Parameter sangat tidak stabil (banyak melompat): {unstable_params}")
    else:
        logger.info("Parameter cukup stabil (sedikit perubahan ekstrem antar fold).")

    slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "%25")
    ts = timestamp
    
    if not df_trades_all.empty:
        trades_csv = csv_dir / f"trades_OOS_ROLLING_{slug}_{ticker.replace('.','_')}_{ts}.csv"
        df_trades_all.to_csv(trades_csv, index=False)
    
    equity_csv = csv_dir / f"equity_ROLLING_{slug}_{ticker.replace('.','_')}_{ts}.csv"
    df_continuous.to_csv(equity_csv)
    
    param_csv = csv_dir / f"params_ROLLING_{slug}_{ticker.replace('.','_')}_{ts}.csv"
    df_params.to_csv(param_csv, index=False)

    if not df_continuous.empty:
        bh_start_price = df_continuous["Close"].iloc[0]
        bh_continuous_equity = capital * (df_continuous["Close"] / bh_start_price)
        
        plt.figure(figsize=(12, 8))
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(df_continuous.index, df_continuous["Equity"], label=f"Strategi OOS (Rolling) ({total_oos_m['total_return']:.1f}%)", color="#2563eb", linewidth=1.5)
        
        bh_ret = (bh_continuous_equity.iloc[-1] / capital - 1) * 100
        ax1.plot(df_continuous.index, bh_continuous_equity, label=f"Buy & Hold OOS ({bh_ret:.1f}%)", color="#ef4444", linestyle="--", alpha=0.8)
        
        ax1.set_title(f"Rolling Walk-Forward (Anchored): {name} — {ticker}", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Equity (Rp)")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="upper left")

        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        peak_strat = df_continuous["Equity"].cummax()
        dd_strat = (df_continuous["Equity"] - peak_strat) / peak_strat * 100
        peak_bh = bh_continuous_equity.cummax()
        dd_bh = (bh_continuous_equity - peak_bh) / peak_bh * 100
        
        ax2.fill_between(df_continuous.index, dd_strat, 0, color="#86efac", alpha=0.7, label="DD Strategi OOS")
        ax2.plot(df_continuous.index, dd_bh, color="#ef4444", linestyle="--", linewidth=1, alpha=0.7, label="DD Buy&Hold OOS")
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="lower left", fontsize="small", ncol=2)
        
        plt.tight_layout()
        chart_path = charts_dir / f"walkforward_ROLLING_{slug}_{ticker.replace('.','_')}_{ts}.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        logger.info(f"[CHART ROLLING] → {chart_path}")

    logger.info(f"\n----------------------------------------------------------------------")
    logger.info(f"  HASIL AKHIR ROLLING WFO: {name} [{ticker}]")
    logger.info(f"----------------------------------------------------------------------")
    logger.info(f"Jumlah Total Trade OOS            {total_oos_m['n_trades']}")
    logger.info(f"Win rate                          {total_oos_m['win_rate']:.1f}%")
    logger.info(f"Total return OOS                  {total_oos_m['total_return']:.1f}%")
    logger.info(f"Sharpe Ratio                      {total_oos_m['sharpe']:.2f}")
    logger.info(f"Max drawdown                      {total_oos_m['max_dd']:.1f}%")
    logger.info(f"----------------------------------------------------------------------")
    
    if total_oos_m['n_trades'] < 5:
        verdict = "INCONCLUSIVE"
    elif total_oos_m['total_return'] > 0 and total_oos_m['sharpe'] > 0.5:
        verdict = "ROBUST"
    elif total_oos_m['total_return'] > 0:
        verdict = "LEMAH TAPI MASIH PROFIT"
    else:
        verdict = "GAGAL"
        
    logger.info(f"VERDICT: {verdict}")

    return {
        "is": {},
        "oos": total_oos_m,
        "score": 0,
        "degradation": 0,
        "verdict": verdict,
        "params": parameter_log[-1] if parameter_log else {},
        "leakage_flags": list(set(leakage_flags))
    }
