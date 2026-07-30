"""
blind.py — Modul eksekusi Blind Out-of-Sample Test.

Menjalankan backtest secara lurus pada data menggunakan
satu set parameter baku yang telah ditentukan sebelumnya,
untuk menguji 'Alpha' secara universal tanpa bias optimasi.
"""

from pathlib import Path
from typing import Callable, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.engine import check_leakage
from src.metrics import compute_metrics


def run_blind_test(
    name: str,
    signal_fn: Callable,
    engine_fn: Callable,
    raw_data: pd.DataFrame,
    params: dict,
    capital: float,
    ticker: str,
    timestamp: str,
    logger,
    results_dir: str = "results",
) -> Optional[dict]:
    """
    Eksekusi satu uji buta (blind test) tanpa pembelahan data atau grid search.
    """
    charts_dir = Path(results_dir) / "charts"
    csv_dir    = Path(results_dir) / "csv"
    charts_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    n_days = len(raw_data)
    logger.info(f"\n{'='*70}")
    logger.info(f"  BLIND TEST — {name} [{ticker}]")
    logger.info(f"  Total Data: {n_days} hari")
    logger.info(f"  Locked Params: {params}")
    logger.info(f"{'='*70}")

    if n_days < 50:
        logger.error(f"[{ticker}] Data terlalu pendek untuk blind test.")
        return None

    # Pisahkan engine params
    ENGINE_ONLY_PARAMS = {"atr_mult"}
    signal_params = {k: v for k, v in params.items() if k not in ENGINE_ONLY_PARAMS}
    engine_kwargs = {k: v for k, v in params.items() if k in ENGINE_ONLY_PARAMS}

    # Generate signals
    df, em, xm = signal_fn(raw_data, **signal_params)
    leakage_flags = check_leakage(df, em, xm)

    # Run engine
    df_result, trades_df = engine_fn(df, em, xm, capital, **engine_kwargs)

    # Hitung metrics
    metrics = compute_metrics(df_result, trades_df, capital)

    # Output log
    logger.info(f"\n----------------------------------------------------------------------")
    logger.info(f"  HASIL AKHIR BLIND TEST: {name} [{ticker}]")
    logger.info(f"----------------------------------------------------------------------")
    logger.info(f"Jumlah Total Trade                {metrics['n_trades']}")
    logger.info(f"Win rate                          {metrics['win_rate']:.1f}%")
    logger.info(f"Total return                      {metrics['total_return']:.1f}%")
    logger.info(f"CAGR                              {metrics['cagr']:.1f}%")
    logger.info(f"Sharpe Ratio                      {metrics['sharpe']:.2f}")
    logger.info(f"Max drawdown                      {metrics['max_dd']:.1f}%")
    logger.info(f"----------------------------------------------------------------------")
    
    if metrics['n_trades'] < 5:
        verdict = "INCONCLUSIVE"
    elif metrics['total_return'] > 0 and metrics['sharpe'] > 0.5:
        verdict = "ROBUST"
    elif metrics['total_return'] > 0:
        verdict = "LEMAH TAPI MASIH PROFIT"
    else:
        verdict = "GAGAL"
        
    logger.info(f"VERDICT: {verdict}")

    slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "%25")
    ts = timestamp
    
    if not trades_df.empty:
        trades_csv = csv_dir / f"trades_BLIND_{slug}_{ticker.replace('.','_')}_{ts}.csv"
        trades_df.to_csv(trades_csv, index=False)
    
    equity_csv = csv_dir / f"equity_BLIND_{slug}_{ticker.replace('.','_')}_{ts}.csv"
    df_result.to_csv(equity_csv)

    # Chart 
    bh_start_price = df_result["Close"].iloc[0]
    bh_equity = capital * (df_result["Close"] / bh_start_price)
    
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df_result.index, df_result["Equity"], label=f"Strategi (Blind) ({metrics['total_return']:.1f}%)", color="#2563eb", linewidth=1.5)
    
    bh_ret = (bh_equity.iloc[-1] / capital - 1) * 100
    ax1.plot(df_result.index, bh_equity, label=f"Buy & Hold ({bh_ret:.1f}%)", color="#ef4444", linestyle="--", alpha=0.8)
    
    ax1.set_title(f"Blind Test: {name} — {ticker} (Locked Params)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Equity (Rp)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    peak_strat = df_result["Equity"].cummax()
    dd_strat = (df_result["Equity"] - peak_strat) / peak_strat * 100
    peak_bh = bh_equity.cummax()
    dd_bh = (bh_equity - peak_bh) / peak_bh * 100
    
    ax2.fill_between(df_result.index, dd_strat, 0, color="#86efac", alpha=0.7, label="DD Strategi")
    ax2.plot(df_result.index, dd_bh, color="#ef4444", linestyle="--", linewidth=1, alpha=0.7, label="DD Buy&Hold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower left", fontsize="small", ncol=2)
    
    plt.tight_layout()
    chart_path = charts_dir / f"walkforward_BLIND_{slug}_{ticker.replace('.','_')}_{ts}.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    logger.info(f"[CHART BLIND] → {chart_path}")

    return {
        "is": {},
        "oos": metrics,
        "score": 0,
        "degradation": 0,
        "verdict": verdict,
        "params": params,
        "leakage_flags": list(set(leakage_flags)) if leakage_flags else []
    }
