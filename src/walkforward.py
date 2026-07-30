"""
walkforward.py — Orkestrator walk-forward IS/OOS backtest.

Alur eksekusi per strategi:
  1. Grid search di IS → pilih best_params berdasarkan Sharpe
  2. Kunci best_params
  3. Jalankan IS ulang dengan best_params (untuk chart dan metrik final)
  4. Jalankan OOS dengan best_params yang SAMA PERSIS (tidak ada re-optimasi!)
  5. Hitung consistency verdict (skor degradasi IS→OOS)
  6. Cek data leakage
  7. Simpan chart, CSV trades, equity curve
"""

from pathlib import Path
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend untuk terminal
import matplotlib.pyplot as plt
import pandas as pd

from src.engine import backtest, check_leakage
from src.metrics import compute_metrics, consistency_verdict


# ------------------------------------------------------------------ #
# Main walk-forward function
# ------------------------------------------------------------------ #
def run_walkforward(
    name: str,
    grid_fn: Callable,
    signal_fn: Callable,
    is_data: pd.DataFrame,
    oos_data: pd.DataFrame,
    split_date,
    capital: float,
    ticker: str,
    timestamp: str,
    logger,
    results_dir: str = "results",
    raw_data: pd.DataFrame = None,   # Full raw data untuk Buy & Hold benchmark
    engine_fn: Callable = None,      # Engine function: backtest (v1) atau backtest_v2
) -> Optional[dict]:
    """
    Jalankan satu siklus walk-forward untuk satu strategi dan satu ticker.

    Args:
        name      : Nama strategi untuk label di output.
        grid_fn   : Fungsi grid search.
        signal_fn : Fungsi sinyal.
        is_data   : DataFrame In-Sample.
        oos_data  : DataFrame Out-of-Sample.
        split_date: Tanggal pemisah IS/OOS (untuk chart).
        capital   : Modal awal.
        ticker    : Kode saham.
        timestamp : String timestamp session.
        logger    : Instance BacktestLogger.
        results_dir: Folder root output.
        raw_data  : (opsional) Full data untuk Buy & Hold benchmark.
        engine_fn : (opsional) Fungsi backtest: default=backtest v1.
                    Gunakan backtest_v2 untuk ATR trailing stop.
    """
    if engine_fn is None:
        engine_fn = backtest   # default: v1 (kompatibel dengan prd.md)
    charts_dir = Path(results_dir) / "charts"
    csv_dir    = Path(results_dir) / "csv"
    charts_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Slug untuk nama file
    strat_slug  = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    ticker_slug = ticker.replace(".", "_")

    # ---------------------------------------------------------------- #
    # 1. Grid Search — HANYA di IS data
    # ---------------------------------------------------------------- #
    logger.info(f"\n{'='*70}")
    logger.info(f"  GRID SEARCH (In-Sample) — {name} [{ticker}]")
    logger.info(f"{'='*70}")

    results = grid_fn(is_data, capital)

    if not results:
        logger.warning(
            f"[{ticker}][{name}] Tidak ada kombinasi parameter valid "
            f"(semua < {5} trade di IS). Coba perpanjang periode data."
        )
        return None

    # Tampilkan top-3
    logger.info(f"  Top-3 parameter (ranked by Sharpe IS):")
    for i, r in enumerate(results[:3], 1):
        logger.info(
            f"  #{i} params={r['params']}  "
            f"Sharpe(IS)={r['sharpe']:.2f}  "
            f"Return(IS)={r['total_return']:.1f}%  "
            f"Trades={r['n_trades']}"
        )

    # ---------------------------------------------------------------- #
    # 2. Kunci best_params dari IS
    # ---------------------------------------------------------------- #
    best_params = results[0]["params"]
    logger.info(f"\n>>> Parameter terkunci untuk OOS: {best_params}")

    # Pisahkan parameter engine (tidak diteruskan ke signal_fn)
    # atr_mult adalah parameter backtest_v2, bukan signal
    ENGINE_ONLY_PARAMS = {"atr_mult"}
    signal_params = {k: v for k, v in best_params.items() if k not in ENGINE_ONLY_PARAMS}
    engine_kwargs  = {k: v for k, v in best_params.items() if k in ENGINE_ONLY_PARAMS}

    # ---------------------------------------------------------------- #
    # 3. Jalankan ulang IS dengan best_params (metrik final + chart)
    # ---------------------------------------------------------------- #
    # IS run dengan best_params
    df_is, em_is, xm_is = signal_fn(is_data, **signal_params)
    leakage_is           = check_leakage(df_is, em_is, xm_is)
    df_is, trades_is     = engine_fn(df_is, em_is, xm_is, capital, **engine_kwargs)
    is_m                 = compute_metrics(df_is, trades_is, capital)

    # OOS run - PARAMETER SAMA PERSIS, tidak ada re-optimasi
    df_oos, em_oos, xm_oos = signal_fn(oos_data, **signal_params)
    leakage_oos            = check_leakage(df_oos, em_oos, xm_oos)
    df_oos, trades_oos     = engine_fn(df_oos, em_oos, xm_oos, capital, **engine_kwargs)
    oos_m                  = compute_metrics(df_oos, trades_oos, capital)

    # ---------------------------------------------------------------- #
    # 5. Leakage report
    # ---------------------------------------------------------------- #
    leakage_flags = leakage_is + leakage_oos
    if leakage_flags:
        for flag in leakage_flags:
            logger.warning(f"[LEAKAGE] {flag}")
    else:
        logger.info(f"[LEAKAGE CHECK] OK — tidak ada data leakage terdeteksi")

    # ---------------------------------------------------------------- #
    # 6. Consistency verdict
    # ---------------------------------------------------------------- #
    score, degradation, verdict = consistency_verdict(is_m, oos_m)
    score_label = "N/A" if score is None else f"{score:.0f}"

    # ---------------------------------------------------------------- #
    # 7. Print hasil tabel
    # ---------------------------------------------------------------- #
    sep = "-" * 70
    logger.info(f"\n{sep}")
    logger.info(f"  HASIL: {name} [{ticker}]")
    logger.info(sep)
    logger.info(f"{'Metrik':<20}{'In-Sample':>15}{'Out-of-Sample':>18}")
    logger.info(f"{'Jumlah trade':<20}{is_m['n_trades']:>15}{oos_m['n_trades']:>18}")
    logger.info(f"{'Win rate':<20}{is_m['win_rate']:>14.1f}%{oos_m['win_rate']:>17.1f}%")
    logger.info(f"{'Total return':<20}{is_m['total_return']:>14.1f}%{oos_m['total_return']:>17.1f}%")
    logger.info(f"{'CAGR':<20}{is_m['cagr']:>14.1f}%{oos_m['cagr']:>17.1f}%")
    logger.info(f"{'Sharpe':<20}{is_m['sharpe']:>15.2f}{oos_m['sharpe']:>18.2f}")
    logger.info(f"{'Max drawdown':<20}{is_m['max_dd']:>14.1f}%{oos_m['max_dd']:>17.1f}%")
    logger.info(f"{'Calmar':<20}{is_m['calmar']:>15.2f}{oos_m['calmar']:>18.2f}")
    logger.info(f"{'Profit Factor':<20}{is_m['profit_factor']:>15.2f}{oos_m['profit_factor']:>18.2f}")
    logger.info(f"{'Avg Hold (days)':<20}{is_m['avg_hold_days']:>15.1f}{oos_m['avg_hold_days']:>18.1f}")
    logger.info(sep)
    if score is None:
        logger.info(f"  {verdict}")
    else:
        logger.info(
            f"  Skor Ketepatan (IS→OOS): {score_label}/100  "
            f"(degradasi Sharpe: {degradation:.0f}%)"
        )
        logger.info(f"  {verdict}")
    logger.info("=" * 70)

    # ---------------------------------------------------------------- #
    # 8. Export CSV
    # ---------------------------------------------------------------- #
    if not trades_is.empty:
        fp = csv_dir / f"trades_IS_{strat_slug}_{ticker_slug}_{timestamp}.csv"
        trades_is.to_csv(fp, index=False)
        logger.info(f"[CSV] IS trades → {fp}")

    if not trades_oos.empty:
        fp = csv_dir / f"trades_OOS_{strat_slug}_{ticker_slug}_{timestamp}.csv"
        trades_oos.to_csv(fp, index=False)
        logger.info(f"[CSV] OOS trades → {fp}")

    # ---------------------------------------------------------------- #
    # 8a. Hitung Buy & Hold benchmark dari harga real
    # ---------------------------------------------------------------- #
    # Gunakan df_is / df_oos (setelah dropna indikator) agar index sama dengan equity curve
    bh_is_prices  = df_is["Close"]
    bh_oos_prices = df_oos["Close"]

    # Normalisasi: modal awal = capital di titik pertama IS
    bh_is_equity  = (bh_is_prices / bh_is_prices.iloc[0]) * capital
    # OOS Buy & Hold: dimulai dari nilai IS akhir (bukan capital baru)
    bh_oos_start  = df_is["Equity"].iloc[-1]
    bh_oos_equity = (bh_oos_prices / bh_oos_prices.iloc[0]) * bh_oos_start

    bh_is_ret  = (bh_is_prices.iloc[-1]  / bh_is_prices.iloc[0]  - 1) * 100
    bh_oos_ret = (bh_oos_prices.iloc[-1] / bh_oos_prices.iloc[0] - 1) * 100

    # ---------------------------------------------------------------- #
    # 8b. Export CSV (tambahkan kolom Buy & Hold)
    # ---------------------------------------------------------------- #
    eq_is  = df_is[["Equity"]].copy()
    eq_is["BuyHold"] = bh_is_equity.values
    eq_is["phase"]   = "IS"

    eq_oos = df_oos[["Equity"]].copy()
    eq_oos["BuyHold"] = bh_oos_equity.values
    eq_oos["phase"]   = "OOS"

    equity_combined = pd.concat([eq_is, eq_oos])
    fp = csv_dir / f"equity_{strat_slug}_{ticker_slug}_{timestamp}.csv"
    equity_combined.to_csv(fp)
    logger.info(f"[CSV] Equity curve → {fp}")

    # ---------------------------------------------------------------- #
    # 9. Chart — 3 panel: Equity, Drawdown, Return Comparison
    # ---------------------------------------------------------------- #
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 12),
        gridspec_kw={"height_ratios": [3, 1.2, 1]}
    )
    ax_eq, ax_dd, ax_ret = axes

    # ── Panel 1: Equity Curve ──────────────────────────────────────── #
    strat_is_ret  = is_m["total_return"]
    strat_oos_ret = oos_m["total_return"]

    ax_eq.plot(
        df_is.index, df_is["Equity"], color="#2563eb", linewidth=2.0,
        label=f"Strategi IS  ({strat_is_ret:+.1f}%)"
    )
    oos_norm = df_oos["Equity"] / df_oos["Equity"].iloc[0] * df_is["Equity"].iloc[-1]
    ax_eq.plot(
        df_oos.index, oos_norm, color="#16a34a", linewidth=2.0,
        label=f'Strategi OOS  ({strat_oos_ret:+.1f}%)  ← "forecast"'
    )

    # Buy & Hold benchmark (harga asli yang sebenarnya terjadi)
    ax_eq.plot(
        bh_is_equity.index, bh_is_equity, color="#f59e0b", linewidth=1.4,
        linestyle="--", alpha=0.85,
        label=f"Buy & Hold IS  ({bh_is_ret:+.1f}%)"
    )
    ax_eq.plot(
        bh_oos_equity.index, bh_oos_equity, color="#dc2626", linewidth=1.4,
        linestyle="--", alpha=0.85,
        label=f"Buy & Hold OOS  ({bh_oos_ret:+.1f}%)  ← harga real"
    )

    ax_eq.axvline(
        split_date, color="#6b7280", linestyle=":", linewidth=1.5,
        label=f"Split IS/OOS  ({str(split_date.date())})"
    )
    ax_eq.axhline(capital, color="#9ca3af", linestyle="-", linewidth=0.8, alpha=0.5)

    verdict_short = verdict.split("—")[0].strip() if "—" in verdict else verdict[:30]
    ax_eq.set_title(
        f"Walk-Forward: {name} — {ticker}\n"
        f"Skor Ketepatan: {score_label}/100 | {verdict_short}",
        fontsize=12, fontweight="bold"
    )
    ax_eq.set_ylabel("Equity (Rp)")
    ax_eq.legend(fontsize=9, loc="upper left")
    ax_eq.grid(alpha=0.25)
    ax_eq.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"Rp{x/1e6:.2f}M")
    )

    # ── Panel 2: Drawdown ─────────────────────────────────────────── #
    dd_is  = (df_is["Equity"]   / df_is["Equity"].cummax()   - 1) * 100
    dd_oos = (df_oos["Equity"]  / df_oos["Equity"].cummax()  - 1) * 100
    dd_bh_is  = (bh_is_equity   / bh_is_equity.cummax()   - 1) * 100
    dd_bh_oos = (bh_oos_equity  / bh_oos_equity.cummax()  - 1) * 100

    ax_dd.fill_between(df_is.index,  dd_is,     0, color="#2563eb", alpha=0.35, label="DD Strategi IS")
    ax_dd.fill_between(df_oos.index, dd_oos,    0, color="#16a34a", alpha=0.35, label="DD Strategi OOS")
    ax_dd.plot(bh_is_equity.index,  dd_bh_is,  color="#f59e0b", linewidth=1.0,
               linestyle="--", alpha=0.7, label="DD Buy&Hold IS")
    ax_dd.plot(bh_oos_equity.index, dd_bh_oos, color="#dc2626", linewidth=1.0,
               linestyle="--", alpha=0.7, label="DD Buy&Hold OOS")
    ax_dd.axvline(split_date, color="#6b7280", linestyle=":", linewidth=1.2)
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.legend(fontsize=7, ncol=4)
    ax_dd.grid(alpha=0.25)

    # ── Panel 3: Return Comparison Bar Chart ──────────────────────── #
    labels    = ["IS\nStrategi", "IS\nBuy&Hold", "OOS\nStrategi", "OOS\nBuy&Hold"]
    returns   = [strat_is_ret, bh_is_ret, strat_oos_ret, bh_oos_ret]
    bar_colors = []
    for v in returns:
        bar_colors.append("#16a34a" if v >= 0 else "#dc2626")

    bars = ax_ret.bar(labels, returns, color=bar_colors, alpha=0.8, width=0.5, edgecolor="white")
    for bar, val in zip(bars, returns):
        ax_ret.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.3 if val >= 0 else -1.5),
            f"{val:+.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )
    ax_ret.axhline(0, color="#374151", linewidth=0.8)
    ax_ret.set_ylabel("Total Return (%)")
    ax_ret.set_title("Perbandingan Return: Strategi vs Harga Real (Buy & Hold)", fontsize=10)
    ax_ret.grid(axis="y", alpha=0.25)
    ax_ret.set_xlabel("")

    plt.tight_layout(pad=2.0)
    chart_path = charts_dir / f"walkforward_{strat_slug}_{ticker_slug}_{timestamp}.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    logger.info(f"[CHART] → {chart_path}")
    logger.info(
        f"[BUY&HOLD] IS={bh_is_ret:+.1f}% vs Strategi IS={strat_is_ret:+.1f}% | "
        f"OOS={bh_oos_ret:+.1f}% vs Strategi OOS={strat_oos_ret:+.1f}%"
    )

    return {
        "name":          name,
        "ticker":        ticker,
        "is":            is_m,
        "oos":           oos_m,
        "score":         score,
        "degradation":   degradation,
        "verdict":       verdict,
        "params":        best_params,
        "leakage_flags": leakage_flags,
    }
