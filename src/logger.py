"""
logger.py — Structured logging: SQLite + rotating .log file
Mencatat setiap run backtest beserta metadata, metrik IS/OOS, dan leakage flags.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path


class BacktestLogger:
    """
    Logger terpusat untuk sesi backtest.

    Output:
      - results/backtest_results.db  → SQLite (machine-readable, bisa di-query)
      - results/logs/backtest_YYYYMMDD_HHMMSS.log → Human-readable per session
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.logs_dir = self.results_dir / "logs"
        self.charts_dir = self.results_dir / "charts"
        self.csv_dir = self.results_dir / "csv"

        for d in [self.logs_dir, self.charts_dir, self.csv_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.db_path = self.results_dir / "backtest_results.db"
        self._init_db()

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{self.timestamp}"
        self.log_path = self.logs_dir / f"backtest_{self.timestamp}.log"

        # Setup Python logger (file + console)
        self._logger = logging.getLogger(f"backtest_{self.timestamp}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        fh = logging.FileHandler(self.log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        self._logger.addHandler(fh)
        self._logger.addHandler(ch)

        self._logger.info(f"=== Backtest Session Start: {self.run_id} ===")
        self._logger.info(f"Log : {self.log_path}")
        self._logger.info(f"DB  : {self.db_path}")

    # ------------------------------------------------------------------ #
    # Public logging interface
    # ------------------------------------------------------------------ #
    def info(self, msg: str):
        self._logger.info(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    # ------------------------------------------------------------------ #
    # SQLite schema + save
    # ------------------------------------------------------------------ #
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id              TEXT NOT NULL,
                timestamp           TEXT NOT NULL,
                ticker              TEXT NOT NULL,
                strategy            TEXT NOT NULL,
                start_date          TEXT,
                end_date            TEXT,
                is_ratio            REAL,
                capital             REAL,
                best_params         TEXT,          -- JSON string
                -- In-Sample metrics
                is_n_trades         INTEGER,
                is_win_rate         REAL,
                is_total_return     REAL,
                is_cagr             REAL,
                is_sharpe           REAL,
                is_max_dd           REAL,
                is_profit_factor    REAL,
                -- Out-of-Sample metrics
                oos_n_trades        INTEGER,
                oos_win_rate        REAL,
                oos_total_return    REAL,
                oos_cagr            REAL,
                oos_sharpe          REAL,
                oos_max_dd          REAL,
                oos_profit_factor   REAL,
                -- Consistency
                consistency_score   REAL,          -- NULL jika INCONCLUSIVE
                sharpe_degradation  REAL,          -- NULL jika INCONCLUSIVE
                verdict             TEXT,
                leakage_flags       TEXT,          -- JSON array
                UNIQUE(run_id, ticker, strategy)
            )
        """)
        conn.commit()
        conn.close()

    def save_run(
        self,
        ticker: str,
        strategy: str,
        start: str,
        end: str,
        is_ratio: float,
        capital: float,
        best_params: dict,
        is_m: dict,
        oos_m: dict,
        score,           # float or None
        degradation,     # float or None
        verdict: str,
        leakage_flags: list = None,
    ):
        """Simpan satu hasil run ke SQLite."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute("""
                INSERT OR REPLACE INTO runs (
                    run_id, timestamp, ticker, strategy, start_date, end_date,
                    is_ratio, capital, best_params,
                    is_n_trades, is_win_rate, is_total_return, is_cagr,
                    is_sharpe, is_max_dd, is_profit_factor,
                    oos_n_trades, oos_win_rate, oos_total_return, oos_cagr,
                    oos_sharpe, oos_max_dd, oos_profit_factor,
                    consistency_score, sharpe_degradation, verdict, leakage_flags
                ) VALUES (
                    ?,?,?,?,?,?,?,?,?,
                    ?,?,?,?,?,?,?,
                    ?,?,?,?,?,?,?,
                    ?,?,?,?
                )
            """, (
                self.run_id, self.timestamp, ticker, strategy, start, end,
                is_ratio, capital, json.dumps(best_params),
                is_m.get("n_trades", 0), is_m.get("win_rate", 0.0), is_m.get("total_return", 0.0),
                is_m.get("cagr", 0.0), is_m.get("sharpe", 0.0), is_m.get("max_dd", 0.0),
                is_m.get("profit_factor", 0.0),
                oos_m.get("n_trades", 0), oos_m.get("win_rate", 0.0), oos_m.get("total_return", 0.0),
                oos_m.get("cagr", 0.0), oos_m.get("sharpe", 0.0), oos_m.get("max_dd", 0.0),
                oos_m.get("profit_factor", 0.0),
                score, degradation, verdict,
                json.dumps(leakage_flags or [])
            ))
            conn.commit()
            self._logger.info(f"[DB] Saved → ticker={ticker} strategy={strategy}")
        except Exception as e:
            self._logger.error(f"[DB] Error saving run: {e}")
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Summary comparison table
    # ------------------------------------------------------------------ #
    def print_comparison_table(self, summary: list):
        """Cetak tabel perbandingan semua ticker & strategi di akhir session."""
        if not summary:
            return

        self._logger.info(f"\n{'='*95}")
        self._logger.info(f"  RINGKASAN PERBANDINGAN MULTI-TICKER WALK-FORWARD")
        self._logger.info(f"{'='*95}")

        header = (
            f"{'Ticker':<12}{'Strategi':<32}{'Skor':>6}"
            f"{'OOS Ret':>10}{'OOS Shrp':>10}{'OOS WR':>9}{'Leakage':>9}  Verdict"
        )
        self._logger.info(header)
        self._logger.info("-" * 95)

        # Sort by score descending (INCONCLUSIVE last)
        def sort_key(r):
            return r["score"] if r["score"] is not None else -1

        for r in sorted(summary, key=sort_key, reverse=True):
            score_str = "  N/A" if r["score"] is None else f"{r['score']:>5.0f}"
            leak_str = "YES" if r.get("leakage_flags") else " ok"
            verdict_word = r["verdict"].split("—")[0].strip() if "—" in r["verdict"] else r["verdict"][:20]
            line = (
                f"{r['ticker']:<12}{r['name']:<32}{score_str:>6}"
                f"{r['oos']['total_return']:>9.1f}%"
                f"{r['oos']['sharpe']:>10.2f}"
                f"{r['oos']['win_rate']:>8.1f}%"
                f"{leak_str:>9}"
                f"  {verdict_word}"
            )
            self._logger.info(line)

        self._logger.info("=" * 95)
