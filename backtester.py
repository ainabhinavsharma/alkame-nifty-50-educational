# 1. Standard library imports
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# 2. Third-party imports
import numpy as np
import pandas as pd

# 3. Local imports
from config import SLIPPAGE_BPS, TRANSACTION_COST_BPS, PREDICTION_HORIZON_BARS, configure_logging
from ensemble_manager import EnsembleManager
from runtime_validator import RuntimeValidator, CalibrationResult, EdgeCheckResult, LiveGateResult

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------
CLASS_TO_DIRECTION = {"UP": 1, "DOWN": -1, "FLAT": 0}


@dataclass
class BacktestResult:
    symbol: str
    n_test_predictions: int
    n_trades_taken: int
    strategy_cumulative_return_pct: float
    baseline_cumulative_return_pct: float
    alpha_pct: float
    edge_check_status: str
    calibration_status: str
    calibration_ece: Optional[float]
    is_live_worthy: bool
    gate_reasons: List[str] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# 6. Classes and functions
# ---------------------------------------------------------------------------
class Backtester:
    """
    Walks the ensemble forward through its own held-out test period (the same
    chronological split ensemble_manager.py already enforces), simulates one
    trade per prediction, applies slippage/transaction costs ONLY to bars
    where a trade actually happened, and compares cumulative strategy return
    against simply holding the NIFTY index over the same window. Feeds both
    a real edge-check and a real calibration dataset into runtime_validator.py.
    """

    def __init__(self, ensemble_manager: Optional[EnsembleManager] = None,
                 runtime_validator: Optional[RuntimeValidator] = None):
        self.ensemble_manager = ensemble_manager or EnsembleManager()
        self.runtime_validator = runtime_validator or RuntimeValidator()

    @staticmethod
    def _forward_return_pct(close: pd.Series, horizon: int = PREDICTION_HORIZON_BARS) -> pd.Series:
        return (close.shift(-horizon) - close) / close * 100.0

    def run_backtest_for_symbol(
        self, symbol: str, stock_df: pd.DataFrame, index_df: pd.DataFrame,
    ) -> BacktestResult:
        try:
            train_result = self.ensemble_manager.train_ensemble_for_symbol(symbol, stock_df, index_df)
            if not train_result.success:
                return BacktestResult(
                    symbol=symbol, n_test_predictions=0, n_trades_taken=0,
                    strategy_cumulative_return_pct=0.0, baseline_cumulative_return_pct=0.0, alpha_pct=0.0,
                    edge_check_status="NO_EDGE", calibration_status="INSUFFICIENT_DATA", calibration_ece=None,
                    is_live_worthy=False, success=False,
                    error=f"Ensemble training failed: {train_result.error}",
                )

            # Recover the exact same chronological test split the ensemble was evaluated on.
            prepared = self.ensemble_manager.model_trainer.prepare_dataset(stock_df, index_df)
            if prepared is None:
                return BacktestResult(
                    symbol=symbol, n_test_predictions=0, n_trades_taken=0,
                    strategy_cumulative_return_pct=0.0, baseline_cumulative_return_pct=0.0, alpha_pct=0.0,
                    edge_check_status="NO_EDGE", calibration_status="INSUFFICIENT_DATA", calibration_ece=None,
                    is_live_worthy=False, success=False, error="Dataset preparation failed for backtest.",
                )
            X, y, _ = prepared
            _, X_test, _, y_test = self.ensemble_manager.model_trainer.time_based_split(X, y)

            if len(X_test) == 0:
                return BacktestResult(
                    symbol=symbol, n_test_predictions=0, n_trades_taken=0,
                    strategy_cumulative_return_pct=0.0, baseline_cumulative_return_pct=0.0, alpha_pct=0.0,
                    edge_check_status="NO_EDGE", calibration_status="INSUFFICIENT_DATA", calibration_ece=None,
                    is_live_worthy=False, success=False, error="Test set is empty — nothing to backtest.",
                )

            ensemble_predictions = self.ensemble_manager.predict(symbol, X_test)
            if ensemble_predictions is None:
                return BacktestResult(
                    symbol=symbol, n_test_predictions=0, n_trades_taken=0,
                    strategy_cumulative_return_pct=0.0, baseline_cumulative_return_pct=0.0, alpha_pct=0.0,
                    edge_check_status="NO_EDGE", calibration_status="INSUFFICIENT_DATA", calibration_ece=None,
                    is_live_worthy=False, success=False, error="Ensemble prediction failed on test set.",
                )

            # Actual forward returns for the stock (for P&L) and the index (for baseline), same horizon.
            stock_forward_return = self._forward_return_pct(stock_df["Close"]).reindex(X_test.index)
            index_forward_return = self._forward_return_pct(index_df["Close"]).reindex(X_test.index)

            total_cost_pct = (SLIPPAGE_BPS + TRANSACTION_COST_BPS) / 100.0  # bps -> %

            per_trade_net_returns = []
            n_trades_taken = 0
            calibration_rows = []

            for i, ts in enumerate(X_test.index):
                pred = ensemble_predictions[i]
                direction = CLASS_TO_DIRECTION.get(pred.predicted_class, 0)
                raw_fwd_return = stock_forward_return.loc[ts]

                if direction == 0 or pd.isna(raw_fwd_return):
                    # FLAT prediction, or no future data to resolve (tail of dataset) -> no trade, no cost.
                    net_return = 0.0
                else:
                    n_trades_taken += 1
                    gross_return = direction * raw_fwd_return  # long profits from up moves, short profits from down moves
                    net_return = gross_return - total_cost_pct  # cost only charged when a trade actually happens

                per_trade_net_returns.append(net_return)

                actual_class = y_test.iloc[i]
                calibration_rows.append({
                    "confidence": pred.confidence,
                    "correct": bool(pred.predicted_class == actual_class),
                })

            strategy_returns = pd.Series(per_trade_net_returns, index=X_test.index)

            # Edge check: costs are already netted in per-trade returns above, so pass zero
            # additional cost here — otherwise costs would be deducted twice.
            edge_result = self.runtime_validator.compute_edge_vs_baseline(
                strategy_returns, index_forward_return.fillna(0.0), slippage_bps=0, transaction_cost_bps=0,
            )

            calibration_df = pd.DataFrame(calibration_rows)
            calibration_result = self.runtime_validator.compute_calibration(calibration_df)

            gate = self.runtime_validator.validate_before_live(calibration_result, edge_result)

            return BacktestResult(
                symbol=symbol, n_test_predictions=len(X_test), n_trades_taken=n_trades_taken,
                strategy_cumulative_return_pct=edge_result.strategy_cumulative_return_pct,
                baseline_cumulative_return_pct=edge_result.baseline_cumulative_return_pct,
                alpha_pct=edge_result.alpha_pct, edge_check_status=edge_result.status,
                calibration_status=calibration_result.status,
                calibration_ece=calibration_result.expected_calibration_error,
                is_live_worthy=(gate.safe_to_show_calibrated_confidence and gate.safe_to_treat_as_live_edge),
                gate_reasons=gate.reasons, success=True,
            )

        except Exception as e:
            logger.error(f"Backtest failed for {symbol}: {e}")
            return BacktestResult(
                symbol=symbol, n_test_predictions=0, n_trades_taken=0,
                strategy_cumulative_return_pct=0.0, baseline_cumulative_return_pct=0.0, alpha_pct=0.0,
                edge_check_status="NO_EDGE", calibration_status="INSUFFICIENT_DATA", calibration_ece=None,
                is_live_worthy=False, success=False, error=str(e),
            )

    def run_backtest_for_all_symbols(
        self, symbol_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    ) -> Dict[str, BacktestResult]:
        """symbol_data maps symbol -> (stock_df, index_df). Runs the full
        per-symbol backtest for each and returns all results keyed by symbol."""
        results: Dict[str, BacktestResult] = {}
        for symbol, (stock_df, index_df) in symbol_data.items():
            results[symbol] = self.run_backtest_for_symbol(symbol, stock_df, index_df)
        n_live_worthy = sum(1 for r in results.values() if r.is_live_worthy)
        logger.info(f"Backtested {len(results)} symbol(s); {n_live_worthy} currently live-worthy.")
        return results


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configure_logging(log_filename="backtester_selftest.log")
    logger.info("Running backtester.py self-test...")

    def _build_synthetic_ohlcv(n_days: int = 40, bars_per_day: int = 75, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows, timestamps = [], []
        price = 1000.0
        base_date = pd.Timestamp("2026-01-05 09:15:00")
        recent_closes = []
        for day in range(n_days):
            day_start = base_date + pd.Timedelta(days=day)
            for bar in range(bars_per_day):
                ts = day_start + pd.Timedelta(minutes=5 * bar)
                if len(recent_closes) >= 10:
                    trend = recent_closes[-1] - recent_closes[-10]
                    bias = 1.5 if trend < -6 else (-1.5 if trend > 6 else 0.0)
                else:
                    bias = 0.0
                drift = rng.normal(bias, 1.5)
                price = max(1.0, price + drift)
                open_p = price
                close_p = max(1.0, price + rng.normal(bias * 0.5, 1.0))
                high_p = max(open_p, close_p) + abs(rng.normal(0, 0.5))
                low_p = min(open_p, close_p) - abs(rng.normal(0, 0.5))
                vol = int(abs(rng.normal(50000, 15000)))
                rows.append([open_p, high_p, low_p, close_p, vol])
                timestamps.append(ts)
                price = close_p
                recent_closes.append(close_p)
        return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex(timestamps))

    test_symbol = "BACKTEST_SYNTH"  # single test symbol allowed in the __main__ block only

    try:
        print("\n=== BACKTESTER SELF-TEST RESULT ===")
        backtester = Backtester()

        stock_df = _build_synthetic_ohlcv(seed=42)
        index_df = _build_synthetic_ohlcv(seed=99)
        index_df.index = stock_df.index

        result = backtester.run_backtest_for_symbol(test_symbol, stock_df, index_df)

        print(f"Success: {result.success}")
        if not result.success:
            print(f"Error: {result.error}")
        print(f"Test predictions: {result.n_test_predictions}, Trades actually taken: {result.n_trades_taken}")
        print(f"Strategy cumulative return: {result.strategy_cumulative_return_pct:.2f}%")
        print(f"Baseline (NIFTY buy-hold) cumulative return: {result.baseline_cumulative_return_pct:.2f}%")
        print(f"Alpha: {result.alpha_pct:.2f}%, edge_check_status={result.edge_check_status}")
        print(f"Calibration status: {result.calibration_status}, ECE: {result.calibration_ece}")
        print(f"Is live-worthy: {result.is_live_worthy}")
        print(f"Gate reasons: {result.gate_reasons}")

        assert result.success, "Backtest pipeline should complete successfully on valid synthetic data"
        assert result.n_test_predictions > 0
        assert result.edge_check_status in ("EDGE_CONFIRMED", "NO_EDGE")
        assert result.calibration_status in ("SUFFICIENT", "INSUFFICIENT_DATA")
        # With ~580 test rows from a 40-day synthetic set, calibration data should clearly be sufficient
        assert result.calibration_status == "SUFFICIENT", "Expected enough test rows for calibration to be sufficient"
        # is_live_worthy must be perfectly consistent with the two underlying statuses — never a third, independent answer
        expected_live_worthy = (result.calibration_status == "SUFFICIENT" and result.edge_check_status == "EDGE_CONFIRMED")
        # Note: calibration ALSO requires is_well_calibrated, which calibration_status alone doesn't capture,
        # so we only assert the weaker necessary condition here rather than full equivalence.
        if result.is_live_worthy:
            assert result.edge_check_status == "EDGE_CONFIRMED"
            assert result.calibration_status == "SUFFICIENT"

        print("STATUS: PASS")
        logger.info("backtester.py self-test passed.")

    except AssertionError as ae:
        logger.error(f"backtester.py self-test assertion failed: {ae}")
        print(f"STATUS: FAIL — {ae}")
    except Exception as e:
        logger.error(f"backtester.py self-test crashed: {e}")
        print(f"STATUS: FAIL — {e}")
