# -*- coding: utf-8 -*-
"""
回测引擎 — Backtrader 策略封装、单/多股票回测、多空策略对比
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd

from us_stock_predictor.config.core import (
    BACKTEST_CONFIG, MODEL_PATHS, MODEL_CONFIG, FEATURE_COLUMNS,
)
from us_stock_predictor.utils.core import handle_exception, get_device, check_model_exists

try:
    import backtrader as bt
    _BT_AVAILABLE = True
except ImportError:
    _BT_AVAILABLE = False
    logging.warning("[警告] backtrader 未安装，回测功能不可用")


# ─── 辅助：特征计算 ──────────────────────────────────────────
def _compute_features_for_backtest(price_history: pd.DataFrame) -> Optional[np.ndarray]:
    """将历史价格 DataFrame 转换为模型输入张量 [1, seq_len, 35]"""
    try:
        from us_stock_predictor.data.collector import USStockDataCollector
        collector = USStockDataCollector()
        df = collector.calculate_technical_indicators(price_history.copy())
        seq_len = MODEL_CONFIG["seq_len"]
        feat_cols = FEATURE_COLUMNS

        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0.0

        if len(df) < seq_len:
            return None

        window = df[feat_cols].tail(seq_len).values.astype(np.float32)
        return window[np.newaxis, ...]  # [1, seq_len, 35]
    except Exception as e:
        handle_exception(e, "计算回测特征")
        return None


def _load_model_for_backtest(model_path: str = None):
    """加载模型，失败则返回 None"""
    path = model_path or MODEL_PATHS["best_model"]
    if not check_model_exists(path):
        return None, None
    try:
        from us_stock_predictor.models.transformer_surge import load_model
        device = get_device()
        model = load_model(path, device=device)
        return model, device
    except Exception as e:
        handle_exception(e, "加载回测模型")
        return None, None


# ─── Backtrader 策略 ─────────────────────────────────────────
if _BT_AVAILABLE:
    class MLSurgeSingleStockStrategy(bt.Strategy):
        """
        基于四分类 AI 预测的单股票回测策略。
        支持 long_only / short_only / long_short 三种模式。
        """
        params = (
            ("model_path", None),
            ("trading_mode", "long_short"),
            ("signal_threshold", BACKTEST_CONFIG["four_class_strategy"]["signal_threshold"]),
            ("min_confidence", BACKTEST_CONFIG["four_class_strategy"]["min_confidence"]),
            ("profit_target", BACKTEST_CONFIG["profit_target"]),
            ("stop_loss", BACKTEST_CONFIG["stop_loss"]),
            ("hold_days", BACKTEST_CONFIG["hold_days"]),
            ("stake_pct", BACKTEST_CONFIG["stake_pct"]),
        )

        def __init__(self):
            self.model = None
            self.device = None
            self._load_model()
            self.order = None
            self.entry_price = 0.0
            self.hold_count = 0
            self.long_trades = 0
            self.short_trades = 0
            self.wins = 0
            self.total_closed = 0

        def _load_model(self):
            self.model, self.device = _load_model_for_backtest(self.p.model_path)

        def _get_prediction(self) -> Dict[str, float]:
            try:
                if self.model is None:
                    return {}
                dates = [self.data.datetime.date(-i) for i in range(60) if len(self.data) > i]
                prices = {
                    "date": [self.data.datetime.date(-i) for i in range(min(60, len(self.data)))],
                    "open": [self.data.open[-i] for i in range(min(60, len(self.data)))],
                    "high": [self.data.high[-i] for i in range(min(60, len(self.data)))],
                    "low": [self.data.low[-i] for i in range(min(60, len(self.data)))],
                    "close": [self.data.close[-i] for i in range(min(60, len(self.data)))],
                    "volume": [self.data.volume[-i] for i in range(min(60, len(self.data)))],
                }
                df = pd.DataFrame(prices)
                df.set_index("date", inplace=True)
                df = df.iloc[::-1]

                features = _compute_features_for_backtest(df)
                if features is None:
                    return {}

                import torch
                x = torch.tensor(features, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    logits, mag = self.model(x)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                cfg = BACKTEST_CONFIG["four_class_strategy"]
                long_sig = (probs[3] * cfg["big_surge_weight"] + probs[2] * cfg["small_surge_weight"]
                            - probs[0] * cfg["big_drop_weight"] - probs[1] * cfg["small_drop_weight"])
                short_sig = (probs[0] * BACKTEST_CONFIG["short_strategy"]["big_drop_weight"]
                             + probs[1] * BACKTEST_CONFIG["short_strategy"]["small_drop_weight"]
                             - probs[3] * BACKTEST_CONFIG["short_strategy"]["big_surge_weight"]
                             - probs[2] * BACKTEST_CONFIG["short_strategy"]["small_surge_weight"])

                return {
                    "probs": probs.tolist(),
                    "long_signal": float(long_sig),
                    "short_signal": float(short_sig),
                    "confidence": float(probs.max()),
                }
            except Exception as e:
                handle_exception(e, "获取预测信号")
                return {}

        def next(self):
            pred = self._get_prediction()
            if not pred:
                return

            cfg_4c = BACKTEST_CONFIG["four_class_strategy"]
            thr = self.p.signal_threshold
            min_conf = self.p.min_confidence
            conf = pred.get("confidence", 0)
            long_sig = pred.get("long_signal", 0)
            short_sig = pred.get("short_signal", 0)

            # ── 持仓管理 ────────────────────────────────────
            if self.position:
                self.hold_count += 1
                price = self.data.close[0]
                pnl_pct = (price - self.entry_price) / max(self.entry_price, 1e-8)
                if self.position.size > 0:
                    pnl_pct_dir = pnl_pct
                else:
                    pnl_pct_dir = -pnl_pct

                if (pnl_pct_dir >= self.p.profit_target or
                        pnl_pct_dir <= self.p.stop_loss or
                        self.hold_count >= self.p.hold_days):
                    self.close()
                    self.total_closed += 1
                    if pnl_pct_dir > 0:
                        self.wins += 1
                    self.hold_count = 0
                return

            # ── 开仓信号 ────────────────────────────────────
            if conf < min_conf:
                return

            cash = self.broker.getcash()
            price = self.data.close[0]
            if price <= 0:
                return
            size = int(cash * self.p.stake_pct / price)
            if size <= 0:
                return

            mode = self.p.trading_mode
            if long_sig > thr and mode in ("long_only", "long_short"):
                self.buy(size=size)
                self.entry_price = price
                self.hold_count = 0
                self.long_trades += 1
            elif short_sig > thr and mode in ("short_only", "long_short") and BACKTEST_CONFIG["enable_short_selling"]:
                self.sell(size=size)
                self.entry_price = price
                self.hold_count = 0
                self.short_trades += 1

        def stop(self):
            total = self.long_trades + self.short_trades
            win_rate = self.wins / max(self.total_closed, 1) * 100
            logging.info(
                f"[策略统计] 多头={self.long_trades} 空头={self.short_trades} "
                f"胜率={win_rate:.1f}%"
            )

else:
    class MLSurgeSingleStockStrategy:
        """Backtrader 不可用时的占位类"""
        pass


# ─── 单股票回测 ──────────────────────────────────────────────
def run_single_stock_backtest(
    symbol: str,
    start_date: str,
    end_date: str = None,
    model_path: str = None,
    trading_mode: str = "long_only",
) -> Optional[Dict[str, Any]]:
    """运行单只股票回测，返回结果字典"""
    if not _BT_AVAILABLE:
        logging.warning("[跳过] backtrader 未安装")
        return None

    try:
        from us_stock_predictor.data.collector import USStockDataCollector
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        collector = USStockDataCollector()
        df = collector.get_stock_data(symbol, start_date, end_date)
        if df.empty:
            logging.warning(f"[回测] {symbol} 无数据")
            return None

        df = df[["open", "high", "low", "close", "volume"]].dropna()

        cerebro = bt.Cerebro()
        cerebro.broker.setcash(BACKTEST_CONFIG["initial_cash"])
        cerebro.broker.setcommission(commission=BACKTEST_CONFIG["commission"])

        data_feed = bt.feeds.PandasData(
            dataname=df,
            datetime=None,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=-1,
        )
        cerebro.adddata(data_feed)
        cerebro.addstrategy(
            MLSurgeSingleStockStrategy,
            model_path=model_path,
            trading_mode=trading_mode,
        )

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.02)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

        initial_value = cerebro.broker.getvalue()
        results = cerebro.run()
        final_value = cerebro.broker.getvalue()

        strat = results[0]
        total_return = (final_value - initial_value) / initial_value * 100

        sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", 0.0) or 0.0
        max_dd = strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0.0) or 0.0
        total_trades = strat.long_trades + strat.short_trades
        win_rate = strat.wins / max(strat.total_closed, 1) * 100

        result = {
            "symbol": symbol,
            "trading_mode": trading_mode,
            "initial_cash": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "total_trades": total_trades,
            "long_trades": strat.long_trades,
            "short_trades": strat.short_trades,
            "win_rate": win_rate,
        }

        logging.info(
            f"[回测] {symbol} ({trading_mode}) | 收益率={total_return:.2f}% "
            f"Sharpe={sharpe:.2f} 最大回撤={max_dd:.2f}% 交易={total_trades}"
        )
        return result

    except Exception as e:
        handle_exception(e, f"run_single_stock_backtest({symbol})")
        return None


# ─── 多股票回测 ──────────────────────────────────────────────
def run_multiple_stocks_backtest(
    symbols: List[str],
    start_date: str,
    end_date: str = None,
    model_path: str = None,
    trading_mode: str = "long_only",
) -> List[Dict[str, Any]]:
    """批量回测多只股票"""
    results = []
    for sym in symbols:
        logging.info(f"[批量回测] 开始回测: {sym}")
        r = run_single_stock_backtest(sym, start_date, end_date, model_path, trading_mode)
        if r:
            results.append(r)
    return results


# ─── 多空回测 ────────────────────────────────────────────────
def run_long_short_backtest(
    symbol: str,
    start_date: str,
    end_date: str = None,
    trading_mode: str = "long_short",
    model_path: str = None,
) -> Optional[Dict[str, Any]]:
    """运行多空回测（trading_mode: long_only / short_only / long_short）"""
    return run_single_stock_backtest(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        model_path=model_path,
        trading_mode=trading_mode,
    )


# ─── 策略对比 ────────────────────────────────────────────────
def compare_trading_strategies(
    symbol: str,
    start_date: str,
    end_date: str = None,
    model_path: str = None,
) -> Dict[str, Any]:
    """对比三种交易模式，返回汇总对比字典"""
    modes = ["long_only", "short_only", "long_short"]
    comparison = {}

    for mode in modes:
        logging.info(f"[对比] 测试策略: {mode}")
        r = run_long_short_backtest(symbol, start_date, end_date, trading_mode=mode, model_path=model_path)
        comparison[mode] = r

    # 汇总统计
    valid = {m: v for m, v in comparison.items() if v is not None}
    if valid:
        best_mode = max(valid, key=lambda m: valid[m]["total_return"])
        logging.info(f"[对比] 最佳策略: {best_mode}，收益率={valid[best_mode]['total_return']:.2f}%")
        comparison["best_strategy"] = best_mode

    return comparison


__all__ = [
    "MLSurgeSingleStockStrategy",
    "run_single_stock_backtest",
    "run_multiple_stocks_backtest",
    "run_long_short_backtest",
    "compare_trading_strategies",
]
