# -*- coding: utf-8 -*-
"""
数据收集器 — AkShare 美股数据拉取 + 35 维技术指标计算
"""
import os
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from us_stock_predictor.config.core import (
    STOCK_CONFIG_FILE, COLUMN_MAPPING, FEATURE_COLUMNS,
    TECHNICAL_INDICATORS_CONFIG, SIGNAL_WEIGHTS,
    PRICE_CHANGE_THRESHOLDS, MIN_HISTORY_LENGTH,
)
from us_stock_predictor.utils.core import handle_exception, clean_data

try:
    import akshare as ak
    _AKSHARE_AVAILABLE = True
except ImportError:
    _AKSHARE_AVAILABLE = False
    logging.warning("[警告] akshare 未安装，数据获取功能不可用")


class USStockDataCollector:
    """美股数据收集器 — 数据拉取 + 特征工程"""

    def __init__(self):
        self.feature_cols: list = FEATURE_COLUMNS.copy()
        self.symbols: list = []
        self._stock_info: dict = {}
        self._load_stock_list()

    # ── 股票列表加载 ─────────────────────────────────────────
    def _load_stock_list(self):
        """从 stock_config.xlsx 加载股票池"""
        try:
            if os.path.exists(STOCK_CONFIG_FILE):
                df = pd.read_excel(STOCK_CONFIG_FILE, dtype=str)
                code_col = df.columns[0]
                name_col = df.columns[1] if len(df.columns) > 1 else code_col
                for _, row in df.iterrows():
                    code = str(row[code_col]).strip()
                    name = str(row[name_col]).strip() if name_col != code_col else code
                    if code:
                        self.symbols.append(code)
                        self._stock_info[code] = name
                logging.info(f"[股票] 从配置文件读取到 {len(self.symbols)} 只股票")
            else:
                logging.warning(f"[警告] 股票配置文件不存在: {STOCK_CONFIG_FILE}，使用默认列表")
                self._default_stock_list()
        except Exception as e:
            handle_exception(e, "加载股票配置文件")
            self._default_stock_list()

    def _default_stock_list(self):
        defaults = {
            "105.AAPL": "苹果", "105.MSFT": "微软",
            "105.GOOGL": "谷歌", "105.AMZN": "亚马逊",
            "105.TSLA": "特斯拉", "105.NVDA": "英伟达",
            "105.META": "Meta", "105.NFLX": "奈飞",
        }
        self.symbols = list(defaults.keys())
        self._stock_info = defaults

    def get_stock_list_with_names(self) -> dict:
        """返回 {symbol: name} 字典"""
        return self._stock_info.copy()

    def get_feature_columns(self) -> list:
        return self.feature_cols.copy()

    # ── 数据获取 ─────────────────────────────────────────────
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """拉取单只股票日线数据，返回标准化 DataFrame"""
        if not _AKSHARE_AVAILABLE:
            logging.warning("[警告] akshare 不可用，返回空 DataFrame")
            return pd.DataFrame()
        try:
            df = ak.stock_us_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
            if df is None or df.empty:
                return pd.DataFrame()

            df.rename(columns=COLUMN_MAPPING, inplace=True)

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df.dropna(subset=["open", "high", "low", "close"], inplace=True)
            df["symbol"] = symbol
            return df

        except Exception as e:
            handle_exception(e, f"获取 {symbol} 数据")
            return pd.DataFrame()

    # ── 技术指标计算 ─────────────────────────────────────────
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 35 维技术特征，返回扩充后的 DataFrame"""
        if df.empty or len(df) < MIN_HISTORY_LENGTH:
            return df
        try:
            df = df.copy()
            cfg = TECHNICAL_INDICATORS_CONFIG

            # === 基础价格派生（7）===
            df["high_low_ratio"] = (df["high"] / df["low"].replace(0, np.nan)).fillna(1.0)
            df["close_open_ratio"] = (df["close"] / df["open"].replace(0, np.nan)).fillna(1.0)
            df["price_range"] = df["high"] - df["low"]

            # === 成交量（6）===
            df["dollar_volume"] = df["close"] * df["volume"]
            vol_sma = df["volume"].rolling(cfg["volume_sma_period"], min_periods=1).mean()
            df["volume_sma_20"] = vol_sma
            df["volume_ratio"] = (df["volume"] / vol_sma.replace(0, np.nan)).fillna(1.0)
            df["volume_momentum"] = df["volume"].pct_change(5).fillna(0.0)
            if "turnover_rate" not in df.columns:
                df["turnover_rate"] = df["volume_ratio"] * 0.01

            # === 移动均线 & VWAP（4）===
            df["sma_20"] = df["close"].rolling(cfg["sma_period"], min_periods=1).mean()
            df["range_ma"] = df["price_range"].rolling(cfg["sma_period"], min_periods=1).mean()
            typical = (df["high"] + df["low"] + df["close"]) / 3
            df["vwap"] = (typical * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, np.nan)
            df["price_vwap_ratio"] = (df["close"] / df["vwap"].replace(0, np.nan)).fillna(1.0)

            # === RSI（2）===
            df["rsi"] = self._calc_rsi(df["close"], cfg["rsi_period"])
            df["rsi_prev_5"] = df["rsi"].shift(5).fillna(50.0)

            # === MFI（3）===
            df["mfi"] = self._calc_mfi(df, cfg["mfi_period"])
            df["mfi_prev"] = df["mfi"].shift(1).fillna(50.0)
            df["mfi_signal"] = (df["mfi"] - df["mfi_prev"]).clip(-10, 10) / 10

            # === K 线形态（2）===
            body = (df["close"] - df["open"]).abs()
            df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"].replace(0, np.nan)
            df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"].replace(0, np.nan)
            df["upper_shadow"] = df["upper_shadow"].fillna(0.0).clip(0, 1)
            df["lower_shadow"] = df["lower_shadow"].fillna(0.0).clip(0, 1)

            # === 动量（3）===
            df["intraday_momentum"] = ((df["close"] - df["open"]) / df["open"].replace(0, np.nan)).fillna(0.0)
            df["price_momentum"] = df["close"].pct_change(cfg["momentum_period"]).fillna(0.0)
            df["momentum_score"] = (
                df["rsi"] / 100 * SIGNAL_WEIGHTS["momentum_score"] +
                df["mfi"] / 100 * SIGNAL_WEIGHTS["mfi_signal"] +
                df["price_momentum"].clip(-0.1, 0.1) * 5 * SIGNAL_WEIGHTS["volatility_signal"]
            ).fillna(0.0)

            # === 复合信号（8）===
            df["price_breakout"] = (
                (df["close"] > df["sma_20"]) & (df["close"] > df["close"].shift(1))
            ).astype(float)
            df["volatility_signal"] = (df["price_range"] / df["range_ma"].replace(0, np.nan)).fillna(1.0).clip(0, 3)
            df["volume_surge"] = (df["volume_ratio"] > 2.0).astype(float)
            df["surge_signal"] = (
                df["volume_surge"] * SIGNAL_WEIGHTS["volume_surge"] +
                df["price_breakout"] * SIGNAL_WEIGHTS["price_breakout"] +
                df["momentum_score"] * SIGNAL_WEIGHTS["momentum_score"]
            ).fillna(0.0)

            df["price_to_sma_ratio"] = (df["close"] / df["sma_20"].replace(0, np.nan)).fillna(1.0)
            df["volume_price_trend"] = (df["volume_momentum"] * df["intraday_momentum"]).fillna(0.0)
            df["rsi_momentum"] = df["rsi"].diff(3).fillna(0.0)
            df["composite_score"] = (
                df["surge_signal"] * 0.4 +
                df["momentum_score"] * 0.3 +
                (df["rsi"] / 100) * 0.15 +
                (df["mfi"] / 100) * 0.15
            ).fillna(0.0)

            df = clean_data(df)
            return df

        except Exception as e:
            handle_exception(e, "计算技术指标")
            return df

    # ── RSI 计算 ─────────────────────────────────────────────
    @staticmethod
    def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        return rsi.fillna(50.0)

    # ── MFI 计算 ─────────────────────────────────────────────
    @staticmethod
    def _calc_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        raw_mf = typical * df["volume"]
        pos_mf = raw_mf.where(typical > typical.shift(1), 0.0).rolling(period, min_periods=1).sum()
        neg_mf = raw_mf.where(typical < typical.shift(1), 0.0).rolling(period, min_periods=1).sum()
        mfi = 100 - 100 / (1 + pos_mf / neg_mf.replace(0, np.nan))
        return mfi.fillna(50.0)

    # ── 四分类标签生成 ────────────────────────────────────────
    def _generate_four_class_labels(self, price_change: pd.Series) -> np.ndarray:
        """将涨跌幅映射为四分类标签"""
        big_drop = PRICE_CHANGE_THRESHOLDS["big_drop"]
        big_surge = PRICE_CHANGE_THRESHOLDS["big_surge"]
        labels = np.where(
            price_change <= big_drop, 0,
            np.where(price_change < 0, 1,
                     np.where(price_change < big_surge, 2, 3))
        )
        return labels.astype(int)

    # ── 全量数据准备（训练用）────────────────────────────────
    def prepare_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        批量拉取所有股票数据、计算特征、生成标签，返回合并 DataFrame。
        标签列：price_change_class（0-3）、price_change（涨跌幅）
        """
        all_dfs = []
        total = len(self.symbols)
        for i, symbol in enumerate(self.symbols, 1):
            logging.info(f"[{i}/{total}] 收集 {symbol} 数据...")
            df = self.get_stock_data(symbol, start_date, end_date)
            if df.empty:
                logging.warning(f"  跳过 {symbol}（无数据）")
                continue

            df = self.calculate_technical_indicators(df)
            if df.empty:
                continue

            df["price_change"] = df["close"].pct_change().fillna(0.0)
            df["price_change_class"] = self._generate_four_class_labels(df["price_change"])

            missing = [c for c in self.feature_cols if c not in df.columns]
            for c in missing:
                df[c] = 0.0

            all_dfs.append(df)

        if not all_dfs:
            logging.warning("[警告] 所有股票数据获取失败，返回空 DataFrame")
            return pd.DataFrame()

        result = pd.concat(all_dfs, ignore_index=False)
        result.sort_index(inplace=True)
        result.dropna(subset=self.feature_cols[:4], inplace=True)
        logging.info(f"[完成] 数据收集完毕，共 {len(result)} 条记录，{len(all_dfs)} 只股票")
        return result


__all__ = ["USStockDataCollector"]
