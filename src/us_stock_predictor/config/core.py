# -*- coding: utf-8 -*-
"""
配置中心 — 所有模块共享的常量与参数
"""
import os

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

# ─── 文件路径 ───────────────────────────────────────────────
STOCK_CONFIG_FILE = os.path.join(ROOT_DIR, "stock_config.xlsx")
MODEL_PATHS = {
    "best_model": os.path.join(ROOT_DIR, "best_surge_model.pth"),
    "checkpoint": os.path.join(ROOT_DIR, "checkpoint_model.pth"),
}
LOG_DIR = os.path.join(ROOT_DIR, "logs")
RECORD_DIR = os.path.join(ROOT_DIR, "record_excel")

# ─── 模型配置 ────────────────────────────────────────────────
MODEL_CONFIG = {
    "input_dim": 35,
    "hidden_dim": 64,
    "num_heads": 4,
    "num_layers": 3,
    "dropout": 0.2,
    "seq_len": 20,
    "num_classes": 4,
}

# ─── 训练配置 ────────────────────────────────────────────────
TRAINING_CONFIG = {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "patience": 10,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "random_seed": 42,
}

# ─── Focal Loss 配置 ─────────────────────────────────────────
FOCAL_LOSS_CONFIG = {
    "alpha": 1.0,
    "gamma": 2.0,
    "use_focal_loss": True,
}

# ─── 价格变动阈值（四分类边界）────────────────────────────────
PRICE_CHANGE_THRESHOLDS = {
    "big_drop": -0.05,    # 跌幅 ≤ -5% → 大跌(0)
    "big_surge": 0.05,    # 涨幅 ≥  5% → 大涨(3)
}
PRICE_CHANGE_LABELS = {
    0: "大跌",
    1: "小跌",
    2: "小涨",
    3: "大涨",
}

# ─── 回测配置 ────────────────────────────────────────────────
BACKTEST_CONFIG = {
    "initial_cash": 100_000.0,
    "threshold": 0.5,
    "profit_target": 0.05,
    "stop_loss": -0.03,
    "hold_days": 5,
    "enable_short_selling": True,
    "commission": 0.001,
    "stake_pct": 0.95,
    # 四分类做多策略权重
    "four_class_strategy": {
        "big_surge_weight": 0.6,
        "small_surge_weight": 0.4,
        "big_drop_weight": 0.6,
        "small_drop_weight": 0.4,
        "signal_threshold": 0.1,
        "min_confidence": 0.4,
    },
    # 做空策略权重
    "short_strategy": {
        "big_drop_weight": 0.6,
        "small_drop_weight": 0.4,
        "big_surge_weight": 0.6,
        "small_surge_weight": 0.4,
        "signal_threshold": 0.1,
        "min_confidence": 0.4,
    },
}

# ─── 技术指标配置 ────────────────────────────────────────────
TECHNICAL_INDICATORS_CONFIG = {
    "rsi_period": 14,
    "mfi_period": 14,
    "sma_period": 20,
    "volume_sma_period": 20,
    "momentum_period": 5,
}

SIGNAL_WEIGHTS = {
    "volume_surge": 0.35,
    "momentum_score": 0.25,
    "volatility_signal": 0.15,
    "price_breakout": 0.10,
    "mfi_signal": 0.15,
}

# ─── 35 个特征列名（顺序与模型一致）────────────────────────────
FEATURE_COLUMNS = [
    # 基础价格（7）
    "open", "high", "low", "close",
    "high_low_ratio", "close_open_ratio", "price_range",
    # 成交量（6）
    "volume", "dollar_volume", "volume_ratio",
    "volume_sma_20", "volume_momentum", "turnover_rate",
    # 技术指标（14）
    "sma_20", "range_ma", "vwap", "price_vwap_ratio",
    "rsi", "rsi_prev_5",
    "mfi", "mfi_prev", "mfi_signal",
    "upper_shadow", "lower_shadow",
    "intraday_momentum", "price_momentum", "momentum_score",
    # 复合信号（8）
    "price_breakout", "volatility_signal", "volume_surge", "surge_signal",
    "price_to_sma_ratio", "volume_price_trend", "rsi_momentum", "composite_score",
]

# ─── 样本失衡配置 ────────────────────────────────────────────
TARGET_RATIO = 0.2
IMBALANCE_METHODS = ["smote", "adasyn", "smote_tomek", "undersample"]
IMBALANCE_CONFIG = {
    "default_method": "smote_tomek",
    "target_ratio": TARGET_RATIO,
    "auto_select": True,
}

# ─── AkShare 列名映射 ─────────────────────────────────────────
COLUMN_MAPPING = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "price_change",
    "换手率": "turnover_rate",
}

# ─── 最短历史数据要求 ────────────────────────────────────────
MIN_HISTORY_LENGTH = 20

# ─── 系统配置 ────────────────────────────────────────────────
SYSTEM_CONFIG = {
    "random_seed": 42,
    "num_workers": 0,
    "pin_memory": False,
    "verbose": True,
}

LOG_CONFIG = {
    "log_dir": LOG_DIR,
    "log_level": "INFO",
    "console_output": True,
    "file_output": True,
}

__all__ = [
    "ROOT_DIR", "STOCK_CONFIG_FILE", "MODEL_PATHS", "LOG_DIR", "RECORD_DIR",
    "MODEL_CONFIG", "TRAINING_CONFIG", "FOCAL_LOSS_CONFIG",
    "PRICE_CHANGE_THRESHOLDS", "PRICE_CHANGE_LABELS",
    "BACKTEST_CONFIG", "TECHNICAL_INDICATORS_CONFIG", "SIGNAL_WEIGHTS",
    "FEATURE_COLUMNS", "TARGET_RATIO", "IMBALANCE_METHODS", "IMBALANCE_CONFIG",
    "COLUMN_MAPPING", "MIN_HISTORY_LENGTH", "SYSTEM_CONFIG", "LOG_CONFIG",
]
