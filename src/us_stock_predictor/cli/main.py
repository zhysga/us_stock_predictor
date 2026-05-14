# -*- coding: utf-8 -*-
"""
主程序入口 — 根据 RUN_MODE 环境变量分发执行逻辑
运行模式：full | train | backtest | demo
"""
import os
import logging
import sys
from datetime import datetime

from us_stock_predictor.config.core import (
    MODEL_CONFIG, TRAINING_CONFIG, MODEL_PATHS,
    FEATURE_COLUMNS, LOG_DIR, RECORD_DIR,
)
from us_stock_predictor.imbalance_handler import ImbalanceHandler
from us_stock_predictor.utils.core import (
    setup_environment, get_device, print_system_info,
    ensure_directories, handle_exception, check_model_exists,
    DualOutputContext,
)


def _setup_logging():
    """初始化日志：同时输出到控制台和文件"""
    ensure_directories(LOG_DIR, RECORD_DIR)
    today = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(LOG_DIR, f"main_log_{today}.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )
    return log_file


# ─── 训练流程 ─────────────────────────────────────────────────
def _run_train(demo: bool = False):
    """完整训练流程"""
    from us_stock_predictor.data.collector import USStockDataCollector
    from us_stock_predictor.datasets.timeseries import (
        StockDataset, split_data, create_data_loaders, analyze_dataset_balance,
    )
    from us_stock_predictor.models.transformer_surge import create_model
    from us_stock_predictor.training.trainer import train_model_with_imbalance_handling, ModelEvaluator

    device = get_device()
    logging.info(f"[训练] 设备: {device}")

    # 1. 数据收集
    collector = USStockDataCollector()
    if demo:
        collector.symbols = collector.symbols[:2]
        start_date, end_date = "20240101", "20241231"
        epochs = 3
    else:
        start_date = "20200101"
        end_date = datetime.now().strftime("%Y%m%d")
        epochs = TRAINING_CONFIG["epochs"]

    logging.info(f"[训练] 收集 {len(collector.symbols)} 只股票数据，{start_date}—{end_date}")
    data = collector.prepare_features(start_date, end_date)

    if data.empty:
        logging.error("[训练] 数据为空，终止训练")
        return None

    # 2. 数据切分
    train_df, val_df, test_df = split_data(data)

    # 3. 失衡处理（可选）
    handler = ImbalanceHandler() if ImbalanceHandler else None

    # 4. DataLoader
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df,
        batch_size=TRAINING_CONFIG["batch_size"],
        feature_cols=collector.feature_cols,
        imbalance_handler=handler,
    )

    # 5. 模型 & 训练
    model = create_model(input_dim=MODEL_CONFIG["input_dim"])
    logging.info(f"[训练] 模型参数：{sum(p.numel() for p in model.parameters()):,}")

    model, history = train_model_with_imbalance_handling(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        use_focal_loss=True,
        device=device,
        save_path=MODEL_PATHS["best_model"],
    )

    # 6. 评估
    evaluator = ModelEvaluator(model, device=device)
    metrics = evaluator.evaluate_model_comprehensive(test_loader)
    logging.info(f"[结果] 测试集 macro_f1={metrics.get('macro_f1', 0):.4f}")
    return model


# ─── 回测流程 ─────────────────────────────────────────────────
def _run_backtest(demo: bool = False):
    """单/多股票回测"""
    from us_stock_predictor.data.collector import USStockDataCollector
    from us_stock_predictor.backtest.engine import run_multiple_stocks_backtest

    collector = USStockDataCollector()
    if demo:
        symbols = collector.symbols[:2]
        start_date = "20240601"
    else:
        symbols = collector.symbols
        start_date = "20240101"

    if not check_model_exists(MODEL_PATHS["best_model"]):
        logging.warning("[回测] 模型文件不存在，跳过回测")
        return

    results = run_multiple_stocks_backtest(symbols, start_date)

    if results:
        returns = [r["total_return"] for r in results if r]
        avg_return = sum(returns) / len(returns) if returns else 0
        logging.info(f"[回测] 完成 {len(results)} 只股票，平均收益率={avg_return:.2f}%")
    else:
        logging.warning("[回测] 无回测结果")


# ─── 演示模式 ─────────────────────────────────────────────────
def _run_demo():
    """快速演示模式：2 只股票、3 epoch 训练 + 简单回测"""
    logging.info("=" * 60)
    logging.info("演示模式：快速验证系统可用性")
    logging.info("=" * 60)
    model = _run_train(demo=True)
    if model is not None:
        logging.info("[演示] 训练完成，运行演示回测...")
        _run_backtest(demo=True)
    logging.info("[演示] 演示模式完成")


# ─── 主入口 ───────────────────────────────────────────────────
def main_entry():
    """
    主程序入口，根据 RUN_MODE 环境变量分发：
      full     — 训练 + 回测
      train    — 仅训练
      backtest — 仅回测
      demo     — 快速演示
    """
    setup_environment()
    log_file = _setup_logging()
    print_system_info()

    mode = os.environ.get("RUN_MODE", "full").lower()
    logging.info(f"[主程序] 运行模式: {mode}")

    try:
        if mode == "train":
            _run_train(demo=False)
        elif mode == "backtest":
            _run_backtest(demo=False)
        elif mode == "demo":
            _run_demo()
        else:
            logging.info("[主程序] full 模式：训练 + 回测")
            model = _run_train(demo=False)
            if model is not None:
                _run_backtest(demo=False)
    except KeyboardInterrupt:
        logging.info("[主程序] 用户中断")
    except Exception as e:
        handle_exception(e, "main_entry", reraise=False)

    logging.info(f"[主程序] 完成。日志: {log_file}")


if __name__ == "__main__":
    main_entry()
