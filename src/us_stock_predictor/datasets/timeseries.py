# -*- coding: utf-8 -*-
"""
数据集模块 — PyTorch Dataset 封装、数据切分、DataLoader 创建
"""
import logging
from collections import Counter
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from us_stock_predictor.config.core import (
    MODEL_CONFIG, TRAINING_CONFIG, FEATURE_COLUMNS,
)
from us_stock_predictor.utils.core import handle_exception, clean_data


class StockDataset(Dataset):
    """
    时序股票数据集。
    每条样本：(features [seq_len, input_dim], label [int], target [float])
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: list = None,
        seq_len: int = None,
        target_col: str = "price_change_class",
        regression_col: str = "price_change",
    ):
        self.seq_len = seq_len or MODEL_CONFIG["seq_len"]
        self.feature_cols = feature_cols or FEATURE_COLUMNS
        self.target_col = target_col
        self.regression_col = regression_col

        data = clean_data(data.copy())

        # 确保所有特征列存在
        for col in self.feature_cols:
            if col not in data.columns:
                data[col] = 0.0

        self.features = data[self.feature_cols].values.astype(np.float32)

        if target_col in data.columns:
            self.labels = data[target_col].values.astype(np.int64)
        else:
            self.labels = np.zeros(len(data), dtype=np.int64)

        if regression_col in data.columns:
            self.targets = data[regression_col].values.astype(np.float32)
        else:
            self.targets = np.zeros(len(data), dtype=np.float32)

        self.sequences, self.seq_labels, self.seq_targets = self._prepare_sequences()

    def _prepare_sequences(self):
        """滑动窗口构建 [seq_len, feature_dim] 序列样本"""
        seqs, labs, tgts = [], [], []
        n = len(self.features)
        for i in range(self.seq_len, n):
            window = self.features[i - self.seq_len: i]  # [seq_len, feat]
            seqs.append(window)
            labs.append(self.labels[i])
            tgts.append(self.targets[i])

        if not seqs:
            logging.warning("[Dataset] 数据量不足，无法构建序列样本")
            return np.array([]).reshape(0, self.seq_len, len(self.feature_cols)), \
                   np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        return (
            np.array(seqs, dtype=np.float32),
            np.array(labs, dtype=np.int64),
            np.array(tgts, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.seq_labels[idx], dtype=torch.long),
            torch.tensor(self.seq_targets[idx], dtype=torch.float32),
        )


# ─── 数据切分 ─────────────────────────────────────────────────
def split_data(
    data: pd.DataFrame,
    train_ratio: float = None,
    val_ratio: float = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按时间顺序切分数据（不随机打乱，避免时序泄漏）。
    返回 (train_df, val_df, test_df)
    """
    train_ratio = train_ratio or TRAINING_CONFIG["train_ratio"]
    val_ratio = val_ratio or TRAINING_CONFIG["val_ratio"]
    test_ratio = 1.0 - train_ratio - val_ratio

    data = data.sort_index()
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_df = data.iloc[:n_train]
    val_df = data.iloc[n_train: n_train + n_val]
    test_df = data.iloc[n_train + n_val:]

    logging.info(
        f"[切分] 训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}"
    )
    return train_df, val_df, test_df


# ─── DataLoader 创建 ─────────────────────────────────────────
def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = None,
    feature_cols: list = None,
    seq_len: int = None,
    imbalance_handler=None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    根据切分后的 DataFrame 创建 train/val/test DataLoader。
    可选传入 imbalance_handler 以创建 WeightedRandomSampler。
    """
    batch_size = batch_size or TRAINING_CONFIG["batch_size"]

    train_ds = StockDataset(train_df, feature_cols=feature_cols, seq_len=seq_len)
    val_ds = StockDataset(val_df, feature_cols=feature_cols, seq_len=seq_len)
    test_ds = StockDataset(test_df, feature_cols=feature_cols, seq_len=seq_len)

    sampler = None
    if imbalance_handler is not None and len(train_ds) > 0:
        try:
            sampler = imbalance_handler.create_weighted_sampler(train_ds.seq_labels)
        except Exception as e:
            handle_exception(e, "创建 WeightedRandomSampler")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logging.info(
        f"[DataLoader] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)} 样本"
    )
    return train_loader, val_loader, test_loader


# ─── 数据集分析 ──────────────────────────────────────────────
def analyze_dataset_balance(dataset: StockDataset) -> dict:
    """分析数据集的类别分布"""
    if len(dataset) == 0:
        return {}
    counter = Counter(dataset.seq_labels.tolist())
    total = sum(counter.values())
    info = {}
    class_names = ["大跌", "小跌", "小涨", "大涨"]
    for cls_id in sorted(counter.keys()):
        cnt = counter[cls_id]
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        pct = cnt / total * 100
        info[cls_id] = {"name": name, "count": cnt, "pct": pct}
        logging.info(f"  类别 {cls_id}({name}): {cnt} ({pct:.1f}%)")
    return info


__all__ = [
    "StockDataset", "split_data",
    "create_data_loaders", "analyze_dataset_balance",
]
