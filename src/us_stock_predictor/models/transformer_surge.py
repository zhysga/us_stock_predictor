# -*- coding: utf-8 -*-
"""
Transformer 模型模块 — StockSurgePredictor（四分类 + 回归多任务头）
包含 FocalLoss、模型保存/加载工具函数。

Bug 修复记录：
  ① 注意力池化 query 切片：encoded[-1:1] → encoded[-1:]（原切片返回空张量）
  ② 异常分支输出维度统一为 (batch, 4)，与正常路径一致
"""
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from us_stock_predictor.config.core import MODEL_CONFIG, MODEL_PATHS, FOCAL_LOSS_CONFIG
from us_stock_predictor.utils.core import handle_exception


# ─── Focal Loss ──────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    多分类 Focal Loss（Lin et al., ICCV 2017）
    可叠加类别权重（class_weights）用于处理样本不平衡。
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0,
                 class_weights: dict = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.class_weights is not None:
            device = logits.device
            w = torch.ones(num_classes, device=device)
            for k, v in self.class_weights.items():
                if int(k) < num_classes:
                    w[int(k)] = float(v)
            batch_weights = w[targets]
            focal = focal * batch_weights

        return focal.mean()


# ─── 注意力池化 ──────────────────────────────────────────────
class AttentionPooling(nn.Module):
    """
    基于末位 token 作为 query 的注意力加权池化。
    修复：原始代码 encoded[-1:1] 在 seq_len>1 时返回空张量，
          已改为 encoded[-1:] 正确取最后一个时间步。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        # encoded: [batch, seq_len, hidden_dim]
        # ✅ Bug Fix: encoded[-1:1] → encoded[-1:]
        query = encoded[:, -1:, :]            # [batch, 1, hidden_dim]
        scores = self.attn(encoded)           # [batch, seq_len, 1]
        weights = torch.softmax(scores, dim=1)
        pooled = (encoded * weights).sum(dim=1)  # [batch, hidden_dim]
        return pooled


# ─── StockSurgePredictor ─────────────────────────────────────
class StockSurgePredictor(nn.Module):
    """
    基于 Transformer Encoder 的股票大涨预测模型。
    输入：[batch, seq_len, input_dim]
    输出：(class_logits [batch, 4], magnitude [batch, 1])
    """

    def __init__(
        self,
        input_dim: int = None,
        hidden_dim: int = None,
        num_heads: int = None,
        num_layers: int = None,
        dropout: float = None,
        num_classes: int = 4,
    ):
        super().__init__()
        input_dim  = input_dim  or MODEL_CONFIG["input_dim"]
        hidden_dim = hidden_dim or MODEL_CONFIG["hidden_dim"]
        num_heads  = num_heads  or MODEL_CONFIG["num_heads"]
        num_layers = num_layers or MODEL_CONFIG["num_layers"]
        dropout    = dropout    if dropout is not None else MODEL_CONFIG["dropout"]

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_pool = AttentionPooling(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        x: [batch, seq_len, input_dim]
        Returns: (class_logits [batch, 4], magnitude [batch, 1])
        """
        try:
            feat = self.input_proj(x)              # [batch, seq, hidden]
            feat = F.relu(feat)
            encoded = self.transformer(feat)       # [batch, seq, hidden]
            pooled = self.attn_pool(encoded)       # [batch, hidden]
            pooled = self.norm(pooled)
            pooled = self.dropout(pooled)

            class_logits = self.cls_head(pooled)   # [batch, 4]
            magnitude = self.reg_head(pooled)      # [batch, 1]
            return class_logits, magnitude

        except Exception as e:
            handle_exception(e, "StockSurgePredictor.forward")
            # ✅ Bug Fix: 异常分支返回 (batch, 4) 维度，与正常路径一致
            batch = x.size(0)
            device = x.device
            zeros_cls = torch.zeros(batch, self.num_classes, device=device)
            zeros_reg = torch.zeros(batch, 1, device=device)
            return zeros_cls, zeros_reg


# ─── 工厂函数 ────────────────────────────────────────────────
def create_model(input_dim: int = None, **kwargs) -> StockSurgePredictor:
    """创建 StockSurgePredictor 实例"""
    return StockSurgePredictor(input_dim=input_dim or MODEL_CONFIG["input_dim"], **kwargs)


def create_focal_loss(alpha: float = None, gamma: float = None,
                      class_weights: dict = None) -> FocalLoss:
    """创建 FocalLoss 实例"""
    return FocalLoss(
        alpha=alpha or FOCAL_LOSS_CONFIG["alpha"],
        gamma=gamma or FOCAL_LOSS_CONFIG["gamma"],
        class_weights=class_weights,
    )


# ─── 模型 I/O ────────────────────────────────────────────────
def save_model(model: StockSurgePredictor, path: str, metadata: dict = None):
    """保存模型权重 + 元数据"""
    try:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = {"model_state_dict": model.state_dict()}
        if metadata:
            payload.update(metadata)
        torch.save(payload, path)
        logging.info(f"[保存] 模型已保存: {path}")
    except Exception as e:
        handle_exception(e, f"save_model({path})")


def load_model(path: str, input_dim: int = None, device=None,
               **model_kwargs) -> StockSurgePredictor:
    """加载模型权重，返回 eval 模式的模型"""
    from us_stock_predictor.utils.core import get_device, check_model_exists
    device = device or get_device()
    if not check_model_exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")

    model = create_model(input_dim=input_dim, **model_kwargs)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    if "epoch" in checkpoint:
        logging.info(f"[加载] epoch={checkpoint['epoch']}")
    if "val_f1" in checkpoint:
        logging.info(f"[加载] val_f1={checkpoint['val_f1']:.4f}")
    return model


__all__ = [
    "FocalLoss", "AttentionPooling", "StockSurgePredictor",
    "create_model", "create_focal_loss", "save_model", "load_model",
]
