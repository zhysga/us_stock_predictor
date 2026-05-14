# -*- coding: utf-8 -*-
"""
训练器模块 — ModelTrainer、ModelEvaluator、train_model_with_imbalance_handling
"""
import logging
import copy
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from us_stock_predictor.config.core import TRAINING_CONFIG, MODEL_PATHS
from us_stock_predictor.models.transformer_surge import FocalLoss, create_focal_loss, save_model
from us_stock_predictor.utils.core import handle_exception, get_device, calculate_f1_score


# ─── ModelTrainer ────────────────────────────────────────────
class ModelTrainer:
    """
    封装训练循环，支持早停、学习率调度、模型检查点。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device=None,
        patience: int = None,
        scheduler=None,
        checkpoint_path: str = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device or get_device()
        self.patience = patience or TRAINING_CONFIG["patience"]
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path or MODEL_PATHS["checkpoint"]

        self.best_val_f1: float = 0.0
        self.best_state: dict = None
        self.no_improve: int = 0
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [], "train_acc": [], "val_f1": []
        }

    # ── 单 epoch 训练 ─────────────────────────────────────────
    def train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """返回 (avg_loss, accuracy)"""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in loader:
            x, labels, targets = batch
            x = x.to(self.device)
            labels = labels.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            class_logits, magnitude = self.model(x)
            cls_loss = self.loss_fn(class_logits, labels)
            reg_loss = nn.functional.mse_loss(magnitude.squeeze(1), targets)
            loss = cls_loss + 0.1 * reg_loss
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), TRAINING_CONFIG.get("grad_clip", 1.0))
            self.optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = class_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        return total_loss / max(total, 1), correct / max(total, 1)

    # ── 单 epoch 验证 ─────────────────────────────────────────
    def validate_epoch(self, loader: DataLoader) -> Tuple[float, float, float]:
        """返回 (avg_loss, accuracy, macro_f1)"""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                x, labels, targets = batch
                x = x.to(self.device)
                labels = labels.to(self.device)
                targets = targets.to(self.device)

                class_logits, magnitude = self.model(x)
                cls_loss = self.loss_fn(class_logits, labels)
                reg_loss = nn.functional.mse_loss(magnitude.squeeze(1), targets)
                loss = cls_loss + 0.1 * reg_loss

                total_loss += loss.item() * len(labels)
                preds = class_logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        f1 = calculate_f1_score(all_labels, all_preds, average="macro")
        return total_loss / max(total, 1), correct / max(total, 1), f1

    # ── 完整训练 ──────────────────────────────────────────────
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = None,
    ) -> Dict[str, list]:
        epochs = epochs or TRAINING_CONFIG["epochs"]
        self.model.to(self.device)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_f1 = self.validate_epoch(val_loader)

            if self.scheduler:
                self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_f1"].append(val_f1)

            logging.info(
                f"[Epoch {epoch:3d}/{epochs}] "
                f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} f1={val_f1:.4f}"
            )

            # 早停 & 模型保存
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.no_improve = 0
                save_model(self.model, self.checkpoint_path,
                           {"epoch": epoch, "val_f1": val_f1})
                logging.info(f"  ✓ 最佳模型更新 val_f1={val_f1:.4f}")
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    logging.info(f"  早停：{self.patience} epoch 无改善")
                    break

        # 恢复最佳权重
        if self.best_state:
            self.model.load_state_dict(self.best_state)
        return self.history


# ─── ModelEvaluator ──────────────────────────────────────────
class ModelEvaluator:
    """全面评估模型性能"""

    def __init__(self, model: nn.Module, device=None):
        self.model = model
        self.device = device or get_device()

    def evaluate_model_comprehensive(self, loader: DataLoader) -> Dict[str, Any]:
        """
        返回：macro_f1, per_class_f1, confusion_matrix, big_surge_recall/precision,
               accuracy, auc（如可用）
        """
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for batch in loader:
                x, labels, _ = batch
                x = x.to(self.device)
                labels = labels.to(self.device)
                class_logits, _ = self.model(x)
                probs = torch.softmax(class_logits, dim=1)
                preds = class_logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        metrics = {}

        try:
            from sklearn.metrics import (
                classification_report, confusion_matrix,
                f1_score, roc_auc_score, accuracy_score,
            )
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
            metrics["per_class_f1"] = {i: float(f) for i, f in enumerate(per_f1)}

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
            metrics["confusion_matrix"] = cm.tolist()

            # 大涨类（class=3）精准率和召回率
            cls3_idx = (y_true == 3)
            if cls3_idx.sum() > 0:
                tp = ((y_pred == 3) & (y_true == 3)).sum()
                pp = (y_pred == 3).sum()
                metrics["big_surge_recall"] = float(tp / max(cls3_idx.sum(), 1))
                metrics["big_surge_precision"] = float(tp / max(pp, 1))
            else:
                metrics["big_surge_recall"] = 0.0
                metrics["big_surge_precision"] = 0.0

            try:
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
                metrics["auc"] = float(roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro"))
            except Exception:
                metrics["auc"] = 0.0

        except ImportError:
            metrics["macro_f1"] = calculate_f1_score(y_true, y_pred)

        logging.info(
            f"[评估] accuracy={metrics.get('accuracy', 0):.4f}  "
            f"macro_f1={metrics.get('macro_f1', 0):.4f}  "
            f"大涨recall={metrics.get('big_surge_recall', 0):.4f}"
        )
        return metrics


# ─── 便捷训练函数 ────────────────────────────────────────────
def train_model_with_imbalance_handling(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = None,
    use_focal_loss: bool = True,
    device=None,
    class_weights: dict = None,
    save_path: str = None,
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    便捷训练入口：自动创建 optimizer、loss_fn、scheduler，调用 ModelTrainer.fit。
    返回 (trained_model, history)
    """
    device = device or get_device()
    epochs = epochs or TRAINING_CONFIG["epochs"]
    save_path = save_path or MODEL_PATHS["best_model"]

    loss_fn = create_focal_loss(class_weights=class_weights) if use_focal_loss else nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5,
    )

    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        patience=TRAINING_CONFIG["patience"],
        scheduler=scheduler,
        checkpoint_path=save_path,
    )

    logging.info(f"[训练] 开始训练，epochs={epochs}, focal_loss={use_focal_loss}, device={device}")
    history = trainer.fit(train_loader, val_loader, epochs=epochs)

    # 训练完成后保存最终最佳模型
    save_model(model, save_path, {"val_f1": trainer.best_val_f1})
    logging.info(f"[训练] 完成，最佳 val_f1={trainer.best_val_f1:.4f}")
    return model, history


__all__ = [
    "ModelTrainer", "ModelEvaluator",
    "train_model_with_imbalance_handling",
]
