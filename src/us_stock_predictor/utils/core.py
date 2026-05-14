# -*- coding: utf-8 -*-
"""
工具函数模块 — 设备检测、编码处理、日志、错误处理等通用工具
"""
import os
import sys
import io
import gc
import time
import logging
import traceback
import contextlib
from typing import Callable, Optional, Any

import numpy as np

# ─── 设备检测 ────────────────────────────────────────────────
def get_device():
    """返回可用的计算设备（优先 CUDA，其次 MPS，最后 CPU）"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        return device
    except ImportError:
        return None


# ─── 环境初始化 ──────────────────────────────────────────────
def setup_environment(seed: int = 42):
    """设置随机种子、编码、日志等全局环境"""
    os.environ["PYTHONIOENCODING"] = "utf-8"

    try:
        import random
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    ensure_directories("logs", "record_excel", "models")


# ─── 目录管理 ────────────────────────────────────────────────
def ensure_directories(*dirs):
    """批量创建目录（如不存在）"""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ─── 模型文件检查 ────────────────────────────────────────────
def check_model_exists(path: str) -> bool:
    """检查模型文件是否存在，存在则打印大小"""
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1024 / 1024
        logging.info(f"[模型] 找到模型文件: {path} ({size_mb:.2f} MB)")
        return True
    logging.warning(f"[模型] 模型文件不存在: {path}")
    return False


# ─── 异常处理 ────────────────────────────────────────────────
def handle_exception(e: Exception, context: str = "", reraise: bool = False):
    """统一异常处理：打印上下文、堆栈；可选是否重新抛出"""
    msg = f"[错误] {context}: {type(e).__name__}: {e}"
    logging.error(msg)
    logging.debug(traceback.format_exc())
    if reraise:
        raise


# ─── 系统信息 ────────────────────────────────────────────────
def print_system_info():
    """打印 Python、PyTorch、设备等系统信息"""
    logging.info("=" * 60)
    logging.info(f"Python: {sys.version}")
    try:
        import torch
        logging.info(f"PyTorch: {torch.__version__}")
        logging.info(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logging.warning("PyTorch 未安装")
    device = get_device()
    logging.info(f"使用设备: {device}")
    logging.info("=" * 60)


# ─── 数据质量检查 ────────────────────────────────────────────
def validate_data_quality(df, name: str = "DataFrame") -> bool:
    """
    检查 DataFrame 的基本质量：空值比例、无穷大、行数。
    返回 True 表示质量合格。
    """
    import pandas as pd

    if df is None or df.empty:
        logging.warning(f"[质量] {name}: 数据为空")
        return False

    nan_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    if nan_ratio > 0.3:
        logging.warning(f"[质量] {name}: NaN 比例过高 ({nan_ratio:.2%})")

    numeric_cols = df.select_dtypes(include=[np.number])
    inf_count = np.isinf(numeric_cols.values).sum()
    if inf_count > 0:
        logging.warning(f"[质量] {name}: 发现 {inf_count} 个 Inf 值")

    logging.info(f"[质量] {name}: shape={df.shape}, NaN={nan_ratio:.2%}")
    return True


def clean_data(df):
    """清理 DataFrame：填充 NaN、替换 Inf"""
    import pandas as pd

    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].ffill().fillna(0.0)
    return df


# ─── 指标计算 ────────────────────────────────────────────────
def calculate_f1_score(y_true, y_pred, average: str = "macro") -> float:
    """计算 F1 分数（依赖 sklearn）"""
    try:
        from sklearn.metrics import f1_score
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))
    except Exception as e:
        handle_exception(e, "calculate_f1_score")
        return 0.0


# ─── 内存监控 ────────────────────────────────────────────────
def memory_usage_check(threshold_mb: float = 4096.0):
    """检查当前进程内存使用，超阈值则警告并触发 GC"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        if mem_mb > threshold_mb:
            logging.warning(f"[内存] 使用量较高: {mem_mb:.1f} MB，触发 GC")
            gc.collect()
        return mem_mb
    except ImportError:
        return 0.0


# ─── 进度跟踪 ────────────────────────────────────────────────
class ProgressTracker:
    """轻量进度跟踪器，支持 ETA 估算"""

    def __init__(self, total: int, desc: str = "进度"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        self.current += n
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            logging.info(
                f"[{self.desc}] {self.current}/{self.total} "
                f"({self.current / self.total * 100:.1f}%) "
                f"ETA: {eta:.0f}s"
            )

    def done(self):
        elapsed = time.time() - self.start_time
        logging.info(f"[{self.desc}] 完成，耗时 {elapsed:.1f}s")


# ─── 双输出上下文 ─────────────────────────────────────────────
class DualOutputContext:
    """将标准输出同时写入控制台和文件"""

    def __init__(self, filepath: str, encoding: str = "utf-8"):
        self.filepath = filepath
        self.encoding = encoding
        self._orig_stdout = None
        self._file = None
        self._tee = None

    def __enter__(self):
        ensure_directories(os.path.dirname(self.filepath) or ".")
        self._orig_stdout = sys.stdout
        self._file = open(self.filepath, "a", encoding=self.encoding)

        class Tee:
            def __init__(self, *streams):
                self.streams = streams

            def write(self, data):
                for s in self.streams:
                    try:
                        s.write(data)
                    except Exception:
                        pass

            def flush(self):
                for s in self.streams:
                    try:
                        s.flush()
                    except Exception:
                        pass

        self._tee = Tee(self._orig_stdout, self._file)
        sys.stdout = self._tee
        return self

    def __exit__(self, *args):
        sys.stdout = self._orig_stdout
        if self._file:
            self._file.close()


# ─── UTF-8 编码 wrapper ──────────────────────────────────────
def main_with_proper_encoding(fn: Callable, *args, **kwargs) -> Any:
    """以 UTF-8 编码环境调用函数，解决 Windows/Linux 编码差异"""
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )
    os.environ["PYTHONIOENCODING"] = "utf-8"
    return fn(*args, **kwargs)


__all__ = [
    "get_device", "setup_environment", "ensure_directories",
    "check_model_exists", "handle_exception", "print_system_info",
    "validate_data_quality", "clean_data", "calculate_f1_score",
    "memory_usage_check", "ProgressTracker",
    "DualOutputContext", "main_with_proper_encoding",
]
