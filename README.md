# 美股大涨预测系统 (模块化版本)

## 📊 项目概述

这是一个基于深度学习的美股大涨预测系统，采用模块化架构设计，集成了数据收集、特征工程、模型训练、样本失衡处理和回测功能。

## 🏗️ 模块化架构

### 核心模块

1. **`config.py`** - 配置中心
   - 训练参数配置
   - 模型架构配置  
   - 回测策略配置
   - 数据处理配置

2. **`data_collector.py`** - 数据收集器
   - 美股数据获取
   - 高级技术指标计算
   - 特征工程处理

3. **`dataset.py`** - 数据集处理
   - PyTorch数据集封装
   - 序列数据处理
   - 数据加载器创建

4. **`models.py`** - 模型定义
   - Transformer架构模型
   - Focal Loss损失函数
   - 模型保存/加载

5. **`imbalance_handler.py`** - 样本失衡处理
   - SMOTE、ADASYN等采样方法
   - 类别权重计算
   - 失衡分析工具

6. **`trainer.py`** - 训练器 ⭐ 新增
   - 模型训练流程
   - 早停机制
   - 模型评估

7. **`backtest.py`** - 回测引擎 ⭐ 新增
   - 单股票回测策略
   - 多股票回测分析
   - 回测结果统计

8. **`utils.py`** - 工具函数
   - 环境设置
   - 日志处理
   - 数据质量检查
   - 编码处理 ⭐ 增强

9. **`main.py`** - 主程序 ⭐ 重构
   - 完整训练流程
   - 多种运行模式
   - 模块集成

## 🚀 主要功能拆分

### 从 `transformer_test3.py` 中拆分出的功能：

#### ✅ 已拆分的功能
- ✅ 数据收集和技术指标计算 → `data_collector.py`
- ✅ 样本失衡处理 → `imbalance_handler.py`  
- ✅ 深度学习模型定义 → `models.py`
- ✅ 数据集处理 → `dataset.py`
- ✅ 配置管理 → `config.py`
- ✅ 基础工具函数 → `utils.py`

#### ⭐ 新增拆分的功能
- ⭐ **增强的模型训练函数** → `trainer.py`
  - `train_model_with_imbalance_handling()`
  - `ModelTrainer` 类
  - `ModelEvaluator` 类

- ⭐ **全面的模型评估函数** → `trainer.py`
  - `evaluate_model_comprehensive()`
  - 详细性能指标计算

- ⭐ **单股票回测策略** → `backtest.py`
  - `MLSurgeSingleStockStrategy` 类
  - 实时特征计算
  - 订单管理

- ⭐ **回测执行函数** → `backtest.py`
  - `run_single_stock_backtest()`
  - `run_multiple_stocks_backtest()`
  - 回测结果分析

- ⭐ **日志双输出与编码处理** → `utils.py`
  - `main_with_proper_encoding()`
  - `DualOutputContext` 上下文管理器
  - UTF-8编码处理

## 🔧 安装与使用

### 环境要求
```
Python >= 3.8
PyTorch >= 1.8
pandas >= 1.3
numpy >= 1.20
scikit-learn >= 1.0
imbalanced-learn >= 0.8
akshare >= 1.8
backtrader >= 1.9
```

### 安装依赖
```bash
pip install torch pandas numpy scikit-learn imbalanced-learn akshare backtrader matplotlib
```

### 运行方式

#### 1. 完整模式 (训练 + 回测)
```python
from us_stock_predictor.main import main_entry
import os

# 设置运行模式
os.environ['RUN_MODE'] = 'full'
main_entry()
```

#### 2. 仅训练模式
```python
import os
os.environ['RUN_MODE'] = 'train'
main_entry()
```

#### 3. 仅回测模式
```python
import os
os.environ['RUN_MODE'] = 'backtest'
main_entry()
```

#### 4. 演示模式
```python
import os
os.environ['RUN_MODE'] = 'demo'
main_entry()
```

### 直接运行
```bash
cd us_stock_predictor
python main.py
```

## 📁 文件结构

```
us_stock_predictor/
├── __init__.py              # 模块初始化
├── config.py                # 配置中心
├── data_collector.py        # 数据收集器
├── dataset.py               # 数据集处理
├── models.py                # 模型定义
├── imbalance_handler.py     # 样本失衡处理
├── trainer.py               # 训练器 (新增)
├── backtest.py              # 回测引擎 (新增)
├── utils.py                 # 工具函数 (增强)
├── main.py                  # 主程序 (重构)
├── README.md                # 项目文档
└── transformer_test3.py     # 原始文件 (保留作参考)
```

## 🔄 工作流程

### 1. 数据收集阶段
```python
from us_stock_predictor.data_collector import USStockDataCollector

# 初始化数据收集器
collector = USStockDataCollector()

# 收集训练数据（使用完整历史数据）
data = collector.prepare_features("19490101", "20250101")
```

### 2. 失衡处理阶段
```python
from us_stock_predictor.imbalance_handler import ImbalanceHandler

handler = ImbalanceHandler()
X_resampled, y_resampled = handler.apply_sampling(X, y, method='smote')
```

### 3. 模型训练阶段
```python
from us_stock_predictor.trainer import train_model_with_imbalance_handling

model, history = train_model_with_imbalance_handling(
    model, train_loader, val_loader,
    epochs=50, use_focal_loss=True
)
```

### 4. 回测验证阶段
```python
from us_stock_predictor.backtest import run_single_stock_backtest

result = run_single_stock_backtest(
    symbol='105.AAPL',
    start_date="20250102"
)
```

## ⚙️ 配置说明

### 主要配置项

#### 训练配置
```python
TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'patience': 10,
    'train_ratio': 0.7,
    'val_ratio': 0.15
}
```

#### 回测配置
```python
BACKTEST_CONFIG = {
    'initial_cash': 100000.0,
    'threshold': 0.5,
    'profit_target': 0.05,
    'stop_loss': -0.03,
    'hold_days': 5
}
```

## 🚨 重要改进

### 1. 模块化设计
- **解耦合**: 各模块职责明确，便于维护
- **可扩展**: 易于添加新功能或替换组件
- **可测试**: 每个模块可独立测试

### 2. 错误处理
- **异常捕获**: 统一的异常处理机制
- **失败恢复**: 模块失败不影响整体运行
- **日志记录**: 详细的错误日志

### 3. 性能优化
- **内存管理**: 定期检查内存使用
- **进度跟踪**: 可视化训练和测试进度
- **批处理**: 优化数据加载性能

### 4. 编码兼容
- **UTF-8支持**: 完整的中文支持
- **双输出**: 控制台和文件同时输出
- **跨平台**: Windows/Linux兼容

## 📊 模型特点

### 深度学习架构
- **Transformer编码器**: 处理序列数据
- **多任务学习**: 分类 + 回归
- **注意力机制**: 自适应特征权重

### 失衡处理策略
- **SMOTE**: 合成少数类样本
- **ADASYN**: 自适应合成采样
- **Focal Loss**: 难样本加权
- **类别权重**: 动态权重调整

### 回测策略
- **风险控制**: 止盈止损机制
- **资金管理**: 固定比例风险
- **信号强度**: AI模型概率阈值

## 🎯 使用建议

### 快速开始
1. 运行演示模式熟悉系统
2. 使用训练模式构建模型
3. 使用回测模式验证策略

### 生产环境
1. 调整配置参数
2. 增加股票池大小
3. 优化模型超参数
4. 实施风险管理

### 扩展开发
1. 添加新的技术指标
2. 实现新的失衡处理方法
3. 增加新的回测策略
4. 集成更多数据源

## 📝 更新日志

### v2.0 (模块化版本)
- ✅ 完整模块化拆分
- ✅ 新增训练器模块
- ✅ 新增回测引擎
- ✅ 增强工具函数
- ✅ 重构主程序
- ✅ 完善配置管理
- ✅ 优化错误处理
- ✅ 改进编码支持

### v1.0 (原始版本)
- 基础功能实现 (transformer_test3.py)

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目！

## 📄 许可证

本项目仅供学习研究使用，投资有风险，决策需谨慎。 



好的，我们来探讨一下 PatchTST 模型，以及它是否适合替换您项目中的现有模型。

### PatchTST 模型介绍

根据您提供的资料，**PatchTST (Patch Time Series Transformer)** 是一种为时间序列预测设计的、基于 Transformer 的新模型，它在多项长期预测任务中取得了当前最先进（State-of-the-Art）的成果。

它的核心思想借鉴了计算机视觉领域的 Vision Transformer (ViT)，非常有创新性：**它不将时间序列看作是单个时间点的连续序列，而是将其看作是一系列“补丁 (Patch)”的组合**。正如其论文标题《A Time Series is Worth 64 Words》所言，它把一段长时间序列数据转换成了数量更少的“单词”（即补丁），然后让 Transformer 来理解这些“单词”之间的关系。

#### 工作原理

1.  **分块 (Patching)**: 模型首先将输入的长时间序列（例如，过去96天的数据）分割成一个个更小的、连续的片段，这些片段就是“补丁”。比如，可以将96天的数据分割成16个补丁，每个补丁包含6天的数据。
2.  **特征嵌入 (Embedding)**: 每个补丁（一个包含多个时间点和多个特征的子序列）通过一个线性层被映射（“压扁”）成一个单一的向量，这个向量就是该补丁的“嵌入表示”，也就是一个“单词”。
3.  **Transformer 处理**: 经过分块和嵌入后，一个很长的时间序列就被转换成了一个很短的嵌入向量序列。这个短序列随后被送入一个标准的 Transformer Encoder 结构中进行处理。因为序列变短了，自注意力机制可以更高效地计算，并且能关注到更长的时间跨度。

#### 主要优势

根据 [PatchTST: A Breakthrough in Time Series Forecasting](https://medium.com/the-forecaster/patchtst-a-breakthrough-in-time-series-forecasting-e02d48869ccc) 和 [Understanding the PatchTST Model for Time Series Prediction](https://www.signalpop.com/2023/11/06/understanding-the-patchtst-model-for-time-series-prediction/) 的介绍，其优势主要有三点：

1.  **保留局部语义信息**: 每个补丁内部保留了一小段连续时间序列的形态信息，例如一个小型的“W底”或“M顶”形态，这使得模型能更好地学习到这些局部价格模式。
2.  **大幅降低计算和内存消耗**: Transformer中注意力计算的复杂度是序列长度的平方 `(O(L²))`。通过将长度为 `L` 的序列转换为长度为 `L/P` 的补丁序列（`P`是补丁长度），计算复杂度被大幅降低。
3.  **能关注更长的历史数据**: 由于计算效率的提升，模型可以在不增加过多计算负担的情况下，回顾（look-back）更长的历史数据，这对于捕捉金融市场中的长周期规律至关重要。

### 在您的项目中使用 PatchTST 替代现有模型

答案是：**非常可行，并且这是一个极具潜力的优化方向**。

您的现有模型 `StockSurgePredictor` 已经是一个基于 Transformer 的强大模型，而 PatchTST 是对标准时间序列 Transformer 的一种演进和优化。用它来替代现有模型是合乎逻辑的下一步。

#### 替换的可行性分析

*   **架构兼容性**: 核心都是 Transformer Encoder。您不需要重写整个模型，而是在输入端进行改造。
*   **性能潜力**: PatchTST 在多个公开数据集上被证实优于传统的 Transformer 模型，有很大潜力提升您模型的预测准确率。
*   **项目契合度**: 您的项目需要处理具有一定长度（`seq_len=20`）的序列，并且可能受益于更长的历史数据。PatchTST 的设计初衷与您的需求高度契合。

#### 需要进行的修改

替换工作主要涉及 `dataset.py` 和 `models.py` 两个文件。

1.  **修改 `models.py` (模型定义)**:
    *   您需要在 `StockSurgePredictor` 模型中增加一个“分块层 (Patching Layer)”。这个层负责将输入的 `(batch, seq_len, features)` 张量转换为 `(batch, num_patches, patch_embedding)` 的形状。
    *   您需要调整 Transformer Encoder 的输入，并可能需要修改位置编码的方式，使其应用于补丁序列而非时间步序列。
    *   模型的其他部分，如多任务头（分类器和回归器）和损失函数可以保持不变。

2.  **修改 `dataset.py` (数据处理)**:
    *   目前的 `_prepare_sequences` 方法是生成一个滑动窗口，每次移动一个时间步。为了适应 PatchTST，您需要修改数据准备的逻辑。
    *   不过，一个好消息是，**您可以不在 `dataset.py` 中进行修改**。可以让 `dataset.py` 依然按原样生成较长的序列（例如，生成长度为96的序列），然后在模型内部的 `forward` 函数中，由分块层动态地将这96个时间步分割成补丁。这种方式更灵活，改动也更小。

#### 潜在的挑战

*   **超参数选择**: 您需要引入新的超参数，如 `patch_len`（每个补丁的长度）和 `stride`（补丁之间的步长）。这些需要通过实验来确定最优值。
*   **多变量处理**: 原始的 PatchTST 论文提出了一个“通道独立 (channel-independent)”的策略，即为每个特征（通道）训练一个独立的模型。对于您的项目，所有特征（如价格、成交量、复合信号）是紧密相关的，因此应该采用“通道混合 (channel-mixing)”的方式，即将一个时间步内的所有特征作为一个整体进行处理，分块层也应该作用于整个特征矩阵。

### 总结

**使用 PatchTST 替换您项目中的现有模型是一个非常值得尝试的、有前途的优化方案**。它通过创新的分块机制解决了标准 Transformer 在处理长序列时的一些固有问题，非常适合金融时间序列预测。

虽然这需要一些实现上的调整，但鉴于您项目扎实的基础，完成这项升级是完全可行的。这将可能在模型性能和处理更长历史数据的能力上带来显著提升。



## 结论先行  
即使无法成功导入 `tsai`，您仍有 2 条靠谱途径在项目里落地 **PatchTST**：  

1. **直接使用 Hugging Face Transformers ≥ 4.41 内置的 PatchTST**（最简单、推荐）。  
2. **克隆官方 PatchTST 源码（PyTorch 实现）并手动集成**。  

下面按“理论——实践——落地到您现有工程”三步详细说明。

---

### 1  理论背景与可行性

1. **PatchTST 已被官方并入 Transformers**  
   Hugging Face 4.41+ 提供 `PatchTSTModel / PatchTSTForRegression / PatchTSTForClassification / PatchTSTForPrediction` 等完整实现及预训练权重 [链接](https://huggingface.co/docs/transformers/model_doc/patchtst)。这意味着只要您能安装 `transformers`，就能零依赖地创建或加载 PatchTST。  
2. **PatchTST 的 patching + Transformer 设计与您现有模型兼容**  
   - 输入仍是 `batch × seq_len × n_features` 的张量。  
   - patching 在模型内部完成，您不用改 `dataset.py`。  
   - 只需把现有的 `StockSurgePredictor` 类换成一个调用 PatchTST 的包装类。  

---

### 2  实践操作

#### 2.1 纯 Hugging Face 方案（推荐）

1. **安装或升级依赖**  
   ```bash
   pip install -U "transformers>=4.41"  # 版本低于 4.41 没有 PatchTST
   pip install accelerate  datasets  # 可选，但训练大模型时很方便
   ```

2. **创建模型与配置**  
   ```python
   import torch
   from transformers import PatchTSTConfig, PatchTSTForRegression

   config = PatchTSTConfig(
       num_input_channels = n_features,   # 您的特征维度
       context_length    = seq_len,       # 输入序列长度，如 96
       patch_length      = 12,            # 每个 patch 包含多少时间步
       patch_stride      = 12,            # patch 之间的间隔
       d_model           = 128,           # 隐藏维度
       num_hidden_layers = 4,
       num_attention_heads = 8,
       prediction_length = 1,             # 如果做回归预测 1 步
       num_targets       = 1
   )

   model = PatchTSTForRegression(config)  # 或 PatchTSTForClassification
   ```

3. **训练循环示例（回归）**  
   ```python
   optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

   for xb, yb in train_loader:             # xb: [B, L, C], yb: [B, 1]
       xb, yb = xb.to(device), yb.to(device)
       out = model(past_values=xb, target_values=yb)
       loss = out.loss
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()
   ```

4. **推理**  
   ```python
   with torch.no_grad():
       preds = model(past_values=xb).regression_outputs
   ```

5. **多任务（四分类 + 回归）**  
   - 方式一：各自实例化 `PatchTSTForClassification` 与 `PatchTSTForRegression`，共享 Encoder 权重（使用 `PatchTSTModel` 基础模型 + 两个自定义头）。  
   - 方式二：继承 `PatchTSTModel` 自行实现一个双头模块，逻辑与您当前 `StockSurgePredictor` 类似，只是把特征提取 + Transformer 替换为 PatchTST Encoder。

#### 2.2 直接克隆官方源码方案（备选）

```bash
git clone https://github.com/yuqinie98/PatchTST.git
cd PatchTST
pip install -r requirements.txt   # 相对干净，无 fastcore 依赖
```

然后把 `patchtst` 目录加到项目 PYTHONPATH，或复制到项目的 `models/` 目录。再按照 README 中的 `patchtst.py` 用法，包装成您自己的模型类即可。该仓库是纯 PyTorch，改动量比 Hugging Face 稍大，但同样无 fastcore 依赖。

---

### 3  集成到您现有工程的最小改动指南

1. **models.py**  
   ```python
   # ... existing code ...
   from transformers import PatchTSTModel

   class PatchTSTSurgePredictor(nn.Module):
       def __init__(self, input_dim, cfg):
           super().__init__()
           self.encoder = PatchTSTModel(cfg)
           hidden = cfg.d_model
           # 四分类头
           self.cls_head = nn.Sequential(
               nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 4))
           # 回归头
           self.reg_head = nn.Sequential(
               nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))

       def forward(self, x):           # x: [B, L, C]
           enc_out = self.encoder(past_values=x).last_hidden_state  # [B, N, H]
           feat = enc_out[:, -1]       # 取最后 patch 的 CLS/token
           return self.cls_head(feat), self.reg_head(feat)
   ```
   这样就完成了与原 `StockSurgePredictor` 的对接，训练代码几乎不用改。

2. **dataset.py**  
   无需修改。保持输出 `[B, seq_len, n_features]` 即可。

3. **config.py**  
   增加 PatchTST 专用参数区块（patch_len、stride 等）。

---

### 4  FAQ

| 问题 | 解决办法 |
| --- | --- |
| tsai 仍想用？ | 等待 tsai 修复 fastcore → fasttransform 兼容；或手动安装 `fasttransform` 再 `tsai`，但 Hugging Face 方案更省事 |
| 想使用官方 PatchTST 预训练权重？ | `PatchTSTForRegression.from_pretrained("namctin/patchtst_etth1_regression")` 等 |
| 金融场景输入是多变量 | 将 `num_input_channels = 特征数`，开启 `channel_attention=True` 可让各特征互相关注。 |

---

## 小结

* **无法导入 tsai 并不影响您用 PatchTST** —— 直接使用 Hugging Face Transformers 的官方实现即可，安装简单、社区活跃、文档完备。  
* 您现有工程只需在 `models.py` 换掉特征抽取 + Transformer 部分，其余数据管道、训练器、回测逻辑几乎不受影响。  
