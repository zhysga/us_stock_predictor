# -*- coding: utf-8 -*-
"""
四分类模型测试脚本
验证模型修改后的功能是否正常
"""

import os
import sys
import torch
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from us_stock_predictor.config.core import PRICE_CHANGE_THRESHOLDS, PRICE_CHANGE_LABELS
from us_stock_predictor.data.collector import USStockDataCollector
from us_stock_predictor.models.transformer_surge import StockSurgePredictor, FocalLoss, create_focal_loss
from us_stock_predictor.datasets.timeseries import StockDataset
from torch.utils.data import DataLoader


def test_four_class_labels():
    """测试四分类标签生成"""
    print("=" * 50)
    print("测试四分类标签生成")
    print("=" * 50)
    
    collector = USStockDataCollector()
    
    # 测试不同的价格变化值
    test_values = [-0.10, -0.08, -0.05, -0.02, 0.00, 0.02, 0.05, 0.08, 0.10]
    
    print("价格变化值 -> 分类标签")
    print("-" * 30)
    
    labels = collector._generate_four_class_labels(pd.Series(test_values))
    
    for value, label in zip(test_values, labels):
        if label == 0:
            category = "大跌"
        elif label == 1:
            category = "小跌"
        elif label == 2:
            category = "小涨"
        else:
            category = "大涨"
        
        print(f"{value:6.2f} -> {label} ({category})")
    
    print(f"\n阈值设定: 大跌<={PRICE_CHANGE_THRESHOLDS['big_drop']}, 大涨>={PRICE_CHANGE_THRESHOLDS['big_surge']}")


def test_model_output():
    """测试模型输出维度"""
    print("\n" + "=" * 50)
    print("测试模型输出维度")
    print("=" * 50)
    
    # 创建模型
    input_dim = 64  # 示例特征维度
    model = StockSurgePredictor(input_dim=input_dim)
    
    # 创建测试输入
    batch_size = 4
    seq_len = 20
    test_input = torch.randn(batch_size, seq_len, input_dim)
    
    # 前向传播
    with torch.no_grad():
        class_logits, magnitude = model(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"分类输出形状: {class_logits.shape}")
    print(f"回归输出形状: {magnitude.shape}")
    print(f"期望分类输出: ({batch_size}, 4)")
    print(f"期望回归输出: ({batch_size}, 1)")
    
    # 验证输出维度
    assert class_logits.shape == (batch_size, 4), f"分类输出维度错误: {class_logits.shape}"
    assert magnitude.shape == (batch_size, 1), f"回归输出维度错误: {magnitude.shape}"
    
    print("✓ 模型输出维度正确")
    
    # 测试softmax概率
    probs = torch.softmax(class_logits, dim=1)
    print(f"\n四分类概率示例 (第一个样本):")
    prob_names = ["大跌", "小跌", "小涨", "大涨"]
    for i, (name, prob) in enumerate(zip(prob_names, probs[0])):
        print(f"  {name}: {prob.item():.4f}")
    
    print(f"概率和: {probs[0].sum().item():.4f}")


def test_focal_loss():
    """测试Focal Loss"""
    print("\n" + "=" * 50)
    print("测试Focal Loss")
    print("=" * 50)
    
    # 创建测试数据
    batch_size = 8
    num_classes = 4
    
    # 模拟预测logits和真实标签
    pred_logits = torch.randn(batch_size, num_classes)
    true_labels = torch.randint(0, num_classes, (batch_size,))
    
    print(f"预测logits形状: {pred_logits.shape}")
    print(f"真实标签形状: {true_labels.shape}")
    print(f"真实标签: {true_labels.tolist()}")
    
    # 测试Focal Loss
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    loss = focal_loss(pred_logits, true_labels)
    
    print(f"Focal Loss: {loss.item():.4f}")
    
    # 测试带类别权重的Focal Loss
    class_weights = {0: 2.0, 1: 1.0, 2: 1.0, 3: 3.0}  # 大跌和大涨权重更高
    focal_loss_weighted = FocalLoss(alpha=1.0, gamma=2.0, class_weights=class_weights)
    loss_weighted = focal_loss_weighted(pred_logits, true_labels)
    
    print(f"加权Focal Loss: {loss_weighted.item():.4f}")
    print("✓ Focal Loss计算正确")


def test_data_pipeline():
    """测试数据管道"""
    print("\n" + "=" * 50)
    print("测试数据管道")
    print("=" * 50)
    
    try:
        # 收集小量数据进行测试
        collector = USStockDataCollector()
        collector.symbols = collector.symbols[:1]  # 只用一只股票
        
        print("正在获取测试数据...")
        data = collector.prepare_features("20240101", "20241201")
        
        if data.empty:
            print("⚠ 无法获取数据，跳过数据管道测试")
            return
        
        print(f"数据形状: {data.shape}")
        print(f"特征列数: {len(collector.feature_cols)}")
        
        # 检查四分类标签
        if 'price_change_class' in data.columns:
            class_counts = data['price_change_class'].value_counts().sort_index()
            print("\n四分类分布:")
            class_names = ['大跌', '小跌', '小涨', '大涨']
            for label, count in class_counts.items():
                print(f"  {class_names[label]}(标签{label}): {count}")
            
            print("✓ 四分类标签生成正确")
        else:
            print("✗ 未找到四分类标签列")
            return
        
        # 创建数据集
        dataset = StockDataset(data, collector.feature_cols)
        print(f"数据集大小: {len(dataset)}")
        
        # 测试数据加载
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        for batch_idx, (features, labels, magnitudes) in enumerate(dataloader):
            print(f"\n批次 {batch_idx}:")
            print(f"  特征形状: {features.shape}")
            print(f"  标签形状: {labels.shape}")
            print(f"  标签值: {labels.squeeze().tolist()}")
            print(f"  涨跌幅: {magnitudes.squeeze().tolist()}")
            
            if batch_idx >= 1:  # 只显示前2个批次
                break
        
        print("✓ 数据管道测试通过")
        
    except Exception as e:
        print(f"✗ 数据管道测试失败: {e}")


def test_training_compatibility():
    """测试训练兼容性"""
    print("\n" + "=" * 50)
    print("测试训练兼容性")
    print("=" * 50)
    
    try:
        # 创建模拟数据
        batch_size = 8
        seq_len = 20
        input_dim = 32
        
        # 模拟特征和标签
        features = torch.randn(batch_size, seq_len, input_dim)
        labels = torch.randint(0, 4, (batch_size,))  # 四分类标签
        magnitudes = torch.randn(batch_size, 1)
        
        # 创建模型
        model = StockSurgePredictor(input_dim=input_dim)
        
        # 创建损失函数
        focal_loss = create_focal_loss()
        mse_loss = torch.nn.MSELoss()
        
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        print("开始模拟训练步骤...")
        
        # 前向传播
        class_logits, pred_magnitudes = model(features)
        
        # 计算损失
        classification_loss = focal_loss(class_logits, labels)
        regression_loss = mse_loss(pred_magnitudes, magnitudes)
        total_loss = classification_loss + 0.1 * regression_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"分类损失: {classification_loss.item():.4f}")
        print(f"回归损失: {regression_loss.item():.4f}")
        print(f"总损失: {total_loss.item():.4f}")
        
        # 验证预测
        with torch.no_grad():
            probs = torch.softmax(class_logits, dim=1)
            predicted_classes = torch.argmax(probs, dim=1)
            
            print(f"真实标签: {labels.tolist()}")
            print(f"预测标签: {predicted_classes.tolist()}")
        
        print("✓ 训练兼容性测试通过")
        
    except Exception as e:
        print(f"✗ 训练兼容性测试失败: {e}")


def main():
    """运行所有测试"""
    print("四分类模型测试")
    print("=" * 50)
    
    # 设置随机种子以确保结果可重复
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        test_four_class_labels()
        test_model_output()
        test_focal_loss()
        test_data_pipeline()
        test_training_compatibility()
        
        print("\n" + "=" * 50)
        print("所有测试完成")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 