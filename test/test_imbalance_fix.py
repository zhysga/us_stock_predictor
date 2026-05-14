# -*- coding: utf-8 -*-
"""
测试SMOTE修复效果的脚本
用于验证失衡处理的改进是否解决了原有问题
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import Counter

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from us_stock_predictor.imbalance_handler import ImbalanceHandler
from us_stock_predictor.config.core import IMBALANCE_CONFIG
import warnings
warnings.filterwarnings('ignore')

def create_test_data():
    """创建各种失衡程度的测试数据"""
    test_cases = {
        'extreme_imbalance': {
            'description': '极度不平衡 (1:100)',
            'positive_samples': 50,
            'negative_samples': 5000,
            'features': 10
        },
        'severe_imbalance': {
            'description': '严重不平衡 (1:20)', 
            'positive_samples': 100,
            'negative_samples': 2000,
            'features': 10
        },
        'moderate_imbalance': {
            'description': '中度不平衡 (1:5)',
            'positive_samples': 200,
            'negative_samples': 1000,
            'features': 10
        },
        'mild_imbalance': {
            'description': '轻度不平衡 (1:2)',
            'positive_samples': 500,
            'negative_samples': 1000,
            'features': 10
        }
    }
    
    datasets = {}
    
    for case_name, config in test_cases.items():
        print(f"\n[创建] {config['description']} 测试数据...")
        
        # 创建特征数据
        pos_features = np.random.randn(config['positive_samples'], config['features'])
        neg_features = np.random.randn(config['negative_samples'], config['features'])
        
        X = np.vstack([pos_features, neg_features])
        y = np.hstack([
            np.ones(config['positive_samples']),  # 正样本
            np.zeros(config['negative_samples'])  # 负样本
        ])
        
        # 打乱数据
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        datasets[case_name] = {'X': X, 'y': y, 'description': config['description']}
        
        # 显示原始分布
        counter = Counter(y)
        ratio = counter[0] / counter[1] if counter[1] > 0 else float('inf')
        print(f"   原始分布: 负样本={counter[0]}, 正样本={counter[1]}, 比例={ratio:.1f}:1")
    
    return datasets

def test_sampling_methods(datasets):
    """测试各种采样方法"""
    print(f"\n{'='*60}")
    print("开始测试失衡处理方法")
    print(f"{'='*60}")
    
    handler = ImbalanceHandler()
    methods = ['smote', 'adasyn', 'smote_tomek', 'undersample']
    
    results = {}
    
    for case_name, data in datasets.items():
        print(f"\n{'-'*40}")
        print(f"测试案例: {data['description']}")
        print(f"{'-'*40}")
        
        X, y = data['X'], data['y']
        case_results = {}
        
        # 原始分布分析
        original_counter, original_ratio = handler.analyze_imbalance(y)
        
        for method in methods:
            print(f"\n--- 测试 {method.upper()} 方法 ---")
            try:
                X_resampled, y_resampled = handler.apply_sampling(X, y, method)
                
                # 分析结果
                new_counter, new_ratio = handler.analyze_imbalance(y_resampled)
                
                case_results[method] = {
                    'success': True,
                    'original_samples': len(y),
                    'resampled_samples': len(y_resampled),
                    'original_ratio': original_ratio,
                    'new_ratio': new_ratio,
                    'improvement': original_ratio / new_ratio if new_ratio > 0 else 0
                }
                
                print(f"✅ {method.upper()} 成功")
                print(f"   样本数变化: {len(y)} → {len(y_resampled)}")
                print(f"   失衡改善: {original_ratio:.2f}:1 → {new_ratio:.2f}:1")
                
            except Exception as e:
                print(f"❌ {method.upper()} 失败: {str(e)}")
                case_results[method] = {
                    'success': False,
                    'error': str(e)
                }
        
        results[case_name] = case_results
    
    return results

def test_adaptive_strategy():
    """测试自适应策略"""
    print(f"\n{'='*60}")
    print("测试自适应失衡处理策略")
    print(f"{'='*60}")
    
    handler = ImbalanceHandler()
    
    # 创建极端不平衡数据
    X = np.random.randn(1010, 5)
    y = np.hstack([np.ones(10), np.zeros(1000)])  # 1:100 的极端不平衡
    
    print("[原始数据]")
    handler.analyze_imbalance(y)
    
    print("\n[测试自动选择最佳方法]")
    try:
        X_best, y_best, best_method = handler.apply_best_sampling(X, y)
        
        if best_method:
            print(f"✅ 自动选择最佳方法: {best_method.upper()}")
            print("[处理后数据]")
            handler.analyze_imbalance(y_best)
        else:
            print("ℹ️ 建议使用原始数据")
            
    except Exception as e:
        print(f"❌ 自适应策略失败: {e}")

def test_edge_cases():
    """测试边界情况"""
    print(f"\n{'='*60}")
    print("测试边界情况")
    print(f"{'='*60}")
    
    handler = ImbalanceHandler()
    
    # 测试案例1：只有一个类别
    print("\n[案例1] 只有一个类别")
    X1 = np.random.randn(100, 5)
    y1 = np.zeros(100)  # 全部是0类
    
    try:
        X1_res, y1_res = handler.apply_sampling(X1, y1, 'smote')
        print("✅ 单类别处理成功")
    except Exception as e:
        print(f"❌ 单类别处理失败: {e}")
    
    # 测试案例2：少数类样本极少
    print("\n[案例2] 少数类样本极少")
    X2 = np.random.randn(102, 5)
    y2 = np.hstack([np.ones(1), np.zeros(101)])  # 只有1个正样本
    
    try:
        X2_res, y2_res = handler.apply_sampling(X2, y2, 'smote')
        print("✅ 极少样本处理成功")
        handler.analyze_imbalance(y2_res)
    except Exception as e:
        print(f"❌ 极少样本处理失败: {e}")
    
    # 测试案例3：已经平衡的数据
    print("\n[案例3] 已经平衡的数据")
    X3 = np.random.randn(200, 5)
    y3 = np.hstack([np.ones(100), np.zeros(100)])  # 1:1 平衡
    
    try:
        X3_res, y3_res = handler.apply_sampling(X3, y3, 'smote')
        print("✅ 平衡数据处理成功")
        handler.analyze_imbalance(y3_res)
    except Exception as e:
        print(f"❌ 平衡数据处理失败: {e}")

def print_summary(results):
    """打印测试总结"""
    print(f"\n{'='*60}")
    print("测试结果总结")
    print(f"{'='*60}")
    
    total_tests = 0
    successful_tests = 0
    
    for case_name, case_results in results.items():
        print(f"\n[{case_name.upper()}]")
        for method, result in case_results.items():
            total_tests += 1
            if result['success']:
                successful_tests += 1
                improvement = result['improvement']
                print(f"  ✅ {method.upper()}: 改善 {improvement:.1f}x")
            else:
                print(f"  ❌ {method.upper()}: {result['error'][:50]}...")
    
    success_rate = successful_tests / total_tests * 100
    print(f"\n[总体结果]")
    print(f"成功率: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 修复效果良好！")
    elif success_rate >= 60:
        print("⚠️ 修复效果一般，仍有改进空间")
    else:
        print("❌ 修复效果不佳，需要进一步改进")

def main():
    """主测试函数"""
    print("🔧 SMOTE修复效果测试")
    print("="*60)
    
    # 1. 创建测试数据
    datasets = create_test_data()
    
    # 2. 测试各种采样方法
    results = test_sampling_methods(datasets)
    
    # 3. 测试自适应策略
    test_adaptive_strategy()
    
    # 4. 测试边界情况
    test_edge_cases()
    
    # 5. 打印总结
    print_summary(results)
    
    print(f"\n🏁 测试完成")

if __name__ == "__main__":
    main() 