# -*- coding: utf-8 -*-
"""
增强的样本失衡处理模块
支持多种采样方法和动态参数调整
"""

import numpy as np
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler

from config import TARGET_RATIO, IMBALANCE_METHODS
from utils import handle_exception

class ImbalanceHandler:
    """样本失衡处理工具类"""
    
    def __init__(self, target_ratio=None):
        """
        初始化失衡处理器
        target_ratio: 目标比例，例如0.1表示少数类:多数类=1:10
        """
        if target_ratio is None:
            target_ratio = TARGET_RATIO
            
        self.target_ratio = target_ratio
        
        # 动态调整采样策略参数
        self.method_map = self._create_dynamic_samplers()

    def _create_dynamic_samplers(self):
        """创建动态调整的采样器"""
        # 使用 'auto' 让算法自动调整，或者设为 None 表示自动平衡
        samplers = {
            'smote': None,  # 延迟初始化
            'adasyn': None,
            'smote_tomek': None,
            'undersample': None
        }
        return samplers
    
    def _get_adaptive_sampling_strategy(self, y, method='smote'):
        """根据数据分布自适应调整采样策略"""
        try:
            counter = Counter(y)
            classes = sorted(counter.keys())
            
            if len(classes) < 2:
                return 'auto'
            
            # 计算当前失衡比例
            minority_count = min(counter.values())
            majority_count = max(counter.values())
            current_ratio = minority_count / majority_count
            
            print(f"[分析] 当前失衡比例: {minority_count}/{majority_count} = {current_ratio:.4f}")
            
            # 根据方法调整策略
            if method in ['smote', 'adasyn']:
                # 对于极度不平衡的数据，逐步增加少数类样本
                if current_ratio < 0.01:  # 小于1%
                    target_minority_samples = max(minority_count * 3, 20)  # 至少增加到3倍或20个
                elif current_ratio < 0.05:  # 小于5%
                    target_minority_samples = max(minority_count * 2, 50)
                else:
                    # 使用目标比例计算
                    target_minority_samples = int(majority_count * self.target_ratio)
                
                # 确保不超过多数类样本数
                target_minority_samples = min(target_minority_samples, majority_count)
                
                # 构建采样策略字典
                strategy = {}
                for class_label in classes:
                    if counter[class_label] == minority_count:
                        strategy[class_label] = target_minority_samples
                    else:
                        strategy[class_label] = counter[class_label]  # 保持多数类不变
                
                print(f"[策略] {method.upper()} 采样策略: {strategy}")
                return strategy
                
            elif method == 'undersample':
                # 下采样：减少多数类样本
                target_majority_samples = max(int(minority_count / self.target_ratio), minority_count * 2)
                
                strategy = {}
                for class_label in classes:
                    if counter[class_label] == majority_count:
                        strategy[class_label] = target_majority_samples
                    else:
                        strategy[class_label] = counter[class_label]  # 保持少数类不变
                
                print(f"[策略] UNDERSAMPLE 采样策略: {strategy}")
                return strategy
            
            else:
                return 'auto'
                
        except Exception as e:
            print(f"[警告] 自适应策略计算失败: {e}, 使用自动模式")
            return 'auto'
    
    def _create_sampler_with_strategy(self, method, y):
        """根据数据分布创建合适的采样器"""
        try:
            strategy = self._get_adaptive_sampling_strategy(y, method)
            
            # 创建采样器时添加更多容错参数
            if method == 'smote':
                return SMOTE(
                    sampling_strategy=strategy,
                    random_state=42,
                    k_neighbors=min(5, len(y)//2)  # 动态调整邻居数
                )
            elif method == 'adasyn':
                return ADASYN(
                    sampling_strategy=strategy,
                    random_state=42,
                    n_neighbors=min(5, len(y)//2)
                )
            elif method == 'smote_tomek':
                return SMOTETomek(
                    sampling_strategy=strategy,
                    random_state=42,
                    smote=SMOTE(k_neighbors=min(5, len(y)//2))
                )
            elif method == 'undersample':
                return RandomUnderSampler(
                    sampling_strategy=strategy,
                    random_state=42
                )
            else:
                raise ValueError(f"不支持的采样方法: {method}")
                
        except Exception as e:
            print(f"[回退] 创建 {method} 采样器失败: {e}, 使用简化策略")
            # 回退到最简单的策略
            if method == 'undersample':
                return RandomUnderSampler(sampling_strategy='auto', random_state=42)
            else:
                return SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=1)

    def analyze_imbalance(self, y):
        """分析样本分布"""
        try:
            counter = Counter(y)
            total = len(y)
            print(f"\n[数据] 样本分布分析:")
            for class_label, count in counter.items():
                print(f"   类别 {class_label}: {count} 样本 ({count/total*100:.2f}%)")
            
            # 计算失衡比例
            if len(counter) >= 2:
                minority_class = min(counter.values())
                majority_class = max(counter.values())
                imbalance_ratio = majority_class / minority_class
                print(f"   失衡比例: {imbalance_ratio:.2f}:1")
            else:
                imbalance_ratio = 1.0
                print(f"   只有一个类别，无失衡问题")
            
            return counter, imbalance_ratio
            
        except Exception as e:
            handle_exception(e, "分析样本分布")
            return {}, 1.0
    
    def apply_sampling(self, X, y, method='smote'):
        """应用采样方法"""
        if method not in ['smote', 'adasyn', 'smote_tomek', 'undersample']:
            raise ValueError(f"不支持的方法: {method}。支持的方法: ['smote', 'adasyn', 'smote_tomek', 'undersample']")
        
        try:
            print(f"\n[处理] 应用 {method.upper()} 采样方法...")
            
            # 原始分布
            original_counter, original_ratio = self.analyze_imbalance(y)
            
            # 检查是否需要采样
            if len(original_counter) < 2:
                print("[警告] 只有一个类别，跳过采样")
                return X, y
            
            # 检查数据量是否足够
            minority_count = min(original_counter.values())
            if minority_count < 2:
                print(f"[警告] 少数类样本过少({minority_count})，无法进行{method}采样")
                if method != 'undersample':
                    print("[回退] 尝试使用下采样方法")
                    return self.apply_sampling(X, y, 'undersample')
                else:
                    print("[跳过] 返回原始数据")
                    return X, y
            
            # 动态创建采样器
            sampler = self._create_sampler_with_strategy(method, y)
            
            # 应用采样
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            # 新分布
            print(f"\n[结果] {method.upper()} 采样后分布:")
            new_counter, new_ratio = self.analyze_imbalance(y_resampled)
            
            # 验证结果合理性
            if len(y_resampled) > len(y) * 5:  # 样本数增加超过5倍
                print(f"[警告] 采样后样本数过多，考虑使用其他方法")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"[错误] {method}采样失败: {e}")
            
            # 智能回退策略
            if method != 'undersample':
                print(f"[回退] 尝试使用下采样方法")
                return self.apply_sampling(X, y, 'undersample')
            else:
                print(f"[最终回退] 返回原始数据并调整类别权重")
                return X, y
    
    def get_class_weights(self, y):
        """计算类别权重"""
        try:
            classes = np.unique(y)
            
            if len(classes) < 2:
                print("[警告] 只有一个类别，返回均等权重")
                return {classes[0]: 1.0} if len(classes) == 1 else {}
            
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            weight_dict = dict(zip(classes, class_weights))
            
            print(f"\n[权重] 计算得到的类别权重:")
            for class_label, weight in weight_dict.items():
                print(f"   类别 {class_label}: {weight:.4f}")
            
            return weight_dict
            
        except Exception as e:
            handle_exception(e, "计算类别权重")
            return {}
    
    def create_weighted_sampler(self, y):
        """创建加权采样器"""
        try:
            class_weights = self.get_class_weights(y)
            
            if not class_weights:
                print("[警告] 无法创建加权采样器，使用默认采样")
                return None
            
            sample_weights = [class_weights.get(label, 1.0) for label in y]
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            print(f"[成功] 创建加权采样器，样本数: {len(sample_weights)}")
            return sampler
            
        except Exception as e:
            handle_exception(e, "创建加权采样器")
            return None
    
    def evaluate_sampling_methods(self, X, y, methods=None):
        """评估多种采样方法的效果"""
        if methods is None:
            methods = IMBALANCE_METHODS
        
        results = {}
        original_counter, original_ratio = self.analyze_imbalance(y)
        
        print(f"\n[评估] 开始评估采样方法效果...")
        print(f"[原始] 失衡比例: {original_ratio:.2f}:1")
        
        for method in methods:
            try:
                print(f"\n--- 评估 {method.upper()} ---")
                X_resampled, y_resampled = self.apply_sampling(X, y, method)
                
                # 计算新的分布
                new_counter, new_ratio = self.analyze_imbalance(y_resampled)
                
                # 计算改善程度
                improvement = original_ratio / new_ratio if new_ratio > 0 else 0
                
                results[method] = {
                    'original_samples': len(y),
                    'resampled_samples': len(y_resampled),
                    'original_ratio': original_ratio,
                    'new_ratio': new_ratio,
                    'improvement_factor': improvement,
                    'distribution': new_counter
                }
                
                print(f"[{method.upper()}] 原始样本: {len(y)} -> 重采样后: {len(y_resampled)}")
                print(f"[{method.upper()}] 失衡改善: {improvement:.2f}倍")
                
            except Exception as e:
                print(f"[错误] {method} 评估失败: {e}")
                results[method] = None
        
        return results
    
    def recommend_best_method(self, X, y, methods=None):
        """推荐最佳的采样方法"""
        print(f"\n[推荐] 开始评估并推荐最佳采样方法...")
        
        results = self.evaluate_sampling_methods(X, y, methods)
        
        best_method = None
        best_score = 0
        
        for method, result in results.items():
            if result is None:
                continue
            
            # 评分标准：平衡改善程度 + 样本效率
            improvement_score = min(result['improvement_factor'], 10)  # 限制最大改善分数
            efficiency_score = result['original_samples'] / result['resampled_samples']  # 样本效率
            
            # 综合评分
            total_score = improvement_score * 0.7 + efficiency_score * 0.3
            
            print(f"[评分] {method.upper()}: 改善分数={improvement_score:.2f}, 效率分数={efficiency_score:.2f}, 总分={total_score:.2f}")
            
            if total_score > best_score:
                best_score = total_score
                best_method = method
        
        if best_method:
            print(f"\n[推荐] 最佳方法: {best_method.upper()} (总分: {best_score:.2f})")
            return best_method, results[best_method]
        else:
            print(f"\n[推荐] 无法找到合适的采样方法，建议使用原始数据")
            return None, None
    
    def apply_best_sampling(self, X, y):
        """自动选择并应用最佳采样方法"""
        try:
            # 首先分析原始数据
            original_counter, original_ratio = self.analyze_imbalance(y)
            
            # 如果失衡程度较轻，可能不需要采样
            if original_ratio <= 3.0:
                print(f"[分析] 失衡程度较轻 ({original_ratio:.2f}:1)，建议使用原始数据")
                return X, y, None
            
            # 推荐最佳方法
            best_method, best_result = self.recommend_best_method(X, y)
            
            if best_method:
                # 应用最佳方法
                X_resampled, y_resampled = self.apply_sampling(X, y, best_method)
                print(f"[应用] 使用 {best_method.upper()} 方法处理数据")
                return X_resampled, y_resampled, best_method
            else:
                print(f"[保持] 使用原始数据")
                return X, y, None
                
        except Exception as e:
            handle_exception(e, "自动选择采样方法")
            return X, y, None
    
    def get_sampling_statistics(self, y_original, y_resampled, method_name="采样"):
        """获取采样前后的统计信息"""
        try:
            print(f"\n[统计] {method_name}前后对比:")
            print("-" * 40)
            
            # 原始数据统计
            original_counter = Counter(y_original)
            print(f"原始数据:")
            for class_label, count in original_counter.items():
                print(f"  类别 {class_label}: {count} ({count/len(y_original)*100:.1f}%)")
            
            # 重采样数据统计
            resampled_counter = Counter(y_resampled)
            print(f"重采样数据:")
            for class_label, count in resampled_counter.items():
                print(f"  类别 {class_label}: {count} ({count/len(y_resampled)*100:.1f}%)")
            
            # 变化统计
            print(f"变化:")
            print(f"  总样本数: {len(y_original)} -> {len(y_resampled)} ({len(y_resampled)/len(y_original):.2f}x)")
            
            # 计算类别比例变化
            for class_label in original_counter.keys():
                original_pct = original_counter[class_label] / len(y_original) * 100
                resampled_pct = resampled_counter.get(class_label, 0) / len(y_resampled) * 100
                change = resampled_pct - original_pct
                print(f"  类别 {class_label}: {original_pct:.1f}% -> {resampled_pct:.1f}% ({change:+.1f}%)")
            
            print("-" * 40)
            
        except Exception as e:
            handle_exception(e, "获取采样统计信息") 