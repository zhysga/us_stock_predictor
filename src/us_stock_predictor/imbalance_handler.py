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

from us_stock_predictor.config.core import TARGET_RATIO, IMBALANCE_METHODS
from us_stock_predictor.utils.core import handle_exception


class ImbalanceHandler:
    """样本失衡处理工具类"""

    def __init__(self, target_ratio=None):
        if target_ratio is None:
            target_ratio = TARGET_RATIO
        self.target_ratio = target_ratio
        self.method_map = self._create_dynamic_samplers()

    def _create_dynamic_samplers(self):
        return {
            'smote': None,
            'adasyn': None,
            'smote_tomek': None,
            'undersample': None,
        }

    def _get_adaptive_sampling_strategy(self, y, method='smote'):
        try:
            counter = Counter(y)
            classes = sorted(counter.keys())
            if len(classes) < 2:
                return 'auto'

            minority_count = min(counter.values())
            majority_count = max(counter.values())
            current_ratio = minority_count / majority_count
            print(f"[分析] 当前失衡比例: {minority_count}/{majority_count} = {current_ratio:.4f}")

            if method in ['smote', 'adasyn']:
                if current_ratio < 0.01:
                    target_minority_samples = max(minority_count * 3, 20)
                elif current_ratio < 0.05:
                    target_minority_samples = max(minority_count * 2, 50)
                else:
                    target_minority_samples = int(majority_count * self.target_ratio)
                target_minority_samples = min(target_minority_samples, majority_count)

                strategy = {}
                for class_label in classes:
                    if counter[class_label] == minority_count:
                        strategy[class_label] = target_minority_samples
                    else:
                        strategy[class_label] = counter[class_label]
                print(f"[策略] {method.upper()} 采样策略: {strategy}")
                return strategy

            if method == 'undersample':
                target_majority_samples = max(int(minority_count / self.target_ratio), minority_count * 2)
                strategy = {}
                for class_label in classes:
                    if counter[class_label] == majority_count:
                        strategy[class_label] = target_majority_samples
                    else:
                        strategy[class_label] = counter[class_label]
                print(f"[策略] UNDERSAMPLE 采样策略: {strategy}")
                return strategy

            return 'auto'
        except Exception as e:
            print(f"[警告] 自适应策略计算失败: {e}, 使用自动模式")
            return 'auto'

    def _create_sampler_with_strategy(self, method, y):
        try:
            strategy = self._get_adaptive_sampling_strategy(y, method)
            if method == 'smote':
                return SMOTE(sampling_strategy=strategy, random_state=42, k_neighbors=min(5, len(y) // 2))
            if method == 'adasyn':
                return ADASYN(sampling_strategy=strategy, random_state=42, n_neighbors=min(5, len(y) // 2))
            if method == 'smote_tomek':
                return SMOTETomek(
                    sampling_strategy=strategy,
                    random_state=42,
                    smote=SMOTE(k_neighbors=min(5, len(y) // 2)),
                )
            if method == 'undersample':
                return RandomUnderSampler(sampling_strategy=strategy, random_state=42)
            raise ValueError(f"不支持的采样方法: {method}")
        except Exception as e:
            print(f"[回退] 创建 {method} 采样器失败: {e}, 使用简化策略")
            if method == 'undersample':
                return RandomUnderSampler(sampling_strategy='auto', random_state=42)
            return SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=1)

    def analyze_imbalance(self, y):
        try:
            counter = Counter(y)
            total = len(y)
            print(f"\n[数据] 样本分布分析:")
            for class_label, count in counter.items():
                print(f"   类别 {class_label}: {count} 样本 ({count/total*100:.2f}%)")
            if len(counter) >= 2:
                minority_class = min(counter.values())
                majority_class = max(counter.values())
                imbalance_ratio = majority_class / minority_class
                print(f"   失衡比例: {imbalance_ratio:.2f}:1")
            else:
                imbalance_ratio = 1.0
                print("   只有一个类别，无失衡问题")
            return counter, imbalance_ratio
        except Exception as e:
            handle_exception(e, "分析样本分布")
            return {}, 1.0

    def apply_sampling(self, X, y, method='smote'):
        if method not in ['smote', 'adasyn', 'smote_tomek', 'undersample']:
            raise ValueError("不支持的方法: %s。支持的方法: ['smote', 'adasyn', 'smote_tomek', 'undersample']" % method)
        try:
            print(f"\n[处理] 应用 {method.upper()} 采样方法...")
            original_counter, _ = self.analyze_imbalance(y)
            if len(original_counter) < 2:
                print("[警告] 只有一个类别，跳过采样")
                return X, y
            minority_count = min(original_counter.values())
            if minority_count < 2:
                print(f"[警告] 少数类样本过少({minority_count})，无法进行{method}采样")
                if method != 'undersample':
                    print("[回退] 尝试使用下采样方法")
                    return self.apply_sampling(X, y, 'undersample')
                print("[跳过] 返回原始数据")
                return X, y
            sampler = self._create_sampler_with_strategy(method, y)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            print(f"\n[结果] {method.upper()} 采样后分布:")
            self.analyze_imbalance(y_resampled)
            if len(y_resampled) > len(y) * 5:
                print("[警告] 采样后样本数过多，考虑使用其他方法")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"[错误] {method}采样失败: {e}")
            if method != 'undersample':
                print("[回退] 尝试使用下采样方法")
                return self.apply_sampling(X, y, 'undersample')
            print("[最终回退] 返回原始数据并调整类别权重")
            return X, y

    def get_class_weights(self, y):
        try:
            classes = np.unique(y)
            if len(classes) < 2:
                print("[警告] 只有一个类别，返回均等权重")
                return {classes[0]: 1.0} if len(classes) == 1 else {}
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            weight_dict = dict(zip(classes, class_weights))
            print("\n[权重] 计算得到的类别权重:")
            for class_label, weight in weight_dict.items():
                print(f"   类别 {class_label}: {weight:.4f}")
            return weight_dict
        except Exception as e:
            handle_exception(e, "计算类别权重")
            return {}

    def create_weighted_sampler(self, y):
        try:
            class_weights = self.get_class_weights(y)
            if not class_weights:
                print("[警告] 无法创建加权采样器，使用默认采样")
                return None
            sample_weights = [class_weights.get(label, 1.0) for label in y]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            print(f"[成功] 创建加权采样器，样本数: {len(sample_weights)}")
            return sampler
        except Exception as e:
            handle_exception(e, "创建加权采样器")
            return None

    def evaluate_sampling_methods(self, X, y, methods=None):
        if methods is None:
            methods = IMBALANCE_METHODS
        results = {}
        original_counter, original_ratio = self.analyze_imbalance(y)
        print("\n[评估] 开始评估采样方法效果...")
        print(f"[原始] 失衡比例: {original_ratio:.2f}:1")
        for method in methods:
            try:
                print(f"\n--- 评估 {method.upper()} ---")
                X_resampled, y_resampled = self.apply_sampling(X, y, method)
                new_counter, new_ratio = self.analyze_imbalance(y_resampled)
                improvement = original_ratio / new_ratio if new_ratio > 0 else 0
                results[method] = {
                    'original_samples': len(y),
                    'resampled_samples': len(y_resampled),
                    'original_ratio': original_ratio,
                    'new_ratio': new_ratio,
                    'improvement_factor': improvement,
                    'distribution': new_counter,
                }
                print(f"[{method.upper()}] 原始样本: {len(y)} -> 重采样后: {len(y_resampled)}")
                print(f"[{method.upper()}] 失衡改善: {improvement:.2f}倍")
            except Exception as e:
                print(f"[错误] {method} 评估失败: {e}")
                results[method] = None
        return results

    def recommend_best_method(self, X, y, methods=None):
        print("\n[推荐] 开始评估并推荐最佳采样方法...")
        results = self.evaluate_sampling_methods(X, y, methods)
        best_method = None
        best_score = 0
        for method, result in results.items():
            if result is None:
                continue
            improvement_score = min(result['improvement_factor'], 10)
            efficiency_score = result['original_samples'] / result['resampled_samples']
            total_score = improvement_score * 0.7 + efficiency_score * 0.3
            print(f"[评分] {method.upper()}: 改善分数={improvement_score:.2f}, 效率分数={efficiency_score:.2f}, 总分={total_score:.2f}")
            if total_score > best_score:
                best_score = total_score
                best_method = method
        if best_method:
            print(f"\n[推荐] 最佳方法: {best_method.upper()} (总分: {best_score:.2f})")
            return best_method, results[best_method]
        print("\n[推荐] 无法找到合适的采样方法，建议使用原始数据")
        return None, None

    def apply_best_sampling(self, X, y):
        try:
            _, original_ratio = self.analyze_imbalance(y)
            if original_ratio <= 3.0:
                print(f"[分析] 失衡程度较轻 ({original_ratio:.2f}:1)，建议使用原始数据")
                return X, y, None
            best_method, _ = self.recommend_best_method(X, y)
            if best_method:
                X_resampled, y_resampled = self.apply_sampling(X, y, best_method)
                print(f"[应用] 使用 {best_method.upper()} 方法处理数据")
                return X_resampled, y_resampled, best_method
            print("[保持] 使用原始数据")
            return X, y, None
        except Exception as e:
            handle_exception(e, "自动选择采样方法")
            return X, y, None

    def get_sampling_statistics(self, y_original, y_resampled, method_name="采样"):
        try:
            print(f"\n[统计] {method_name}前后对比:")
            print("-" * 40)
            original_counter = Counter(y_original)
            print("原始数据:")
            for class_label, count in original_counter.items():
                print(f"  类别 {class_label}: {count} ({count/len(y_original)*100:.1f}%)")
            resampled_counter = Counter(y_resampled)
            print("重采样数据:")
            for class_label, count in resampled_counter.items():
                print(f"  类别 {class_label}: {count} ({count/len(y_resampled)*100:.1f}%)")
            print("变化:")
            print(f"  总样本数: {len(y_original)} -> {len(y_resampled)} ({len(y_resampled)/len(y_original):.2f}x)")
            for class_label in original_counter.keys():
                original_pct = original_counter[class_label] / len(y_original) * 100
                resampled_pct = resampled_counter.get(class_label, 0) / len(y_resampled) * 100
                change = resampled_pct - original_pct
                print(f"  类别 {class_label}: {original_pct:.1f}% -> {resampled_pct:.1f}% ({change:+.1f}%)")
            print("-" * 40)
        except Exception as e:
            handle_exception(e, "获取采样统计信息")


__all__ = ["ImbalanceHandler"]
