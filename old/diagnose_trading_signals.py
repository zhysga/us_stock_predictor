# -*- coding: utf-8 -*-
"""
交易信号诊断脚本
分析模型预测分布和信号生成情况
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import BACKTEST_CONFIG, MODEL_PATHS
from data_collector import USStockDataCollector  
from models import StockSurgePredictor
from utils import get_device
import warnings
warnings.filterwarnings('ignore')

def analyze_model_predictions(symbol='105.AAPL', days=10):
    """分析模型预测分布"""
    print("=" * 60)
    print(f"模型预测诊断 - {symbol}")
    print("=" * 60)
    
    try:
        # 加载模型
        device = get_device()
        print(f"[设备] 使用设备: {device}")
        
        model_path = MODEL_PATHS['best_model']
        if not os.path.exists(model_path):
            print(f"[错误] 模型文件不存在: {model_path}")
            return None
            
        model = StockSurgePredictor()
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
        print(f"[模型] 已加载模型: {model_path}")
        
        # 获取数据
        collector = USStockDataCollector()
        
        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # 获取更多历史数据
        
        print(f"[数据] 获取 {start_date.strftime('%Y%m%d')} 到 {end_date.strftime('%Y%m%d')} 的数据")
        
        # 获取股票数据
        data = collector.get_stock_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d')
        )
        
        if data is None or len(data) < 30:
            print(f"[错误] 无法获取足够的数据，数据长度: {len(data) if data is not None else 0}")
            return None
            
        print(f"[数据] 获取到 {len(data)} 条数据")
        
        # 准备特征
        features = collector.prepare_features_for_prediction(data)
        if features is None:
            print("[错误] 特征准备失败")
            return None
            
        feature_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        print(f"[特征] 特征维度: {feature_tensor.shape}")
        
        # 获取最近几天的预测
        predictions = []
        confidence_scores = []
        signals = []
        
        seq_len = feature_tensor.shape[1]
        for i in range(max(0, seq_len - days), seq_len):
            if i + 20 <= seq_len:  # 确保有足够的序列长度
                # 获取序列
                seq = feature_tensor[:, i:i+20, :]
                
                with torch.no_grad():
                    output = model(seq)
                    probabilities = torch.softmax(output, dim=1)
                    
                    # 提取概率
                    big_drop_prob = probabilities[0, 0].item()
                    small_drop_prob = probabilities[0, 1].item()
                    small_surge_prob = probabilities[0, 2].item()
                    big_surge_prob = probabilities[0, 3].item()
                    
                    # 计算置信度
                    confidence = torch.max(probabilities).item()
                    
                    # 计算交易信号
                    long_signal, short_signal = calculate_trading_signals(
                        big_drop_prob, small_drop_prob, small_surge_prob, big_surge_prob
                    )
                    
                    predictions.append({
                        'index': i,
                        'big_drop_prob': big_drop_prob,
                        'small_drop_prob': small_drop_prob,
                        'small_surge_prob': small_surge_prob,
                        'big_surge_prob': big_surge_prob,
                        'confidence': confidence,
                        'long_signal': long_signal,
                        'short_signal': short_signal
                    })
        
        # 分析预测结果
        analyze_prediction_distribution(predictions)
        analyze_trading_potential(predictions)
        
        return predictions
        
    except Exception as e:
        print(f"[错误] 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_trading_signals(big_drop_prob, small_drop_prob, small_surge_prob, big_surge_prob):
    """计算交易信号"""
    config = BACKTEST_CONFIG['four_class_strategy']
    
    # 做多信号
    long_signal = (
        big_surge_prob * config['big_surge_weight'] + 
        small_surge_prob * config['small_surge_weight']
    )
    
    # 做空信号  
    short_signal = (
        big_drop_prob * config['big_drop_weight'] +
        small_drop_prob * config['small_drop_weight']
    )
    
    return long_signal, short_signal

def analyze_prediction_distribution(predictions):
    """分析预测分布"""
    if not predictions:
        print("[错误] 没有预测数据")
        return
        
    print("\n" + "=" * 40)
    print("预测分布分析")
    print("=" * 40)
    
    # 统计各类别概率
    big_drop_probs = [p['big_drop_prob'] for p in predictions]
    small_drop_probs = [p['small_drop_prob'] for p in predictions]
    small_surge_probs = [p['small_surge_prob'] for p in predictions]
    big_surge_probs = [p['big_surge_prob'] for p in predictions]
    confidences = [p['confidence'] for p in predictions]
    long_signals = [p['long_signal'] for p in predictions]
    short_signals = [p['short_signal'] for p in predictions]
    
    print(f"样本数量: {len(predictions)}")
    print()
    
    print("概率分布统计:")
    print(f"  大跌概率: 平均={np.mean(big_drop_probs):.3f}, 最大={np.max(big_drop_probs):.3f}")
    print(f"  小跌概率: 平均={np.mean(small_drop_probs):.3f}, 最大={np.max(small_drop_probs):.3f}")
    print(f"  小涨概率: 平均={np.mean(small_surge_probs):.3f}, 最大={np.max(small_surge_probs):.3f}")
    print(f"  大涨概率: 平均={np.mean(big_surge_probs):.3f}, 最大={np.max(big_surge_probs):.3f}")
    print()
    
    print("信号分布统计:")
    print(f"  置信度: 平均={np.mean(confidences):.3f}, 最大={np.max(confidences):.3f}")
    print(f"  做多信号: 平均={np.mean(long_signals):.3f}, 最大={np.max(long_signals):.3f}")
    print(f"  做空信号: 平均={np.mean(short_signals):.3f}, 最大={np.max(short_signals):.3f}")
    
def analyze_trading_potential(predictions):
    """分析交易潜力"""
    if not predictions:
        return
        
    print("\n" + "=" * 40)
    print("交易潜力分析")
    print("=" * 40)
    
    config = BACKTEST_CONFIG['four_class_strategy']
    signal_threshold = config['signal_threshold']
    min_confidence = config['min_confidence']
    
    print(f"当前交易阈值:")
    print(f"  信号阈值: {signal_threshold}")
    print(f"  置信度阈值: {min_confidence}")
    print()
    
    # 统计满足条件的交易机会
    long_opportunities = 0
    short_opportunities = 0
    
    for p in predictions:
        # 做多机会
        if p['long_signal'] > signal_threshold and p['confidence'] >= min_confidence:
            long_opportunities += 1
            
        # 做空机会
        if p['short_signal'] > signal_threshold and p['confidence'] >= min_confidence:
            short_opportunities += 1
    
    print(f"当前阈值下的交易机会:")
    print(f"  做多机会: {long_opportunities} / {len(predictions)} ({long_opportunities/len(predictions)*100:.1f}%)")
    print(f"  做空机会: {short_opportunities} / {len(predictions)} ({short_opportunities/len(predictions)*100:.1f}%)")
    print(f"  总交易机会: {long_opportunities + short_opportunities}")
    
    # 测试不同阈值下的交易机会
    print(f"\n不同阈值下的交易机会:")
    test_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    for threshold in test_thresholds:
        long_count = sum(1 for p in predictions if p['long_signal'] > threshold and p['confidence'] >= min_confidence)
        short_count = sum(1 for p in predictions if p['short_signal'] > threshold and p['confidence'] >= min_confidence)
        total = long_count + short_count
        print(f"  阈值{threshold:.2f}: {total}笔交易 (多{long_count}, 空{short_count})")
    
    # 建议最佳阈值
    suggest_optimal_threshold(predictions)

def suggest_optimal_threshold(predictions):
    """建议最佳阈值"""
    print(f"\n" + "=" * 40)
    print("最佳阈值建议")
    print("=" * 40)
    
    min_confidence = BACKTEST_CONFIG['four_class_strategy']['min_confidence']
    
    # 寻找能产生适量交易的阈值 (目标: 每10天1-3笔交易)
    target_trade_rate = 0.2  # 20%的天数有交易
    
    best_threshold = None
    best_trade_count = 0
    
    for threshold in np.arange(0.01, 0.5, 0.01):
        trade_count = 0
        for p in predictions:
            if ((p['long_signal'] > threshold or p['short_signal'] > threshold) and 
                p['confidence'] >= min_confidence):
                trade_count += 1
        
        trade_rate = trade_count / len(predictions)
        
        if abs(trade_rate - target_trade_rate) < abs(best_trade_count / len(predictions) - target_trade_rate):
            best_threshold = threshold
            best_trade_count = trade_count
    
    if best_threshold:
        print(f"建议信号阈值: {best_threshold:.3f}")
        print(f"预期交易次数: {best_trade_count} / {len(predictions)} ({best_trade_count/len(predictions)*100:.1f}%)")
        
        # 检查置信度分布
        high_conf_count = sum(1 for p in predictions if p['confidence'] >= min_confidence)
        print(f"满足置信度要求的样本: {high_conf_count} / {len(predictions)} ({high_conf_count/len(predictions)*100:.1f}%)")
        
        if high_conf_count < len(predictions) * 0.3:
            suggested_conf = np.percentile([p['confidence'] for p in predictions], 30)
            print(f"建议降低置信度阈值至: {suggested_conf:.3f}")

if __name__ == "__main__":
    # 分析几个主要股票
    test_symbols = ['105.AAPL', '105.MSFT', '105.NVDA']
    
    for symbol in test_symbols:
        print(f"\n分析 {symbol}...")
        predictions = analyze_model_predictions(symbol, days=15)
        if predictions:
            print(f"✅ {symbol} 分析完成")
        else:
            print(f"❌ {symbol} 分析失败")
        print("-" * 60)