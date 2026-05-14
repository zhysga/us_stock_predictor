#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多空交易策略测试脚本
演示基于AI预测的对称多空交易功能
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from us_stock_predictor.backtest.engine import run_long_short_backtest, compare_trading_strategies
from us_stock_predictor.config.core import BACKTEST_CONFIG

def test_long_short_strategy():
    """测试多空交易策略"""
    print("=" * 80)
    print("多空交易策略测试")
    print("=" * 80)
    
    # 测试股票
    test_symbol = '105.AAPL'
    start_date = "20240901"  # 使用更长的回测周期
    
    print(f"\n[测试配置]")
    print(f"测试股票: {test_symbol}")
    print(f"开始日期: {start_date}")
    print(f"初始资金: ${BACKTEST_CONFIG['initial_cash']:,.0f}")
    print(f"做空功能: {'启用' if BACKTEST_CONFIG['enable_short_selling'] else '禁用'}")
    
    # 1. 测试纯做多策略
    print(f"\n{'='*60}")
    print("1. 测试纯做多策略")
    print(f"{'='*60}")
    long_only_result = run_long_short_backtest(test_symbol, start_date, trading_mode='long_only')
    
    # 2. 测试纯做空策略
    print(f"\n{'='*60}")
    print("2. 测试纯做空策略")
    print(f"{'='*60}")
    short_only_result = run_long_short_backtest(test_symbol, start_date, trading_mode='short_only')
    
    # 3. 测试多空结合策略
    print(f"\n{'='*60}")
    print("3. 测试多空结合策略")
    print(f"{'='*60}")
    long_short_result = run_long_short_backtest(test_symbol, start_date, trading_mode='long_short')
    
    # 4. 策略比较
    print(f"\n{'='*60}")
    print("4. 策略比较分析")
    print(f"{'='*60}")
    comparison_results = compare_trading_strategies(test_symbol, start_date)
    
    return {
        'long_only': long_only_result,
        'short_only': short_only_result,
        'long_short': long_short_result,
        'comparison': comparison_results
    }

def analyze_results(results):
    """分析测试结果"""
    print(f"\n{'='*80}")
    print("详细结果分析")
    print(f"{'='*80}")
    
    strategies = ['long_only', 'short_only', 'long_short']
    
    for strategy in strategies:
        if strategy in results and results[strategy]:
            result = results[strategy]
            print(f"\n[{strategy.upper()} 策略分析]")
            print(f"  最终收益率: {result['total_return']:+.2f}%")
            print(f"  多头交易次数: {result['long_trades']}")
            print(f"  空头交易次数: {result['short_trades']}")
            print(f"  总交易次数: {result['total_trades']}")
            print(f"  胜率: {result['win_rate']:.1f}%")
            
            # 计算风险调整收益
            if result['total_trades'] > 0:
                avg_return_per_trade = result['total_return'] / result['total_trades']
                print(f"  平均每笔收益: {avg_return_per_trade:+.3f}%")
            
            # 判断策略有效性
            if result['total_return'] > 0:
                print(f"  策略评价: ✓ 盈利策略")
            else:
                print(f"  策略评价: ✗ 亏损策略")

def main():
    """主函数"""
    try:
        print("开始多空交易策略测试...")
        
        # 运行测试
        results = test_long_short_strategy()
        
        # 分析结果
        analyze_results(results)
        
        print(f"\n{'='*80}")
        print("多空交易策略测试完成！")
        print(f"{'='*80}")
        
        return results
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main() 