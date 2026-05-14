# -*- coding: utf-8 -*-
"""
实盘预测模块
用于预测当天股票走势，输出四分类概率和交易信号
"""

import os
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

from us_stock_predictor.config.core import (
    MODEL_PATHS, BACKTEST_CONFIG, MIN_HISTORY_LENGTH, LOG_DIR,
)
from us_stock_predictor.data.collector import USStockDataCollector
from us_stock_predictor.models.transformer_surge import StockSurgePredictor
from us_stock_predictor.utils.core import get_device, handle_exception, setup_environment, ensure_directories


class RealTimePredictor:
    """实盘预测器 - 基于训练好的模型进行当日预测"""

    def __init__(self, model_path=None):
        logging.info("[系统] 初始化实盘预测器...")
        setup_environment()
        self.device = get_device()
        self.collector = USStockDataCollector()
        self.feature_cols = self.collector.get_feature_columns()
        self.feature_dim = len(self.feature_cols)
        logging.info(f"[特征] 特征维度: {self.feature_dim}")
        self.model_path = model_path or MODEL_PATHS['best_model']
        self.model = self._load_model(self.model_path)
        self.stock_info = self.collector.get_stock_list_with_names()
        logging.info(f"[股票] 从配置文件读取到 {len(self.stock_info)} 只股票")

    def _load_model(self, model_path):
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            model = StockSurgePredictor(input_dim=self.feature_dim)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state = checkpoint.get('model_state_dict', checkpoint)
            incompat = model.load_state_dict(state, strict=False)
            if incompat.missing_keys or incompat.unexpected_keys:
                logging.warning(
                    f"[警告] 模型权重不完全匹配（架构版本不同），使用部分权重。 建议重新训练。"
                    f" missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}"
                )
            model.to(self.device)
            model.eval()
            logging.info(f"[成功] 模型加载成功: {model_path}")
            if 'epoch' in checkpoint:
                logging.info(f"[信息] 训练轮次: {checkpoint['epoch']}")
            if 'val_f1' in checkpoint:
                logging.info(f"[信息] 验证F1分数: {checkpoint['val_f1']:.4f}")
            return model
        except Exception as e:
            handle_exception(e, "加载模型")
            return None

    def get_latest_stock_data(self, symbol, days=100):
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            df = self.collector.get_stock_data(symbol, start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
            if df.empty:
                logging.warning(f"[警告] {symbol} 无法获取数据")
                return None
            df = self.collector.calculate_technical_indicators(df)
            if df is None or len(df) < MIN_HISTORY_LENGTH:
                logging.warning(f"[警告] {symbol} 数据不足")
                return None
            return df
        except Exception as e:
            handle_exception(e, f"获取{symbol}最新数据")
            return None

    def prepare_features_for_prediction(self, df):
        try:
            if len(df) < 20:
                return None
            last_20_days = df.tail(20).copy()
            missing_cols = [col for col in self.feature_cols if col not in last_20_days.columns]
            for col in missing_cols:
                last_20_days[col] = 0.0
            features_matrix = last_20_days[self.feature_cols].values
            features_matrix = np.nan_to_num(features_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            return torch.FloatTensor(features_matrix).unsqueeze(0).to(self.device)
        except Exception as e:
            handle_exception(e, "准备预测特征")
            return None

    def predict_single_stock(self, symbol):
        try:
            df = self.get_latest_stock_data(symbol)
            if df is None:
                return None
            features = self.prepare_features_for_prediction(df)
            if features is None or self.model is None:
                return None
            with torch.no_grad():
                class_logits, magnitude = self.model(features)
                class_probs = torch.softmax(class_logits, dim=1)[0]
            big_drop_prob = class_probs[0].item()
            small_drop_prob = class_probs[1].item()
            small_surge_prob = class_probs[2].item()
            big_surge_prob = class_probs[3].item()
            long_config = BACKTEST_CONFIG['four_class_strategy']
            short_config = BACKTEST_CONFIG['short_strategy']
            long_positive_signal = big_surge_prob * long_config['big_surge_weight'] + small_surge_prob * long_config['small_surge_weight']
            long_negative_signal = big_drop_prob * long_config['big_drop_weight'] + small_drop_prob * long_config['small_drop_weight']
            long_signal_strength = long_positive_signal - long_negative_signal
            short_positive_signal = big_drop_prob * short_config['big_drop_weight'] + small_drop_prob * short_config['small_drop_weight']
            short_negative_signal = big_surge_prob * short_config['big_surge_weight'] + small_surge_prob * short_config['small_surge_weight']
            short_signal_strength = short_positive_signal - short_negative_signal
            max_confidence = max(big_drop_prob, small_drop_prob, small_surge_prob, big_surge_prob)
            prediction_result = {
                'symbol': symbol,
                'name': self.stock_info.get(symbol, 'Unknown'),
                'latest_price': float(df['close'].iloc[-1]),
                'latest_date': df.index[-1].strftime('%Y-%m-%d'),
                'probabilities': {
                    'big_drop': big_drop_prob,
                    'small_drop': small_drop_prob,
                    'small_surge': small_surge_prob,
                    'big_surge': big_surge_prob,
                },
                'signals': {
                    'long_signal': long_signal_strength,
                    'short_signal': short_signal_strength,
                    'confidence': max_confidence,
                },
                'predicted_magnitude': magnitude.item() if magnitude is not None else 0.0,
            }
            prediction_result['advice'] = self._generate_trading_advice(prediction_result)
            return prediction_result
        except Exception as e:
            handle_exception(e, f"预测{symbol}")
            return None

    def _generate_trading_advice(self, result):
        long_signal = result['signals']['long_signal']
        short_signal = result['signals']['short_signal']
        confidence = result['signals']['confidence']
        long_config = BACKTEST_CONFIG['four_class_strategy']
        short_config = BACKTEST_CONFIG['short_strategy']
        can_long = long_signal > long_config['signal_threshold'] and confidence >= long_config['min_confidence']
        can_short = short_signal > short_config['signal_threshold'] and confidence >= short_config['min_confidence']
        if can_long and can_short:
            if long_signal >= short_signal:
                action = '做多'
                signal_strength = long_signal
            else:
                action = '做空'
                signal_strength = short_signal
        elif can_long:
            action = '做多'
            signal_strength = long_signal
        elif can_short:
            action = '做空'
            signal_strength = short_signal
        else:
            action = '观望'
            signal_strength = max(long_signal, short_signal)
        if action != '观望':
            if confidence >= 0.7:
                position = '高仓位'
            elif confidence >= 0.5:
                position = '中仓位'
            else:
                position = '低仓位'
        else:
            position = '空仓'
        return {
            'action': action,
            'signal_strength': signal_strength,
            'position': position,
            'confidence_level': '高' if confidence >= 0.7 else '中' if confidence >= 0.5 else '低',
        }

    def predict_all_stocks(self, output_file='predictions.csv'):
        logging.info(f"\n[开始] 预测 {len(self.stock_info)} 只股票...")
        logging.info(f"[时间] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("=" * 80)
        results = []
        for i, (symbol, name) in enumerate(self.stock_info.items(), 1):
            logging.info(f"\n[{i}/{len(self.stock_info)}] 正在预测: {symbol} - {name}")
            result = self.predict_single_stock(symbol)
            if result:
                results.append(result)
                probs = result['probabilities']
                signals = result['signals']
                advice = result['advice']
                logging.info(f"  最新价格: ${result['latest_price']:.2f}")
                logging.info(f"  四分类概率: 大跌{probs['big_drop']:.3f}, 小跌{probs['small_drop']:.3f}, 小涨{probs['small_surge']:.3f}, 大涨{probs['big_surge']:.3f}")
                logging.info(f"  信号强度: 做多{signals['long_signal']:.3f}, 做空{signals['short_signal']:.3f}")
                logging.info(f"  置信度: {signals['confidence']:.3f}")
                logging.info(f"  交易建议: {advice['action']} ({advice['position']}, {advice['confidence_level']}置信度)")
            else:
                logging.warning("  [失败] 无法预测")
        if results:
            self._save_predictions_to_csv(results, output_file)
            self._print_summary_statistics(results)
        return results

    def _save_predictions_to_csv(self, results, output_file):
        try:
            rows = []
            for r in results:
                rows.append({
                    '股票代码': r['symbol'],
                    '股票名称': r['name'],
                    '最新价格': r['latest_price'],
                    '最新日期': r['latest_date'],
                    '大跌概率': r['probabilities']['big_drop'],
                    '小跌概率': r['probabilities']['small_drop'],
                    '小涨概率': r['probabilities']['small_surge'],
                    '大涨概率': r['probabilities']['big_surge'],
                    '做多信号': r['signals']['long_signal'],
                    '做空信号': r['signals']['short_signal'],
                    '置信度': r['signals']['confidence'],
                    '交易建议': r['advice']['action'],
                    '仓位建议': r['advice']['position'],
                    '置信度等级': r['advice']['confidence_level'],
                })
            df = pd.DataFrame(rows).sort_values('做多信号', ascending=False)
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logging.info(f"\n[保存] 预测结果已保存到: {output_file}")
        except Exception as e:
            handle_exception(e, "保存预测结果")

    def _print_summary_statistics(self, results):
        logging.info("\n" + "=" * 80)
        logging.info("预测汇总统计")
        logging.info("=" * 80)
        advice_counts = {'做多': 0, '做空': 0, '观望': 0}
        high_confidence_stocks = []
        for r in results:
            action = r['advice']['action']
            advice_counts[action] += 1
            if r['signals']['confidence'] >= 0.7 and action != '观望':
                high_confidence_stocks.append({'symbol': r['symbol'], 'name': r['name'], 'action': action, 'confidence': r['signals']['confidence']})
        total = len(results)
        logging.info(f"总预测股票数: {total}")
        logging.info(f"做多建议: {advice_counts['做多']} ({advice_counts['做多']/total*100:.1f}%)")
        logging.info(f"做空建议: {advice_counts['做空']} ({advice_counts['做空']/total*100:.1f}%)")
        logging.info(f"观望建议: {advice_counts['观望']} ({advice_counts['观望']/total*100:.1f}%)")
        if high_confidence_stocks:
            logging.info("\n高置信度交易机会 (置信度 >= 0.7):")
            logging.info("-" * 60)
            for stock in sorted(high_confidence_stocks, key=lambda x: x['confidence'], reverse=True)[:10]:
                logging.info(f"{stock['symbol']:10s} {stock['name']:20s} {stock['action']:4s} 置信度:{stock['confidence']:.3f}")


def main():
    ensure_directories(LOG_DIR)
    today = datetime.now().strftime('%Y%m%d')
    log_file_path = os.path.join(LOG_DIR, f'real_time_log_{today}.txt')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    logging.info('美股实盘预测系统')
    logging.info('=' * 80)
    predictor = RealTimePredictor()
    output_file = f'predictions_{today}.csv'
    predictor.predict_all_stocks(output_file)
    logging.info(f"\n[完成] 预测完成！结果已保存到 {output_file}")


if __name__ == '__main__':
    main()
