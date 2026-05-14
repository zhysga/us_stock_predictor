# -*- coding: utf-8 -*-
"""数据集处理 — 代理到 src/us_stock_predictor/datasets/timeseries.py"""
import os, sys
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from us_stock_predictor.datasets.timeseries import *
from us_stock_predictor.datasets.timeseries import __all__
