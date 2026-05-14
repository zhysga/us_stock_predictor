# -*- coding: utf-8 -*-
"""数据收集器 — 代理到 src/us_stock_predictor/data/collector.py"""
import os, sys
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from us_stock_predictor.data.collector import *
from us_stock_predictor.data.collector import __all__
