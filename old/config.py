# -*- coding: utf-8 -*-
"""配置中心 — 代理到 src/us_stock_predictor/config/core.py"""
import os, sys
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from us_stock_predictor.config.core import *
from us_stock_predictor.config.core import __all__
