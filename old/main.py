# -*- coding: utf-8 -*-
"""主程序入口 — 代理到 src/us_stock_predictor/cli/main.py"""
import os, sys
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from us_stock_predictor.cli.main import main_entry
from us_stock_predictor.utils.core import main_with_proper_encoding

if __name__ == "__main__":
    main_with_proper_encoding(main_entry)
