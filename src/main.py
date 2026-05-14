# -*- coding: utf-8 -*-
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from us_stock_predictor.cli.main import main_entry
from us_stock_predictor.utils.core import main_with_proper_encoding

if __name__ == "__main__":
    main_with_proper_encoding(main_entry)
