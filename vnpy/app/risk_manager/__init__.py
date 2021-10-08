# -*- coding:utf-8 -*-
from pathlib import Path
from vnpy.trader.app import BaseApp
from .engine import RiskManagerEngine, APP_NAME
import sys

import vnpy_riskmanager


sys.modules[__name__] = vnpy_riskmanager
