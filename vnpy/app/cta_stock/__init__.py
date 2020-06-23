from pathlib import Path

from vnpy.trader.app import BaseApp
from .base import APP_NAME, StopOrder

from .engine import CtaEngine

from .template import (
    Exchange,
    Direction,
    Offset,
    Status,
    Color,
    Interval,
    OrderType,
    TickData,
    BarData,
    TradeData,
    OrderData,
    CtaPolicy,
    StockPolicy,
    CtaTemplate,  CtaStockTemplate)  # noqa

from vnpy.trader.utility import BarGenerator, ArrayManager  # noqa


class CtaStockApp(BaseApp):
    """"""
    app_name = APP_NAME
    app_module = __module__
    app_path = Path(__file__).parent
    display_name = "股票CTA策略"
    engine_class = CtaEngine
    widget_name = "CtaManager"
    icon_name = "cta.ico"
