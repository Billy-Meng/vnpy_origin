from math import floor, ceil
from decimal import Decimal
from unicodedata import decimal
from vnpy.app.spread_trading import (
    SpreadStrategyTemplate,
    SpreadAlgoTemplate,
    SpreadData,
    OrderData,
    TradeData
)
from vnpy.trader.utility import BarGenerator
from vnpy.trader.object import BarData


class GridSpreadStrategy(SpreadStrategyTemplate):
    author = "Billy"

    grid_start = 0.0
    grid_price = 5.0
    grid_volume = 1.0
    pay_up = 10

    price_change = 0.0
    current_grid = 0.0
    max_target = 0.0
    min_target = 0.0

    parameters = [
        "grid_start", "grid_price", "grid_volume", "pay_up"
    ]
    variables = [
        "price_change", "current_grid", "max_target", "min_target"
    ]

    def __init__(
        self,
        strategy_engine,
        strategy_name: str,
        spread: SpreadData,
        setting: dict
    ):
        super().__init__(strategy_engine, strategy_name, spread, setting)

        self.bg = BarGenerator(self.on_bar)

    def on_init(self):
        self.write_log("策略初始化")

    def on_start(self):
        self.write_log("策略启动")

    def on_stop(self):
        self.write_log("策略停止")

    def on_spread_data(self):
        self.spread_pos = self.get_spread_pos()

        tick = self.spread.to_tick()
        self.bg.update_tick(tick)

        # mid_price = (self.spread.bid_price + self.spread.ask_price) / 2
        # self.price_change = mid_price - self.grid_start                                  # 计算价格相对初始位置的变动

    def on_bar(self, bar: BarData):
        self.stop_all_algos()

        # 计算当前网格位置
        self.price_change = Decimal(str(bar.close_price)) - self.grid_start 
        self.current_grid = self.price_change / Decimal(str(self.grid_price))       # 计算网格水平

        self.max_target = ceil(-self.current_grid) * self.grid_volume               # 计算当前最大持仓
        self.min_target = floor(-self.current_grid) * self.grid_volume              # 计算当前最小持仓

        # 做多，检查最小持仓和当前持仓的差值
        long_volume = self.min_target - self.spread_pos
        if long_volume > 0:
            long_price = bar.close_price + self.pay_up
            self.start_long_algo(long_price, long_volume, 5, 5)

        # 做空，检查最大持仓和当前持仓的差值
        short_volume = self.max_target - self.spread_pos
        if short_volume < 0:
            short_price = bar.close_price - self.pay_up
            self.start_short_algo(short_price, abs(short_volume), 5, 5)

        self.put_event()
    
    def on_spread_pos(self):
        self.spread_pos = self.get_spread_pos()
        self.put_event()
    
    def on_spread_algo(self, algo: SpreadAlgoTemplate):
        pass

    def on_order(self, order: OrderData):
        pass

    def on_trade(self, trade: TradeData):
        self.put_event()
