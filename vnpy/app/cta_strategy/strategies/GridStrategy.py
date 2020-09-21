from math import floor, ceil
from decimal import Decimal
from vnpy.app.cta_strategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)


class GridStrategy(CtaTemplate):
    author = "Billy"

    grid_start = 102
    grid_price = 0.2
    grid_volume = 1
    pay_up = 10

    price_change = 0
    current_grid = 0
    max_target = 0
    min_target = 0

    parameters = [
        "grid_start", "grid_price", "grid_volume", "pay_up"
    ]
    variables = [
        "price_change", "current_grid", "max_target", "min_target"
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.bg = BarGenerator(self.on_bar)

    def on_init(self):
        self.write_log("策略初始化")

    def on_start(self):
        self.write_log("策略启动")

    def on_stop(self):
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        self.tick = tick
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        self.cancel_all()

        # 计算交易信号
        self.price_change = Decimal(str(bar.close_price)) - self.grid_start         # 计算价格相对初始位置的变动
        self.current_grid = self.price_change / Decimal(str(self.grid_price))       # 计算网格水平

        self.max_target = ceil(-self.current_grid) * self.grid_volume               # 计算当前最大持仓
        self.min_target = floor(-self.current_grid) * self.grid_volume              # 计算当前最小持仓

        # 做多，检查最小持仓和当前持仓的差值
        long_volume = self.min_target - self.pos
        if long_volume > 0:
            long_price = bar.close_price + self.pay_up
            if self.pos >= 0:
                self.buy(long_price, long_volume)
            else:
                self.cover(long_price, long_volume)

        # 做空，检查最大持仓和当前持仓的差值
        short_volume = self.max_target - self.pos
        if short_volume < 0:
            short_price = bar.close_price - self.pay_up
            if self.pos <= 0:
                self.short(short_price, abs(short_volume))
            else:
                self.sell(short_price, abs(short_volume))


        self.put_event()

    def on_order(self, order: OrderData):
        pass

    def on_trade(self, trade: TradeData):
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        pass
