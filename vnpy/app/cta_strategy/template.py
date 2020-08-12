# -*- coding:utf-8 -*-
from abc import ABC
from copy import copy, deepcopy
from typing import Any, Callable, Union
from datetime import datetime, time, timedelta
from threading import Thread

import pandas as pd

from vnpy.trader.constant import Interval, Direction, Offset, RateType
from vnpy.trader.object import BarData, TickData, OrderData, TradeData, AccountData, ContractData
from vnpy.trader.utility import virtual, get_file_path, get_folder_path, save_json, load_json, send_dingding, send_weixin, popup_warning

from .base import StopOrder, EngineType

class CtaTemplate(ABC):
    """"""

    author = ""
    parameters = []
    variables = []

    def __init__(
        self,
        cta_engine: Any,
        strategy_name: str,
        vt_symbol: str,
        setting: dict,
    ):
        """"""
        self.cta_engine = cta_engine
        self.strategy_name = strategy_name
        self.vt_symbol = vt_symbol
        self.symbol = self.vt_symbol.split(".")[0]

        self.inited = False
        self.trading = False
        self.pos = 0

        # Copy a new variables list here to avoid duplicate insert when multiple
        # strategy instances are created with the same strategy class.
        self.variables = deepcopy(self.variables)
        self.variables.insert(0, "inited")
        self.variables.insert(1, "trading")
        self.variables.insert(2, "pos")
        self.variables.insert(3, "cost")
        self.variables.insert(4, "track_highest")
        self.variables.insert(5, "track_lowest")
        self.variables.insert(6, "track_minute")
        self.variables.insert(7, "entry_minute")
        self.variables.insert(8, "exit_minute")

        self.engine_type = self.get_engine_type()       # 初始化时获取策略引擎类型

        self.buy_orderids   = []
        self.short_orderids = []
        self.sell_orderids  = []
        self.cover_orderids = []

        self.cost = 0                    # 记录开仓均价
        self.track_highest = 0           # 记录多单开仓后的最高点，空单入场后的回调高点
        self.track_lowest  = 0           # 记录空单开仓后的最低点，多单入场后的回调低点
        self.track_minute = 0            # 记录新高低点保持多少分钟
        self.entry_minute  = 0           # 统计开仓后历时多少分钟
        self.exit_minute   = 0           # 统计平仓后历时多少分钟
        self.floating_point  = 0         # 记录开仓后的浮动盈亏点数
        self.floating_pnl = 0            # 记录开仓后的浮动盈亏金额

        self.position = None             # 实盘模式记录持仓信息
        self.account = None              # 实盘模式记录账号信息
        self.long_cost = 0               # 实盘模式记录最新多头持仓均价
        self.short_cost = 0              # 实盘模式记录最新空头持仓均价
        self.show_account_data = True    # 实盘模式每次启动策略时显示资金状况
        self.balance = 0
        self.lock = False                # 实盘模式记录发出委托是否锁仓模式

        self.signal: float = 0           # 实盘和回测记录当笔交易的信号：1 buy, 2 short, 3 sell, 4 cover
        self.signal_list = []            # 实盘和回测缓存每笔交易的信号
        self.trade_data_dict = {}        # 实盘模式缓存成交数据

        self.symbol_strategy = f"【{self.symbol}】{self.__class__.__name__}"

        # 委托状态控制初始化     
        self.chase_long_trigger = False   
        self.chase_sell_trigger = False     
        self.chase_short_trigger = False   
        self.chase_cover_trigger = False    
        self.long_trade_volume = 0    
        self.short_trade_volume = 0    
        self.sell_trade_volume = 0       
        self.cover_trade_volume = 0    
        self.chase_interval = 10         # 拆单间隔:秒

        self.trade_number = 0            # 回测时缓存当笔交易的序号
        self.trade_volume = 0            # 回测时缓存当笔交易的累计成交手数
        self.trade_net_volume = 0        # 回测时缓存当笔交易的净持仓量
        self.trade_pnl = 0               # 实盘和回测时缓存当笔交易的盈亏金额
        self.trade_commission = 0        # 实盘和回测时缓存当笔交易的手续费
        self.trade_slippage = 0          # 回测时缓存当笔交易的滑点费
        self.net_pnl = 0                 # 实盘和回测时缓存当笔交易的净盈亏金额
        self.pnl_point = 0               # 实盘和回测时缓存当笔交易的净盈亏点数

        self.trade_number_list = []      # 回测时缓存每一次成交的序号
        self.trade_volume_list = []      # 回测时缓存每笔交易的累计成交手数
        self.trade_net_volume_list = []  # 回测时缓存每一次成交的净持仓量状态
        self.trade_type_list = []        # 回测时缓存每笔交易的多空类型
        self.trade_pnl_list = []         # 回测时缓存每笔交易的盈亏金额
        self.commission_list = []        # 回测时缓存每笔交易的手续费
        self.slippage_list = []          # 回测时缓存每笔交易的滑点费
        self.net_pnl_list = []           # 回测时缓存每笔交易的净盈亏金额
        self.pnl_point_list = []         # 回测时缓存每笔交易的净盈亏点数
        self.balance_list = []           # 回测时缓存每笔交易结束后的Balance
        self.trade_duration_list = []    # 回测时缓存每笔交易的有效持仓时间（分钟）
        self.trade_date_list = []        # 回测时缓存每笔交易的交易日期
        self.trade_start_time_list = []  # 回测时缓存每笔交易的起始时间
        self.trade_end_time_list = []    # 回测时缓存每笔交易的结束时间
        self.record_start_switch = True  # 回测时每笔交易起始时间记录开关

        self.update_setting(setting)

    def update_setting(self, setting: dict):
        """
        Update strategy parameter wtih value in setting dict.
        """
        for name in self.parameters:
            if name in setting:
                setattr(self, name, setting[name])

    @classmethod
    def get_class_parameters(cls):
        """
        Get default parameters dict of strategy class.
        """
        class_parameters = {}
        for name in cls.parameters:
            class_parameters[name] = getattr(cls, name)
        return class_parameters

    def get_parameters(self):
        """
        Get strategy parameters dict.
        """
        strategy_parameters = {}
        for name in self.parameters:
            strategy_parameters[name] = getattr(self, name)
        return strategy_parameters

    def get_variables(self):
        """
        Get strategy variables dict.
        """
        strategy_variables = {}
        for name in self.variables:
            strategy_variables[name] = getattr(self, name)
        return strategy_variables

    def get_data(self):
        """
        Get strategy data.
        """
        strategy_data = {
            "strategy_name": self.strategy_name,
            "vt_symbol": self.vt_symbol,
            "class_name": self.__class__.__name__,
            "author": self.author,
            "parameters": self.get_parameters(),
            "variables": self.get_variables(),
        }
        return strategy_data

    @virtual
    def on_init(self):
        """
        Callback when strategy is inited.
        """
        pass

    @virtual
    def on_start(self):
        """
        Callback when strategy is started.
        """
        pass

    @virtual
    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        pass

    @virtual
    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        pass

    @virtual
    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        pass
    
    @virtual
    def on_second_bar(self, bar: BarData):
        """
        Callback of new second bar data update.
        """
        pass

    @virtual
    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        pass

    @virtual
    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        pass

    @virtual
    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass

    def on_position(self, holding: "PositionHolding"):
        """
        Callback of position update.
        """
        if holding.long_price != self.long_cost or holding.short_price != self.short_cost:
            self.write_log(f"【{self.symbol}持仓信息】多头持仓数量：{holding.long_pos}，多头持仓均价：{holding.long_price}；空头持仓数量：{holding.short_pos}，空头持仓均价：{holding.short_price}")
            
        self.position = holding
        self.long_cost = holding.long_price
        self.short_cost = holding.short_price
        

    def on_account(self, account: AccountData):
        """
        Callback of account update.
        """
        if self.show_account_data:
            self.write_log(f"【账户信息】账号：{account.accountid}，账户净值：{account.balance}，可用资金：{account.available}，资金使用率：{round(account.percent, 1)}%")
            self.show_account_data = False

        self.account = account
        self.balance = account.balance


    def buy(self, price: float, volume: float, stop: bool = False, lock: bool = False, market: bool = False):
        """
        Send buy order to open a long position.
        """
        vt_orderid = self.send_order(Direction.LONG, Offset.OPEN, price, abs(volume), stop, lock, market)
        self.buy_orderids.extend(vt_orderid)
        return vt_orderid

    def sell(self, price: float, volume: float, stop: bool = False, lock: bool = False, market: bool = False):
        """
        Send sell order to close a long position.
        """
        vt_orderid = self.send_order(Direction.SHORT, Offset.CLOSE, price, abs(volume), stop, lock, market)
        self.sell_orderids.extend(vt_orderid)
        return vt_orderid

    def short(self, price: float, volume: float, stop: bool = False, lock: bool = False, market: bool = False):
        """
        Send short order to open as short position.
        """
        vt_orderid = self.send_order(Direction.SHORT, Offset.OPEN, price, abs(volume), stop, lock, market)
        self.short_orderids.extend(vt_orderid)
        return vt_orderid

    def cover(self, price: float, volume: float, stop: bool = False, lock: bool = False, market: bool = False):
        """
        Send cover order to close a short position.
        """
        vt_orderid = self.send_order(Direction.LONG, Offset.CLOSE, price, abs(volume), stop, lock, market)
        self.cover_orderids.extend(vt_orderid)
        return vt_orderid

    def send_order(
        self,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        stop: bool = False,
        lock: bool = False,
        market: bool = False
    ):
        """
        Send a new order.
        """
        self.lock = lock    # 是否锁仓

        if self.trading:
            vt_orderids = self.cta_engine.send_order(
                self, direction, offset, price, volume, stop, lock, market
            )
            return vt_orderids
        else:
            return []

    def cancel_order(self, vt_orderid: str):
        """
        Cancel an existing order.
        """
        if self.trading:
            self.cta_engine.cancel_order(self, vt_orderid)

    def cancel_all(self):
        """
        Cancel all orders sent by strategy.
        """
        if self.trading:
            self.cta_engine.cancel_all(self)

    def write_log(self, msg: str):
        """
        Write a log message.
        """
        self.cta_engine.write_log(msg, self)

    def get_engine_type(self):
        """
        Return whether the cta_engine is backtesting or live trading.
        """
        return self.cta_engine.get_engine_type()

    def get_pricetick(self):
        """
        Return pricetick data of trading contract.
        """
        return self.cta_engine.get_pricetick(self)

    def load_bar(
        self,
        days: int,
        interval: Interval = Interval.MINUTE,
        frequency: int = 60,
        callback: Callable = None,
        use_database: bool = False
    ):
        """
        Load historical bar data for initializing strategy.
        """
        if not callback:
            callback = self.on_bar

        self.cta_engine.load_bar(
            vt_symbol = self.vt_symbol,
            days = days,
            interval = interval,
            frequency = frequency,
            callback = callback,
            use_database = use_database
        )

    def load_tick(self, days: int, use_database: bool = False):
        """
        Load historical tick data for initializing strategy.
        """
        self.cta_engine.load_tick(self.vt_symbol, days, self.on_tick, use_database)

    def put_event(self):
        """
        Put an strategy data event for ui update.
        """
        if self.inited:
            self.cta_engine.put_strategy_event(self)

    def send_email(self, msg):
        """
        Send email to default receiver.
        """
        if self.inited:
            self.cta_engine.send_email(msg, self)

    def sync_data(self):
        """
        Sync strategy variables value into disk storage.
        """
        if self.trading:
            self.cta_engine.sync_strategy_data(self)


    # ========================================================================================================================================================================================= #
    # 自定义增强功能

    def init_strategy_data(self):
        self.write_log("策略初始化")
        self.get_contract_data()               # 策略初始化时获取合约信息，动态赋值给 contract_data, gateway_name, symbol, pricetick, size
        self.product_trade_time()              # 策略初始化时获取品种的交易时间信息字典，动态赋值给 trade_time

    def popup(self, msg:str):
        """ 弹窗消息通知 """
        if self.inited and self.engine_type == EngineType.LIVE:
            thread_popup = Thread(target=popup_warning, name="popup_warning", args=(self.symbol_strategy + "\n" + msg,) )
            thread_popup.start()

    def dingding(self, msg:str):
        """ 钉钉机器人消息通知 """
        if self.inited and self.engine_type == EngineType.LIVE:
            thread_dingding = Thread(target=send_dingding, name="dingding", args=(self.symbol_strategy + "\n" + msg,))
            thread_dingding.start()

    def weixin(self, msg:str):
        """ 通过FTQQ发送微信消息 http://sc.ftqq.com/3.version """
        if self.inited and self.engine_type == EngineType.LIVE:
            thread_weixin = Thread(target=send_weixin, name="weixin", args=(self.symbol_strategy + "\n" + msg,))
            thread_weixin.start()

    def get_contract_data(self) -> ContractData:
        """
        获取合约信息。策略初始化时调用，动态赋值contract_data, gateway_name, symbol, pricetick, size, margin_rate
        """
        if self.engine_type == EngineType.LIVE:
            self.contract_data = self.cta_engine.main_engine.get_contract(self.vt_symbol)

            if self.contract_data:                
                self.gateway_name = self.contract_data.gateway_name      # 接口名称
                self.symbol = self.contract_data.symbol                  # 合约代码
                self.pricetick = self.contract_data.pricetick            # 最小变动价位
                self.size = self.contract_data.size                      # 合约乘数
                self.margin_ratio = self.contract_data.margin_ratio      # 保证金比率
                self.commission = self.contract_data.open_commission + self.contract_data.close_commission                          # 手续费
                self.commission_ratio = self.contract_data.open_commission_ratio + self.contract_data.close_commission_ratio        # 手续费率
                self.close_commission_today = self.contract_data.close_commission_today                                             # 平今手续费
                self.close_commission_today_ratio = self.contract_data.close_commission_today_ratio                                 # 平今手续费率
                self.write_log((f"合约信息查询成功：Symbol【 {self.symbol} 】，Pricetick【 {self.pricetick} 】，Size【 {self.size} 】，" +
                                f"保证金率【 {self.margin_ratio} 】，手续费【 {self.commission} 】，手续费率【 {self.commission_ratio} 】，" +
                                f"平今手续费【 {self.close_commission_today} 】，平今手续费率【 {self.close_commission_today_ratio} 】"))

        else:
            # 回测模式记录合约信息
            self.gateway_name = self.cta_engine.gateway_name      # 接口名称，回测的gateway_name为"BACKTESTING"
            self.vt_symbol = self.cta_engine.vt_symbol            # 本地合约代码
            self.symbol = self.cta_engine.symbol                  # 合约代码
            self.pricetick = self.cta_engine.pricetick            # 最小变动价位
            self.size = self.cta_engine.size                      # 合约乘数
            self.rate_type = self.cta_engine.rate_type            # 手续费模式
            self.rate = self.cta_engine.rate                      # 【单向】手续费/手续费率
            self.slippage = self.cta_engine.slippage              # 【单向】滑点数
            self.capital = self.cta_engine.capital                # 初始资金
            self.balance = self.cta_engine.capital                # 初始化Balance

    def product_trade_time(self) -> dict:
        """获取品种的交易时间信息，包括字段：symbol, exchange, name, am_start, rest_start, rest_end, am_end, pm_start, pm_end, night_trade, night_start, night_end"""

        filepath = get_file_path("期货品种交易时间.xlsx")

        if filepath.exists():

            for count, word in enumerate(self.vt_symbol):
                if word.isdigit():
                    break
            product = self.vt_symbol[:count].upper()

            df = pd.read_excel(filepath)
            df["symbol"] = df["symbol"].apply(lambda x: x.upper())
            df = df.set_index("symbol")
            
            try:
                self.trade_time = df.loc[product].to_dict()
                if self.trade_time["night_trade"] == True:
                    night_time = f'夜盘【 {self.trade_time["night_start"]} ~ {self.trade_time["night_end"]} 】'
                else:
                    night_time = "无夜盘。"

                self.write_log(f'品种交易时间：日盘【 {self.trade_time["am_start"]} ~ {self.trade_time["pm_end"]} 】，{night_time}')

            except KeyError:
                self.write_log("找不到该品种交易时间")

        else:
            self.write_log("找不到该品种交易时间")

    def tick_chase_trade(self, tick: TickData):
        """
        该方法用于在Tick级别进行追击成交，需在策略实例的on_tick函数中调用。委托挂撤单管理逻辑如下：
        1、调用CTA策略模板新增的get_position_detail函数，通过engine获取活动委托字典active_orders。注意的是，该字典只缓存未成交或者部分成交的委托，其中key是字符串格式的vt_orderid，value对应OrderData对象；
        2、engine取到的活动委托为空，表示委托已完成，即order_finished=True；否则，表示委托还未完成，即order_finished=False，需要进行精细度管理；
        3、精细管理的第一步是先处理最老的活动委托，先获取委托号vt_orderid和OrderData对象，然后对OrderData对象的开平仓属性（即offst）判断是进行开仓追单还是平仓追单：
           a. 开仓追单情况下，先得到未成交量order.untraded，若当前委托超过10秒还未成交（chase_interval = 10)，并且没有触发追单（chase_long_trigger = False），先把该委托撤销掉，然后把触发追单器启动；
           b. 平仓追单情况下，同样先得到未成交量，若委托超时还未成交并且平仓触发器没有启动，先撤单，然后启动平仓触发器；
        4、当所有未成交委托处理完毕后，活动委托字典将清空，此时order_finished状态从False变成True，用最新的买卖一档行情，该追单开仓的追单开仓，该追单平仓的赶紧平仓，每次操作后恢复委托触发器的初始状态：
        
        注意要点：Tick级精细挂撤单的管理逻辑，无法通过K线来进行回测检验，因此需通过仿真交易（比如期货基于SimNow）进行充分的测试。
        """
        active_orders = self.cta_engine.offset_converter.get_position_holding(self.vt_symbol).active_orders

        if active_orders:
            # 委托完成状态
            order_finished = False
            vt_orderid = list(active_orders.keys())[0]   # 委托单vt_orderid
            order = list(active_orders.values())[0]      # 委托单字典

            # 开仓追单，部分交易没有平仓指令(Offset.NONE)
            if order.offset in (Offset.NONE, Offset.OPEN):
                if order.direction == Direction.LONG:
                    self.long_trade_volume = order.untraded
                    if (tick.datetime - order.datetime).seconds > self.chase_interval and self.long_trade_volume > 0 and (not self.chase_long_trigger) and vt_orderid:
                        # 撤销之前发出的未成交订单
                        self.cancel_order(vt_orderid)
                        self.chase_long_trigger = True
                elif order.direction == Direction.SHORT:
                    self.short_trade_volume = order.untraded
                    if (tick.datetime - order.datetime).seconds > self.chase_interval and self.short_trade_volume > 0 and (not self.chase_short_trigger) and vt_orderid:
                        self.cancel_order(vt_orderid)
                        self.chase_short_trigger = True
            # 平仓追单
            else:
                if order.direction == Direction.SHORT:
                    self.sell_trade_volume = order.untraded
                    if (tick.datetime - order.datetime).seconds > self.chase_interval and self.sell_trade_volume > 0 and (not self.chase_sell_trigger) and vt_orderid:
                        self.cancel_order(vt_orderid)
                        self.chase_sell_trigger = True
                if order.direction == Direction.LONG:
                    self.cover_trade_volume = order.untraded
                    if (tick.datetime - order.datetime).seconds > self.chase_interval and self.cover_trade_volume > 0 and (not self.chase_cover_trigger) and vt_orderid:
                        self.cancel_order(vt_orderid)
                        self.chase_cover_trigger = True
        else:
            order_finished = True

        if self.chase_long_trigger and order_finished:
            self.buy(tick.ask_price_1, self.long_trade_volume)
            self.chase_long_trigger = False

        elif self.chase_short_trigger and order_finished:
            self.short(tick.bid_price_1, self.short_trade_volume)
            self.chase_short_trigger = False

        elif self.chase_sell_trigger and order_finished:
            self.sell(tick.bid_price_1, self.sell_trade_volume)
            self.chase_sell_trigger = False

        elif self.chase_cover_trigger and order_finished:
            self.cover(tick.ask_price_1, self.cover_trade_volume)
            self.chase_cover_trigger = False

    def calculate_trade_result(self, trade: TradeData) -> None:
        """计算各项成交数据。在 on_trade 中调用"""
        # 回测模式
        if self.inited and self.engine_type == EngineType.BACKTESTING:
            if trade.offset == Offset.OPEN:
                self.trade_net_volume += trade.volume
                self.cost = (trade.price * trade.volume + self.cost * (self.trade_net_volume - trade.volume)) / self.trade_net_volume

                if self.rate_type == RateType.FIXED:        # 固定手续费模式
                    self.trade_commission += trade.volume * self.rate
                else:                                       # 浮动手续费模式
                    self.trade_commission += trade.price * trade.volume * self.size * self.rate

                self.trade_slippage += trade.volume * self.size * self.slippage

            else:
                self.trade_net_volume -= trade.volume
                if trade.direction == Direction.LONG:       # 买平，平空仓
                    self.trade_pnl += (self.cost - trade.price) * trade.volume * self.size
                    self.pnl_point += self.cost - trade.price
                elif trade.direction == Direction.SHORT:    # 卖平，平多仓
                    self.trade_pnl += (trade.price - self.cost) * trade.volume * self.size
                    self.pnl_point += trade.price - self.cost

                if self.rate_type == RateType.FIXED:
                    self.trade_commission += trade.volume * self.rate
                else:
                    self.trade_commission += trade.price * trade.volume * self.size * self.rate
                self.trade_slippage += trade.volume * self.size * self.slippage

                self.net_pnl += self.trade_pnl - self.trade_commission - self.trade_slippage                       # 交易净盈亏 = 交易盈亏 - 手续费 - 滑点费
                self.balance += round(self.net_pnl, 2)

            if self.record_start_switch:
                self.trade_start_time_list.append(trade.datetime)
                self.record_start_switch = False

            self.trade_volume += trade.volume
            self.trade_net_volume_list.append(self.trade_net_volume)

            # 每完成一次完整交易
            if self.trade_net_volume == 0:
                self.trade_number += 1
                self.trade_number_list.append(self.trade_number)
                self.trade_type_list.append("多头" if trade.direction == Direction.SHORT else "空头")
                self.trade_volume_list.append(self.trade_volume)
                self.trade_pnl_list.append(self.trade_pnl)
                self.commission_list.append(self.trade_commission)
                self.slippage_list.append(self.trade_slippage)
                self.net_pnl_list.append(self.net_pnl)
                self.pnl_point_list.append(self.pnl_point - self.trade_volume * self.slippage)                    # 净盈亏点数，已扣除滑点
                self.balance_list.append(self.balance)
                self.trade_duration_list.append(round(self.entry_minute/60, 2))
                self.trade_date_list.append(trade.datetime.date())
                self.trade_end_time_list.append(trade.datetime)
                self.trade_volume = 0
                self.cost = 0
                self.trade_pnl = 0
                self.trade_commission = 0
                self.trade_slippage = 0
                self.net_pnl = 0
                self.pnl_point = 0
                self.record_start_switch = True

        # 实盘模式
        elif self.inited and self.engine_type == EngineType.LIVE:
            if self.pos > 0:
                if trade.direction == Direction.LONG:
                    self.cost = round((trade.price * trade.volume + self.cost * (self.pos - trade.volume)) / self.pos, 6)
                    self.trade_pnl = 0
                    self.trade_commission = 0
                    self.net_pnl = 0
                    self.pnl_point = 0

                    msg = f'开多！成交价格：{trade.price}，成交手数：{trade.volume}手，成交时间：{trade.datetime.replace(tzinfo=None)}，多头持仓均价：{self.cost}，交易信号：{self.signal} {"【锁仓】" if self.lock else ""}'
                    self.write_log(msg)
                    self.dingding(msg)
                    # self.weixin(msg)

                else:                                         # 卖平，平多仓
                    self.trade_pnl = round((trade.price - self.cost) * trade.volume * self.size, 2)

                    if trade.offset == Offset.CLOSETODAY:
                        self.trade_commission = round(((trade.volume * self.contract_data.open_commission + self.cost * trade.volume * self.size * self.contract_data.open_commission_ratio)
                                                + (trade.volume * self.contract_data.close_commission + trade.price * trade.volume * self.size * self.contract_data.close_commission_ratio)
                                                + (trade.volume * self.contract_data.close_commission_today + trade.price * trade.volume * self.size * self.contract_data.close_commission_today_ratio)), 2)
                    else:
                        self.trade_commission = round(((trade.volume * self.contract_data.open_commission + self.cost * trade.volume * self.size * self.contract_data.open_commission_ratio)
                                                + (trade.volume * self.contract_data.close_commission + trade.price * trade.volume * self.size * self.contract_data.close_commission_ratio)), 2)
                    
                    self.net_pnl = self.trade_pnl - self.trade_commission                 # 交易净盈亏 = 交易盈亏 - 手续费
                    self.pnl_point = round(trade.price - self.cost, 6)

                    msg = f'平多！平仓价格：{trade.price}，平仓手数：{trade.volume}手，平仓时间：{trade.datetime.replace(tzinfo=None)}，交易信号：{self.signal}，盈亏点数：{self.pnl_point}，平仓手续费：{self.trade_commission}，{"净盈利" if self.net_pnl >= 0 else "净亏损"}金额：{self.net_pnl}元 {"【锁仓】" if self.lock else ""}'
                    self.write_log(msg)
                    self.dingding(msg)
                    # self.weixin(msg)

            elif self.pos < 0:
                if trade.direction == Direction.SHORT:
                    self.cost = round((trade.price * trade.volume + self.cost * (abs(self.pos) - trade.volume)) / abs(self.pos), 6)
                    self.trade_pnl = 0
                    self.trade_commission = 0
                    self.net_pnl = 0
                    self.pnl_point = 0

                    msg = f'开空！成交价格：{trade.price}，成交手数：{trade.volume}手，成交时间：{trade.datetime.replace(tzinfo=None)}，空头持仓均价：{self.cost}，交易信号：{self.signal} {"【锁仓】" if self.lock else ""}'
                    self.write_log(msg)
                    self.dingding(msg)
                    # self.weixin(msg)

                else:                                       # 买平，平空仓
                    self.trade_pnl = round((self.cost - trade.price) * trade.volume * self.size, 2)

                    if trade.offset == Offset.CLOSETODAY:
                        self.trade_commission = round(((trade.volume * self.contract_data.open_commission + self.cost * trade.volume * self.size * self.contract_data.open_commission_ratio)
                                                + (trade.volume * self.contract_data.close_commission + trade.price * trade.volume * self.size * self.contract_data.close_commission_ratio)
                                                + (trade.volume * self.contract_data.close_commission_today + trade.price * trade.volume * self.size * self.contract_data.close_commission_today_ratio)), 2)
                    else:
                        self.trade_commission = round(((trade.volume * self.contract_data.open_commission + self.cost * trade.volume * self.size * self.contract_data.open_commission_ratio)
                                                + (trade.volume * self.contract_data.close_commission + trade.price * trade.volume * self.size * self.contract_data.close_commission_ratio)), 2)
                    
                    self.net_pnl = self.trade_pnl - self.trade_commission                 # 交易净盈亏 = 交易盈亏 - 手续费
                    self.pnl_point = round(self.cost - trade.price, 6)

                    msg = f'平空！平仓价格：{trade.price}，平仓手数：{trade.volume}手，平仓时间：{trade.datetime.replace(tzinfo=None)}，交易信号：{self.signal}，盈亏点数：{self.pnl_point}，平仓手续费：{self.trade_commission}，{"净盈利" if self.net_pnl >= 0 else "净亏损"}金额：{self.net_pnl}元 {"【锁仓】" if self.lock else ""}'
                    self.write_log(msg)
                    self.dingding(msg)
                    # self.weixin(msg)

            else:
                if trade.direction == Direction.LONG:       # 买平，平空仓
                    self.trade_pnl = round((self.cost - trade.price) * trade.volume * self.size, 2)

                    if trade.offset == Offset.CLOSETODAY:
                        self.trade_commission = round(((trade.volume * self.contract_data.open_commission + self.cost * trade.volume * self.size * self.contract_data.open_commission_ratio)
                                                + (trade.volume * self.contract_data.close_commission + trade.price * trade.volume * self.size * self.contract_data.close_commission_ratio)
                                                + (trade.volume * self.contract_data.close_commission_today + trade.price * trade.volume * self.size * self.contract_data.close_commission_today_ratio)), 2)
                    else:
                        self.trade_commission = round(((trade.volume * self.contract_data.open_commission + self.cost * trade.volume * self.size * self.contract_data.open_commission_ratio)
                                                + (trade.volume * self.contract_data.close_commission + trade.price * trade.volume * self.size * self.contract_data.close_commission_ratio)), 2)
                    
                    self.net_pnl = self.trade_pnl - self.trade_commission                 # 交易净盈亏 = 交易盈亏 - 手续费
                    self.pnl_point = round(self.cost - trade.price, 6)

                    msg = f'平空！平仓价格：{trade.price}，平仓手数：{trade.volume}手，平仓时间：{trade.datetime.replace(tzinfo=None)}，交易信号：{self.signal}，盈亏点数：{self.pnl_point}，平仓手续费：{self.trade_commission}，{"净盈利" if self.net_pnl >= 0 else "净亏损"}金额：{self.net_pnl}元 {"【锁仓】" if self.lock else ""}'
                    self.write_log(msg)
                    self.dingding(msg)
                    # self.weixin(msg)

                elif trade.direction == Direction.SHORT:    # 卖平，平多仓
                    self.trade_pnl = round((trade.price - self.cost) * trade.volume * self.size, 2)

                    if trade.offset == Offset.CLOSETODAY:
                        self.trade_commission = round(((trade.volume * self.contract_data.open_commission + self.cost * trade.volume * self.size * self.contract_data.open_commission_ratio)
                                                + (trade.volume * self.contract_data.close_commission + trade.price * trade.volume * self.size * self.contract_data.close_commission_ratio)
                                                + (trade.volume * self.contract_data.close_commission_today + trade.price * trade.volume * self.size * self.contract_data.close_commission_today_ratio)), 2)
                    else:
                        self.trade_commission = round(((trade.volume * self.contract_data.open_commission + self.cost * trade.volume * self.size * self.contract_data.open_commission_ratio)
                                                + (trade.volume * self.contract_data.close_commission + trade.price * trade.volume * self.size * self.contract_data.close_commission_ratio)), 2)
                    
                    self.net_pnl = self.trade_pnl - self.trade_commission                 # 交易净盈亏 = 交易盈亏 - 手续费
                    self.pnl_point = round(trade.price - self.cost, 6)

                    msg = f'平多！平仓价格：{trade.price}，平仓手数：{trade.volume}手，平仓时间：{trade.datetime.replace(tzinfo=None)}，交易信号：{self.signal}，盈亏点数：{self.pnl_point}，平仓手续费：{self.trade_commission}，{"净盈利" if self.net_pnl >= 0 else "净亏损"}金额：{self.net_pnl}元 {"【锁仓】" if self.lock else ""}'
                    self.write_log(msg)
                    self.dingding(msg)
                    # self.weixin(msg)

                self.cost = 0

        # 将每次成交的信号保存，以便后期复查
        self.signal_list.append(self.signal)
        self.signal = 0

    def save_trade_data_to_json(self, trade: TradeData):
        """实盘模式下实时记录成交信息至JSON文件。在 on_trade 中调用"""
        if self.inited and self.engine_type == EngineType.LIVE:
            # 判断.vntrader文件夹下是否存在vt_trade_data_record文件夹，不存在则创建文件夹
            folder_path = get_folder_path(f'vt_trade_data_record/{self.account.accountid}/{datetime.now().date().strftime("%Y%m%d")}')
            file_path = folder_path.joinpath(f"{self.symbol_strategy}.json")

            if file_path.exists() and not self.trade_data_dict:
                self.trade_data_dict = load_json(file_path)

            _trade = deepcopy(trade).__dict__
            _trade["exchange"] = trade.exchange.value
            _trade["direction"] = trade.direction.value
            _trade["offset"] = trade.offset.value
            _trade["datetime"] = trade.datetime.isoformat()
            _trade["trade_pnl"] = self.trade_pnl
            _trade["commission"] = self.trade_commission
            _trade["net_pnl"] = self.net_pnl
            _trade["pnl_point"] = self.pnl_point
            _trade["signal"] = self.signal_list[-1]
            _trade["symbol"] = self.symbol
            _trade["strategy"] = self.__class__.__name__
            _trade["accountid"] = self.account.accountid

            temp_dict = {_trade["datetime"]:_trade}
            self.trade_data_dict.update(temp_dict)
            save_json(file_path, self.trade_data_dict)

    def record_price_and_status(self, bar: BarData):
        """记录和更新每一分钟策略定义的各类价格和状态"""
        if self.pos == 0:
            self.track_highest = bar.high_price         # 记录多单开仓后的最高点
            self.track_lowest  = bar.low_price          # 记录空单开仓后的最低点
            self.track_minute = 0                       # 记录新高低点保持多少分钟
            self.entry_minute  = 0                      # 记录开仓后历时多少分钟
            self.exit_minute  += 1                      # 记录平仓后历时多少分钟
            self.floating_point  = 0                    # 记录开仓后的浮动盈亏点数
            self.floating_pnl = 0                       # 记录开仓后的浮动盈亏金额

        elif self.pos > 0:
            if bar.high_price > self.track_highest:
                self.track_highest = bar.high_price
                self.track_lowest = bar.low_price
                self.track_minute = 0
            else:
                self.track_lowest = min(self.track_lowest, bar.low_price)
                self.track_minute += 1

            self.entry_minute += 1
            self.exit_minute = 0
            self.floating_point  = bar.close_price - self.cost
            self.floating_pnl = self.floating_point * self.pos * self.size

        elif self.pos < 0:
            if bar.low_price < self.track_lowest:
                self.track_lowest  = bar.low_price
                self.track_highest = bar.high_price
                self.track_minute = 0
            else:
                self.track_highest = max(self.track_highest, bar.high_price)
                self.track_minute += 1

            self.entry_minute += 1
            self.exit_minute = 0
            self.floating_point  = self.cost - bar.close_price
            self.floating_pnl = self.floating_point * abs(self.pos) * self.size

    def empty_position(self, data: Union[BarData, TickData], exit_time: time = time(14, 58), lock: bool = False):
        """收盘前清仓"""
        if isinstance(data, BarData):
            if exit_time <= data.datetime.time() <= time(16, 00):

                self.cancel_all()       # 撤销当前所有挂单
                
                if self.pos > 0:
                    self.sell(data.close_price - 100 * self.pricetick, self.pos, lock=lock)
                    self.signal = 3.99

                elif self.pos < 0:
                    self.cover(data.close_price + 100 * self.pricetick, abs(self.pos), lock=lock)
                    self.signal = 4.99

        else:
            if exit_time <= data.datetime.time() <= time(16, 00):

                self.cancel_all()       # 撤销当前所有挂单

                if self.pos > 0:
                    self.sell(data.limit_down, self.pos, lock=lock)
                    self.signal = 3.99

                elif self.pos < 0:
                    self.cover(data.limit_up, abs(self.pos), lock=lock)
                    self.signal = 4.99
      

class CtaSignal(ABC):
    """"""

    def __init__(self):
        """"""
        self.signal_pos = 0

    @virtual
    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        pass

    @virtual
    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        pass

    def set_signal_pos(self, pos):
        """"""
        self.signal_pos = pos

    def get_signal_pos(self):
        """"""
        return self.signal_pos


class TargetPosTemplate(CtaTemplate):
    """"""
    tick_add = 1

    last_tick = None
    last_bar = None
    target_pos = 0

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.active_orderids = []
        self.cancel_orderids = []

        self.variables.append("target_pos")

    @virtual
    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        self.last_tick = tick

        if self.trading:
            self.trade()

    @virtual
    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self.last_bar = bar

    @virtual
    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        vt_orderid = order.vt_orderid

        if not order.is_active():
            if vt_orderid in self.active_orderids:
                self.active_orderids.remove(vt_orderid)

            if vt_orderid in self.cancel_orderids:
                self.cancel_orderids.remove(vt_orderid)

    def check_order_finished(self):
        """"""
        if self.active_orderids:
            return False
        else:
            return True

    def set_target_pos(self, target_pos):
        """"""
        self.target_pos = target_pos
        self.trade()

    def trade(self):
        """"""
        if not self.check_order_finished():
            self.cancel_old_order()
        else:
            self.send_new_order()

    def cancel_old_order(self):
        """"""
        for vt_orderid in self.active_orderids:
            if vt_orderid not in self.cancel_orderids:
                self.cancel_order(vt_orderid)
                self.cancel_orderids.append(vt_orderid)

    def send_new_order(self):
        """"""
        pos_change = self.target_pos - self.pos
        if not pos_change:
            return

        long_price = 0
        short_price = 0

        if self.last_tick:
            if pos_change > 0:
                long_price = self.last_tick.ask_price_1 + self.tick_add
                if self.last_tick.limit_up:
                    long_price = min(long_price, self.last_tick.limit_up)
            else:
                short_price = self.last_tick.bid_price_1 - self.tick_add
                if self.last_tick.limit_down:
                    short_price = max(short_price, self.last_tick.limit_down)

        else:
            if pos_change > 0:
                long_price = self.last_bar.close_price + self.tick_add
            else:
                short_price = self.last_bar.close_price - self.tick_add

        if self.engine_type == EngineType.BACKTESTING:
            if pos_change > 0:
                vt_orderids = self.buy(long_price, abs(pos_change))
            else:
                vt_orderids = self.short(short_price, abs(pos_change))
            self.active_orderids.extend(vt_orderids)

        else:
            if self.active_orderids:
                return

            if pos_change > 0:
                if self.pos < 0:
                    if pos_change < abs(self.pos):
                        vt_orderids = self.cover(long_price, pos_change)
                    else:
                        vt_orderids = self.cover(long_price, abs(self.pos))
                else:
                    vt_orderids = self.buy(long_price, abs(pos_change))
            else:
                if self.pos > 0:
                    if abs(pos_change) < self.pos:
                        vt_orderids = self.sell(short_price, abs(pos_change))
                    else:
                        vt_orderids = self.sell(short_price, abs(self.pos))
                else:
                    vt_orderids = self.short(short_price, abs(pos_change))
            self.active_orderids.extend(vt_orderids)
