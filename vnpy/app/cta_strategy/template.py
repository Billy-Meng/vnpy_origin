# -*- coding:utf-8 -*-
import os
from abc import ABC
from copy import copy, deepcopy
from typing import Any, Callable, Union
from datetime import datetime, timedelta
from threading import Thread, Timer

import pandas as pd

from vnpy.trader.constant import Interval, Direction, Offset
from vnpy.trader.object import BarData, TickData, OrderData, TradeData, AccountData, ContractData
from vnpy.trader.utility import virtual, get_file_path, save_json

from .base import StopOrder, EngineType

# 弹窗提醒所需库
import win32api, win32con

# 钉钉通知所需库
import urllib, requests
import json


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

        self.inited = False
        self.trading = False
        self.pos = 0

        # Copy a new variables list here to avoid duplicate insert when multiple
        # strategy instances are created with the same strategy class.
        self.variables = copy(self.variables)
        self.variables.insert(0, "inited")
        self.variables.insert(1, "trading")
        self.variables.insert(2, "pos")

        self.trade_data_list = []        # 成交信息列表
        self.trade_data_dict = dict()    # 成交信息字典

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

    def buy(self, price: float, volume: float, stop: bool = False, lock: bool = False):
        """
        Send buy order to open a long position.
        """
        return self.send_order(Direction.LONG, Offset.OPEN, price, volume, stop, lock)

    def sell(self, price: float, volume: float, stop: bool = False, lock: bool = False):
        """
        Send sell order to close a long position.
        """
        return self.send_order(Direction.SHORT, Offset.CLOSE, price, volume, stop, lock)

    def short(self, price: float, volume: float, stop: bool = False, lock: bool = False):
        """
        Send short order to open as short position.
        """
        return self.send_order(Direction.SHORT, Offset.OPEN, price, volume, stop, lock)

    def cover(self, price: float, volume: float, stop: bool = False, lock: bool = False):
        """
        Send cover order to close a short position.
        """
        return self.send_order(Direction.LONG, Offset.CLOSE, price, volume, stop, lock)

    def send_order(
        self,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        stop: bool = False,
        lock: bool = False
    ):
        """
        Send a new order.
        """
        if self.trading:
            vt_orderids = self.cta_engine.send_order(
                self, direction, offset, price, volume, stop, lock
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


    # 自定义增强功能

    def get_contract_data(self) -> ContractData:
        """
        获取合约信息。策略初始化时调用，动态赋值contract_data, gateway_name, symbol, pricetick, size, margin_rate
        """
        if self.get_engine_type() == EngineType.LIVE:
            self.contract_data = self.cta_engine.main_engine.get_contract(self.vt_symbol)

            if self.contract_data:                
                self.gateway_name = self.contract_data.gateway_name      # 接口名称
                self.symbol = self.contract_data.symbol                  # 合约代码
                self.pricetick = self.contract_data.pricetick            # 最小变动价位
                self.size = self.contract_data.size                      # 合约乘数
                self.margin_ratio = self.contract_data.margin_ratio      # 保证金比率
                self.commission = self.contract_data.open_commission + self.contract_data.close_commission                          # 手续费
                self.commission_ratio = self.contract_data.open_commission_ratio + self.contract_data.close_commission_ratio        # 手续费率
                self.write_log((f"合约信息查询成功：Symbol【 {self.symbol} 】，Pricetick【 {self.pricetick} 】，Size【 {self.size} 】，" +
                                f"保证金率【 {self.margin_ratio} 】，手续费【 {self.commission} 】，手续费率【 {self.commission_ratio} 】"))

        else:
            self.gateway_name = self.cta_engine.gateway_name      # 接口名称，回测的gateway_name为"BACKTESTING"
            self.vt_symbol = self.cta_engine.vt_symbol            # 本地合约代码
            self.symbol = self.cta_engine.symbol                  # 合约代码
            self.pricetick = self.cta_engine.pricetick            # 最小变动价位
            self.size = self.cta_engine.size                      # 合约乘数
            self.write_log(f"合约信息查询成功：Symbol【 {self.symbol} 】，Pricetick【 {self.pricetick} 】，Size【 {self.size} 】")

    def get_account_data(self, account_id:str) -> AccountData:
        """获取账户信息"""
        if self.get_engine_type() == EngineType.LIVE:
            vt_account_id = f"{self.gateway_name}.{account_id}"
            account_data = self.cta_engine.main_engine.get_account(vt_account_id)

            if account_data:
                return account_data
            else:
                return None
        
    def popup_warning(self, msg:str):
        """
        弹窗消息通知
        """
        def run_popup_warning(msg:str):

            info_strategy = f"【{self.symbol}】{self.strategy_name}\n"
            info_time = f'\n时间:{datetime.now().strftime("%Y-%m-%d %H:%M:%S %a")}'
            win32api.MessageBox(0, info_strategy + msg + info_time, "交易提醒", win32con.MB_ICONWARNING)

        if self.inited and self.get_engine_type() == EngineType.LIVE:
            thread_popup = Thread(target=run_popup_warning, name="popup_warning", args=(msg,) )
            thread_popup.start()

    def dingding(self, msg:str, url:str):
        """
        钉钉机器人消息通知【记得修改URL】
        """
        def run_dingding(msg:str, url:str):
            
                info_strategy = f"【{self.symbol}】{self.strategy_name}\n"
                info_time = f'\n时间:{datetime.now().strftime("%Y-%m-%d %H:%M:%S %a")}'

                program = {
                    "msgtype": "text",
                    "text": {"content": info_strategy + msg + info_time},
                }

                headers = {'Content-Type': 'application/json'}

                requests.post(url, data=json.dumps(program), headers=headers)

        if self.inited and self.get_engine_type() == EngineType.LIVE:
            thread_dingding = Thread(target=run_dingding, name="dingding", args=(msg, url) )
            thread_dingding.start()

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
            
            self.trade_time = df.loc[product].to_dict()
            if self.trade_time["night_trade"] == True:
                night_time = f'夜盘【 {self.trade_time["night_start"]} ~ {self.trade_time["night_end"]} 】'
            else:
                night_time = "无夜盘。"

            self.write_log(f'品种交易时间：日盘【 {self.trade_time["am_start"]} ~ {self.trade_time["pm_end"]} 】，{night_time}')

        else:
            self.write_log("找不到该品种交易时间")


    def record_trade_data(self, trade: TradeData):
        """实时记录成交记录，在 on_trade 调用"""
        self.trade_data_list.append(trade.__dict__)

    def clean_trade_data(self):
        """清洗成交记录"""
        if self.trade_data_list:
            trade_df = pd.DataFrame(self.trade_data_list)
            trade_df = trade_df.set_index("datetime")
            # trade_df包括字段："gateway_name", "symbol", "exchange", "orderid", "tradeid", "direction", "offset", "price", "volume", "vt_symbol", "vt_orderid", "vt_tradeid"
            trade_df = trade_df.rename(columns={"price": "trade_price", "volume": "trade_volume"})
            trade_df["exchange"] = trade_df.exchange.apply(lambda x : x.value)
            trade_df["direction"] = trade_df.direction.apply(lambda x : x.value)
            trade_df["offset"] = trade_df.offset.apply(lambda x : x.value)

            return trade_df
        else:
            return None

    def save_trade_data(self):
        """启动定时器线程保存成交记录至CSV，在 on_init 调用"""
        def run_save():
            trade_df = self.clean_trade_data()
            if trade_df:

                # 判断.vntrader文件夹下是否存在Trade_Record文件夹，不存在则创建文件夹
                if not os.path.exists(get_file_path(f"成交记录信息/{datetime.now().date()}")):
                    os.makedirs(get_file_path(f"成交记录信息/{datetime.now().date()}"))             # os.makedirs()创建多级目录

                filepath = get_file_path(f'成交记录信息/{datetime.now().date()}/{self.symbol}-{self.strategy_name}.csv')
                trade_df.to_csv(filepath)

        if self.inited and self.get_engine_type() == EngineType.LIVE:
            """实盘中每天下午三点半执行保存成交记录的任务"""
            if datetime.now().time() > datetime.now().time().replace(hour=15, minute=30, microsecond=0):
                save_thread = Timer(interval=(datetime.now().replace(hour=15, minute=30, second=0, microsecond=0) + timedelta(days=1) - datetime.now()).seconds, function=run_save)
                save_thread.start()
                self.write_log(f"定时器已启动，即将于明日 {datetime.now().replace(hour=15, minute=30, second=0, microsecond=0) + timedelta(days=1)} 保存成交记录至CSV")

            else:
                save_thread = Timer(interval=(datetime.now().replace(hour=15, minute=30, second=0, microsecond=0) - datetime.now()).seconds, function=run_save)
                save_thread.start()
                self.write_log(f"定时器已启动，即将于今日 {datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)} 保存成交记录至CSV")

    def save_trade_data_to_json(self, trade: TradeData):
        """实时记录成交信息至JSON文件，在 on_trade 调用"""
        if self.inited and self.get_engine_type() == EngineType.LIVE:
        
            # 判断.vntrader文件夹下是否存在Trade_Record文件夹，不存在则创建文件夹
            if not os.path.exists(get_file_path(f"成交记录信息/{datetime.now().date()}")):
                os.makedirs(get_file_path(f"成交记录信息/{datetime.now().date()}"))             # os.makedirs()创建多级目录

            trade_deepcopy = deepcopy(trade)
            trade_ = trade_deepcopy.__dict__
            trade_["exchange"] = trade_deepcopy.exchange.value
            trade_["direction"] = trade_deepcopy.direction.value
            trade_["offset"] = trade_deepcopy.offset.value
            trade_["datetime"] = f'{trade_deepcopy.datetime.replace(tzinfo=None)}'
            temp_dict = {f'{trade_["datetime"]} {self.symbol} {self.strategy_name}':trade_}
            self.trade_data_dict.update(temp_dict)
            save_json(f'成交记录信息/{datetime.now().date()}/{self.symbol}-{self.strategy_name}.json', self.trade_data_dict)


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

        if self.get_engine_type() == EngineType.BACKTESTING:
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
