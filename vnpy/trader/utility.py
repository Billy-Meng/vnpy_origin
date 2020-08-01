# -*- coding:utf-8 -*-
"""
General utility functions.
"""

import json
import logging
import os
import sys
import shelve
import requests
import win32api, win32con
from datetime import datetime, timedelta
from time import perf_counter
from pathlib import Path
from typing import Callable, Dict, Tuple, Union
from decimal import Decimal
from math import floor, ceil
from collections import defaultdict

import numpy as np
import pandas as pd
import talib
from chinese_calendar import is_workday, is_holiday

from .object import BarData, TickData
from .constant import Exchange, Interval


log_formatter = logging.Formatter('[%(asctime)s] %(message)s')


def extract_vt_symbol(vt_symbol: str) -> Tuple[str, Exchange]:
    """
    :return: (symbol, exchange)
    """
    symbol, exchange_str = vt_symbol.split(".")
    return symbol, Exchange(exchange_str)


def generate_vt_symbol(symbol: str, exchange: Exchange) -> str:
    """
    return vt_symbol
    """
    return f"{symbol}.{exchange.value}"


def _get_trader_dir(temp_name: str) -> Tuple[Path, Path]:
    """
    Get path where trader is running in.
    """
    cwd = Path.cwd()
    temp_path = cwd.joinpath(temp_name)

    # If .vntrader folder exists in current working directory,
    # then use it as trader running path.
    if temp_path.exists():
        return cwd, temp_path

    # Otherwise use home path of system.
    home_path = Path.home()
    temp_path = home_path.joinpath(temp_name)

    # Create .vntrader folder under home path if not exist.
    if not temp_path.exists():
        temp_path.mkdir()

    return home_path, temp_path


TRADER_DIR, TEMP_DIR = _get_trader_dir(".vntrader")
sys.path.append(str(TRADER_DIR))


def get_file_path(filename: str) -> Path:
    """
    Get path for temp file with filename.
    """
    return TEMP_DIR.joinpath(filename)


def get_folder_path(folder_name: str) -> Path:
    """
    Get path for temp folder with folder name.
    """
    folder_path = TEMP_DIR.joinpath(folder_name)

    # if not folder_path.exists():
    #     folder_path.mkdir()

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)                       # os.makedirs()创建多级目录

    return folder_path


def get_icon_path(filepath: str, ico_name: str) -> str:
    """
    Get path for icon file with ico name.
    """
    ui_path = Path(filepath).parent
    icon_path = ui_path.joinpath("ico", ico_name)
    return str(icon_path)


def load_json(filename: str) -> dict:
    """
    Load data from json file in temp path.
    """
    filepath = get_file_path(filename)

    if filepath.exists():
        with open(filepath, mode="r", encoding="UTF-8") as f:
            data = json.load(f)
        return data
    else:
        save_json(filename, {})
        return {}


def save_json(filename: str, data: dict) -> None:
    """
    Save data into json file in temp path.
    """
    filepath = get_file_path(filename)
    with open(filepath, mode="w+", encoding="UTF-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


def load_shelve(filename: str) -> dict:
    """
    Load data from shelve file in temp path.(文件名不需加扩展名)
    """
    file_path = get_file_path(f"{filename}.dat")

    if file_path.exists():
        with shelve.open(str(file_path).replace(".dat", "")) as f:
            data = f["data"]
        return data
    else:
        save_shelve(filename, {})
        return {}


def save_shelve(filename: str, data: dict) -> None:
    """
    Save data into shelve file in temp path. (文件名不需加扩展名)
    """
    if "/" in filename:
        index = filename.rfind("/")
        folder_path = get_folder_path(filename[:index])

        with shelve.open(f"{folder_path}\\{filename[index + 1:]}") as f:
            f["data"] = data

    else:
        index = filename.rfind("\\")
        folder_path = get_folder_path(filename[:index])

        with shelve.open(f"{folder_path}\\{filename[index + 1:]}") as f:
            f["data"] = data


def remain_alpha(symbol: str) -> str:
    """
    返回合约的字母字符串大写
    """
    symbol_mark = "".join(list(filter(str.isalpha, symbol)))
    
    return symbol_mark.upper()


def round_to(value: float, target: float) -> float:
    """
    Round price to price tick value.
    """
    value = Decimal(str(value))
    target = Decimal(str(target))
    rounded = float(int(round(value / target)) * target)
    return rounded


def floor_to(value: float, target: float) -> float:
    """
    Similar to math.floor function, but to target float number.
    """
    value = Decimal(str(value))
    target = Decimal(str(target))
    result = float(int(floor(value / target)) * target)
    return result


def ceil_to(value: float, target: float) -> float:
    """
    Similar to math.ceil function, but to target float number.
    """
    value = Decimal(str(value))
    target = Decimal(str(target))
    result = float(int(ceil(value / target)) * target)
    return result


def get_digits(value: float) -> int:
    """
    Get number of digits after decimal point.
    """
    value_str = str(value)

    if "e-" in value_str:
        _, buf = value_str.split("e-")
        return int(buf)
    elif "." in value_str:
        _, buf = value_str.split(".")
        return len(buf)
    else:
        return 0


def run_time(func):
    '''计算函数运行耗时装饰器'''
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        print (f"{datetime.now()}\t{func.__name__} 运行耗时：{end_time - start_time}秒")
        return result
    return wrapper

def workdays(start, end):
    '''
    计算两个日期间的工作日
    格式为 datetime.date() 或 "%Y-%m-%d"
    '''
     # 字符串格式日期的处理
    if type(start) == str:
        start = datetime.strptime(start,'%Y-%m-%d').date()
    if type(end) == str:
        end = datetime.strptime(end,'%Y-%m-%d').date()

    # 开始日期大，颠倒开始日期和结束日期
    if start > end:
        start,end = end,start

    counts = 0
    while True:
        if start > end:
            break
        if is_workday(start):
            counts += 1
        start += timedelta(days=1)

    return counts

def tradedays(start, end):
    '''
    计算两个日期间的中国市场交易日
    格式为 datetime.date() 或 "%Y-%m-%d"
    '''
    # 字符串格式日期的处理
    if type(start) == str:
        start = datetime.strptime(start,'%Y-%m-%d').date()
    if type(end) == str:
        end = datetime.strptime(end,'%Y-%m-%d').date()

    # 开始日期大，颠倒开始日期和结束日期
    if start > end:
        start,end = end,start
        
    counts = 0
    while True:
        if start > end:
            break
        if is_holiday(start) or start.weekday()==5 or start.weekday()==6:
            start += timedelta(days=1)
            continue
        counts += 1
        start += timedelta(days=1)

    return counts

def popup_warning(msg: str):
    """ 弹窗消息通知 """
    info_time = f'\n时间:{datetime.now().strftime("%Y-%m-%d %H:%M:%S %a")}'
    win32api.MessageBox(0, msg + info_time, "交易提醒", win32con.MB_ICONWARNING)


SETTINGS = load_json("vt_setting.json")

def send_dingding(msg: str):
    """ 钉钉机器人消息通知 """

    info_time = f'\n时间:{datetime.now().strftime("%Y-%m-%d %H:%M:%S %a")}'
    dingding_url_list = SETTINGS["dingding.url"].split(";")

    program = {
        "msgtype": "text",
        "text": {"content": msg + info_time},
        # "at": {"isAtAll": True}
    }

    # program = {
    #     "msgtype": "markdown",
    #     "markdown": {
    #         "title":"标题",
    #         "text": f"#### {msg} \n ###### {info_time}"
    #     },
    #     "at": {"isAtAll": True}
    # }

    headers = {'Content-Type': 'application/json'}

    for url in dingding_url_list:
        requests.post(url, data=json.dumps(program), headers=headers)

def send_weixin(msg: str):
    """通过FTQQ发送微信消息 http://sc.ftqq.com/3.version"""

    info_time = f'\n时间:{datetime.now().strftime("%Y-%m-%d %H:%M:%S %a")}'
    weixin_sckey_list = SETTINGS["weixin.sckey"].split(";")

    program = {
        "text": msg,
        "desp": info_time
    }

    for sckey in weixin_sckey_list:
        url = f"https://sc.ftqq.com/{sckey}.send"
        requests.get(url, params=program)


class BarGenerator:
    """
    For:
    1. generating 1 minute bar data from tick data
    2. generating x minute bar/x hour bar data from 1 minute data

    Notice:
    1. for x minute bar, x must be able to divide 60: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60
    2. for x hour bar, x can be any number

    增强版K线合成器。可合成 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60秒钟K线。
    """

    def __init__(
        self,
        on_second_bar: Callable = None,
        second_window: int = 60,
        on_bar: Callable = None,
        window: int = 1,
        on_window_bar: Callable = None,
        interval: Interval = Interval.MINUTE,
        division_method: bool = False,
        nature_day: bool = False
    ):
        """Constructor"""
        self.bar: BarData = None
        self.second_window: int = second_window
        self.second_bar_range = [i for i in range(0, 60, self.second_window)]
        
        self.second_bar: BarData = None
        self.on_second_bar: Callable = on_second_bar
        self.last_tick: TickData = None

        self.second_window_bar: BarData = None
        self.on_bar: Callable = on_bar

        self.interval: Interval = interval
        self.window_bar: BarData = None        
        self.interval_count: int = 0
        
        self.last_bar: BarData = None
        self.window: int = window
        self.on_window_bar: Callable = on_window_bar

        self.division_method = division_method
        self.nature_day = nature_day

        self.bar_data_list = []      # 生成指定周期的Bar缓存列表

    def update_tick(self, tick: TickData) -> None:
        """
        Update new tick data into generator.
        """
        new_minute = False

        # Filter tick data with 0 last price
        if not tick.last_price:
            return

        # Filter tick data with older timestamp
        if self.last_tick and tick.datetime < self.last_tick.datetime:
            return

        if not self.bar:
            new_minute = True
        
        # 官方版本合成方式。从每分钟的00秒开始，到本分钟59秒，合成一分钟bar，bar的时间戳为合成起始时间。Bug:会丢失交易所休市时最后一两个推送的Tick数据。
        elif self.bar.datetime.minute != tick.datetime.minute:
        
        # 从每分钟的50秒开始，到下一分钟的49秒，合成一分钟bar
        # elif tick.datetime.second >= 50 and self.last_tick.datetime.second < 50:

        # 从每分钟的01秒开始，到下一分钟的00秒，合成一分钟bar，bar的时间戳为合成结束时间。Bug:会丢失交易所休市时最后一分钟bar数据。
        # elif tick.datetime.second >= 1 and self.last_tick.datetime.second < 1:

            self.bar.datetime = self.bar.datetime.replace(second=0, microsecond=0)
            self.on_bar(self.bar)

            new_minute = True

        if new_minute:
            self.bar = BarData(
                symbol=tick.symbol,
                exchange=tick.exchange,
                interval=Interval.MINUTE,
                datetime=tick.datetime,
                gateway_name=tick.gateway_name,
                open_price=tick.last_price,
                high_price=tick.last_price,
                low_price=tick.last_price,
                close_price=tick.last_price,
                open_interest=tick.open_interest
            )
        else:
            self.bar.high_price = max(self.bar.high_price, tick.last_price)
            self.bar.low_price = min(self.bar.low_price, tick.last_price)
            self.bar.close_price = tick.last_price
            self.bar.open_interest = tick.open_interest
            self.bar.datetime = tick.datetime

        if self.last_tick:
            volume_change = tick.volume - self.last_tick.volume
            self.bar.volume += max(volume_change, 0)

        self.last_tick = tick

    def update_second_bar(self, tick: TickData):
        """
        Update new tick data into second generator.
        """
        # 待合成的second_bar仅限 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60秒钟 K线
        if self.second_window not in [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]:
            return

        # Filter tick data with 0 last price
        if not tick.last_price:
            return

        # Filter tick data with older timestamp
        if self.last_tick and tick.datetime < self.last_tick.datetime:
            return


        new_second_bar = False
        
        if not self.second_bar:
            new_second_bar = True

        elif (tick.datetime.second in self.second_bar_range) and (tick.datetime.second != self.last_tick.datetime.second):
            bar_second = (tick.datetime - timedelta(seconds=self.second_window)).second
            self.second_bar.datetime = self.second_bar.datetime.replace(second=bar_second, microsecond=0)
            self.on_second_bar(self.second_bar)
            new_second_bar = True

        if new_second_bar:
            self.second_bar = BarData(
                symbol=tick.symbol,
                exchange=tick.exchange,
                interval=f"{self.second_window}s",
                datetime=tick.datetime,
                gateway_name=tick.gateway_name,
                open_price=tick.last_price,
                high_price=tick.last_price,
                low_price=tick.last_price,
                close_price=tick.last_price,
                open_interest=tick.open_interest
            )
        else:
            self.second_bar.high_price = max(self.second_bar.high_price, tick.last_price)
            self.second_bar.low_price = min(self.second_bar.low_price, tick.last_price)
            self.second_bar.close_price = tick.last_price
            self.second_bar.open_interest = tick.open_interest
            self.second_bar.datetime = tick.datetime

        if self.last_tick:
            volume_change = tick.volume - self.last_tick.volume
            self.second_bar.volume += max(volume_change, 0)

        # Cache last tick object
        self.last_tick = tick

    def update_1_minute_bar(self, second_bar: BarData) -> None:
        """
        将Tick合成或数据接口加载的X秒钟second_bar，合成1分钟bar,并回调on_bar。
        """
        if self.second_window == 60:
            self.on_bar(second_bar)
            return

        # If not inited, creaate second window bar object
        if not self.second_window_bar:
            # Generate timestamp for bar data
            dt = second_bar.datetime.replace(second=0, microsecond=0)

            self.second_window_bar = BarData(
                symbol=second_bar.symbol,
                exchange=second_bar.exchange,
                interval=Interval.MINUTE,
                datetime=dt,
                gateway_name=second_bar.gateway_name,
                open_price=second_bar.open_price,
                high_price=second_bar.high_price,
                low_price=second_bar.low_price
            )

        # Otherwise, update high/low price into window bar
        else:
            self.second_window_bar.high_price = max(
                self.second_window_bar.high_price, second_bar.high_price)
            self.second_window_bar.low_price = min(
                self.second_window_bar.low_price, second_bar.low_price)

        # Update close price/volume into window bar
        self.second_window_bar.close_price = second_bar.close_price
        self.second_window_bar.volume += int(second_bar.volume)
        self.second_window_bar.open_interest = second_bar.open_interest

        new_minute = False

        # 传入的second_bar已经走完一分钟，则启动 on_bar
        if (second_bar.datetime + timedelta(seconds=self.second_window)).second == 0:
            new_minute = True

        if new_minute:
            self.on_bar(self.second_window_bar)
            self.second_window_bar = None

    def update_bar(self, bar: BarData) -> None:
        """
        Update 1 minute bar into generator
        """
        if self.window == 1 and self.interval == Interval.MINUTE:
            self.on_window_bar(bar)
            return

        # If not inited, creaate window bar object
        if not self.window_bar:
            # Generate timestamp for bar data
            if self.interval == Interval.MINUTE:
                dt = bar.datetime.replace(second=0, microsecond=0)

            elif self.interval == Interval.HOUR:
                dt = bar.datetime.replace(minute=0, second=0, microsecond=0)

            elif self.interval == Interval.DAILY:
                dt = bar.datetime.replace(hour=0, minute=0, second=0, microsecond=0)

            self.window_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                interval=self.interval,
                datetime=dt,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price
            )

        # Otherwise, update high/low price into window bar
        else:
            self.window_bar.high_price = max(
                self.window_bar.high_price, bar.high_price)
            self.window_bar.low_price = min(
                self.window_bar.low_price, bar.low_price)

        # Update close price/volume into window bar
        self.window_bar.close_price = bar.close_price
        self.window_bar.volume += int(bar.volume)
        self.window_bar.open_interest = bar.open_interest


        # Check if window bar completed
        finished = False

        # X分钟K线合成
        if self.interval == Interval.MINUTE:
            if self.division_method:
                # 整除切分法进行分钟K线合成，合成 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60分钟K线，可刚好被1小时整除，切分间隔一致。合成其他周期K线则切分间隔不一致。
                if not (bar.datetime.minute + 1) % self.window:
                    finished = True

            else:
                # 计数切分法进行分钟K线合成，可以合成任意分钟K线，非整点
                if self.last_bar and bar.datetime.minute != self.last_bar.datetime.minute:
                    if self.window == 1:
                            finished = True
                    
                    else:
                        self.interval_count += 1
                        if not self.interval_count % self.window:
                            finished = True
                            self.interval_count = 0
            
        # X小时K线合成，计数切分法进行N小时K线合成，可以合成任意小时K线
        elif self.interval == Interval.HOUR:
            if self.last_bar and bar.datetime.hour != self.last_bar.datetime.hour:
                # 1-hour bar
                if self.window == 1:
                    finished = True
                # x-hour bar
                else:
                    self.interval_count += 1

                    if not self.interval_count % self.window:
                        finished = True
                        self.interval_count = 0

        # 日K线合成
        elif self.interval == Interval.DAILY:
            if self.nature_day:
                # 针对每天开盘和收盘为同一天的日K线合成，按自然交易日时间划分新的一天
                if self.last_bar and bar.datetime.date() != self.last_bar.datetime.date():
                    finished = True
            else:
                # 针对国内商品期货每天开盘时间为晚上九点，收盘时间为次日的情况
                if self.last_bar and bar.datetime.date() != ArrayManager().RealDate(bar):
                    finished = True


        if finished:
            self.on_window_bar(self.window_bar)
            self.window_bar = None

        # Cache last bar object
        self.last_bar = bar

    def generate_bar(self, bar: BarData) -> None:
        """
        通过一分钟的Bar，生成各级别周期Bar，并缓存在列表中
        """
        # If not inited, creaate window bar object
        if not self.window_bar:
            # Generate timestamp for bar data
            if self.interval == Interval.MINUTE:
                dt = bar.datetime.replace(second=0, microsecond=0)

            elif self.interval == Interval.HOUR:
                dt = bar.datetime.replace(minute=0, second=0, microsecond=0)

            elif self.interval == Interval.DAILY:
                dt = bar.datetime.replace(hour=0, minute=0, second=0, microsecond=0)

            self.window_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                interval=self.interval,
                datetime=dt,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price
            )

        # Otherwise, update high/low price into window bar
        else:
            self.window_bar.high_price = max(
                self.window_bar.high_price, bar.high_price)
            self.window_bar.low_price = min(
                self.window_bar.low_price, bar.low_price)

        # Update close price/volume into window bar
        self.window_bar.close_price = bar.close_price
        self.window_bar.volume += int(bar.volume)
        self.window_bar.open_interest = bar.open_interest


        # Check if window bar completed
        finished = False

        # X分钟K线合成
        if self.interval == Interval.MINUTE:
            if self.division_method:
                # 整除切分法进行分钟K线合成，合成 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60分钟K线，可刚好被1小时整除，切分间隔一致。合成其他周期K线则切分间隔不一致。
                if self.window not in [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]:
                    print("整除法合成N分钟K线，时间窗口须为 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60 其中之一！")
                    return
                else:
                    if not (bar.datetime.minute + 1) % self.window:
                        finished = True

            else:
                # 计数切分法进行分钟K线合成，可以合成任意分钟K线，非整点
                if self.last_bar and bar.datetime.minute != self.last_bar.datetime.minute:
                    if self.window == 1:
                            finished = True
                    
                    else:
                        self.interval_count += 1
                        if not self.interval_count % self.window:
                            finished = True
                            self.interval_count = 0
            
        # X小时K线合成，计数切分法进行N小时K线合成，可以合成任意小时K线
        elif self.interval == Interval.HOUR:
            if self.last_bar and bar.datetime.hour != self.last_bar.datetime.hour:
                # 1-hour bar
                if self.window == 1:
                    finished = True
                # x-hour bar
                else:
                    self.interval_count += 1

                    if not self.interval_count % self.window:
                        finished = True
                        self.interval_count = 0

        # 日K线合成
        elif self.interval == Interval.DAILY:
            if self.nature_day:
                # 针对每天开盘和收盘为同一天的日K线合成，按自然交易日时间划分新的一天
                if self.last_bar and bar.datetime.date() != self.last_bar.datetime.date():
                    finished = True
            else:
                # 针对国内商品期货每天开盘时间为晚上九点，收盘时间为次日的情况
                if self.last_bar and bar.datetime.date() != ArrayManager().RealDate(bar):
                    finished = True

        if finished:
            self.bar_data_list.append(self.window_bar)
            self.window_bar = None

        # Cache last bar object
        self.last_bar = bar

    def get_bar_data_df(self):
        bar_data = [bar.__dict__ for bar in self.bar_data_list]
        bar_data_df = pd.DataFrame(bar_data)
        bar_data_df = bar_data_df.set_index("datetime")
        bar_data_df = bar_data_df[["symbol", "open_price", "high_price", "low_price", "close_price", "volume", "open_interest"]]

        return bar_data_df

    def generate(self) -> None:
        """
        Generate the bar data and call callback immediately.
        """
        bar = self.bar

        if self.bar:
            bar.datetime = bar.datetime.replace(second=0, microsecond=0)
            self.on_bar(bar)

        self.bar = None
        return bar


class ArrayManager(object):
    """
    For:
    1. time series container of bar data
    2. calculating technical indicator value
    """

    def __init__(self, size: int = 100, nature_day: bool = False, log: bool = False):
        """Constructor"""
        self.count: int = 0
        self.size: int = size
        self.nature_day: bool = nature_day
        self.log: bool = log
        self.inited: bool = False
        self.new_day: bool = False
        self.last_bar: BarData = None

        self.open_array: np.ndarray = np.zeros(size)
        self.high_array: np.ndarray = np.zeros(size)
        self.low_array: np.ndarray = np.zeros(size)
        self.close_array: np.ndarray = np.zeros(size)
        self.volume_array: np.ndarray = np.zeros(size)
        self.open_interest_array: np.ndarray = np.zeros(size)

        self.countif_dict = defaultdict(list)

    def update_bar(self, bar: BarData) -> None:
        """
        Update new bar data into array manager.
        """
        self.count += 1
        if not self.inited and self.count >= self.size:
            self.inited = True

        if self.nature_day:
            if self.last_bar and self.last_bar.datetime.date() != bar.datetime.date():        # 按自然交易日时间划分新的一天
                self.new_day = True
            else:
                self.new_day = False
        else:
            if self.last_bar and self.last_bar.datetime.date() != self.RealDate(bar):         # 针针对国内夜盘开盘时间为次日开盘时间
                self.new_day = True
            else:
                self.new_day = False

        self.open_array[:-1] = self.open_array[1:]
        self.high_array[:-1] = self.high_array[1:]
        self.low_array[:-1] = self.low_array[1:]
        self.close_array[:-1] = self.close_array[1:]
        self.volume_array[:-1] = self.volume_array[1:]
        self.open_interest_array[:-1] = self.open_interest_array[1:]

        if not self.log:
            self.open_array[-1] = bar.open_price
            self.high_array[-1] = bar.high_price
            self.low_array[-1] = bar.low_price
            self.close_array[-1] = bar.close_price
            self.volume_array[-1] = bar.volume
            self.open_interest_array[-1] = bar.open_interest

        else:
            self.open_array[-1] = np.log(bar.open_price)
            self.high_array[-1] = np.log(bar.high_price)
            self.low_array[-1] = np.log(bar.low_price)
            self.close_array[-1] = np.log(bar.close_price)
            self.volume_array[-1] = np.log(bar.volume)
            self.open_interest_array[-1] = np.log(bar.open_interest)

        self.last_bar = bar

    @property
    def open(self) -> np.ndarray:
        """
        Get open price time series.
        """
        return self.open_array

    @property
    def high(self) -> np.ndarray:
        """
        Get high price time series.
        """
        return self.high_array

    @property
    def low(self) -> np.ndarray:
        """
        Get low price time series.
        """
        return self.low_array

    @property
    def close(self) -> np.ndarray:
        """
        Get close price time series.
        """
        return self.close_array

    @property
    def volume(self) -> np.ndarray:
        """
        Get trading volume time series.
        """
        return self.volume_array

    @property
    def open_interest(self) -> np.ndarray:
        """
        Get trading volume time series.
        """
        return self.open_interest_array

    # ======================================================================================================================================================================================================== #
    # ======================================================================================================================================================================================================== #
    # Overlap Studies 重叠研究指标
    def sma(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Simple moving average. 简单移动平均线
        """
        if log:
            result = talib.SMA(np.log(self.close), timeperiod=n)
        else:
            result = talib.SMA(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def ema(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Exponential moving average.  指数移动平均线：趋向类指标，其构造原理是仍然对价格收盘价进行算术平均，并根据计算结果来进行分析，用于判断价格未来走势的变动趋势。
        """
        if log:
            result = talib.EMA(np.log(self.close), timeperiod=n)
        else:
            result = talib.EMA(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def dema(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Double Exponential Moving Average.  双指数移动平均线：两条指数移动平均线来产生趋势信号，较长期者用来识别趋势，较短期者用来选择时机。正是两条平均线及价格三者的相互作用，才共同产生了趋势信号。
        """
        if log:
            result = talib.DEMA(np.log(self.close), timeperiod=n)
        else:
            result = talib.DEMA(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def kama(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        KAMA. 考夫曼自适应移动平均线：短期均线贴近价格走势，灵敏度高，但会有很多噪声，产生虚假信号；长期均线在判断趋势上一般比较准确 ，但是长期均线有着严重滞后的问题。
        我们想得到这样的均线，当价格沿一个方向快速移动时，短期的移动 平均线是最合适的；当价格在横盘的过程中，长期移动平均线是合适的。
        """
        if log:
            result = talib.KAMA(np.log(self.close), timeperiod=n)
        else:
            result = talib.KAMA(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def wma(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Weighted Moving Average. 加权移动平均线
        """
        if log:
            result = talib.WMA(np.log(self.close), timeperiod=n)
        else:
            result = talib.WMA(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def ma(self, n: int, matype: int = 0, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Moving average.  移动平均线：matype: 0=SMA(默认), 1=EMA(指数移动平均线), 2=WMA(加权移动平均线), 3=DEMA(双指数移动平均线), 4=TEMA(三重指数移动平均线), 5=TRIMA, 6=KAMA(考夫曼自适应移动平均线), 7=MAMA, 8=T3(三重指数移动平均线)
        """
        if log:
            result = talib.MA(np.log(self.close), timeperiod=n, matype=matype)
        else:
            result = talib.MA(self.close, timeperiod=n, matype=matype)

        if array:
            return result
        return result[-1]

    def sar(self, acceleration: int = 0, maximum: int = 0, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Parabolic SAR.  抛物线指标：抛物线转向也称停损点转向，是利用抛物线方式，随时调整停损点位置以观察买卖点。由于停损点（又称转向点SAR）以弧形的方式移动，故称之为抛物线转向指标。
        """
        if log:
            result = talib.SAR(np.log(self.high), np.log(self.low), acceleration=acceleration, maximum=maximum)
        else:
            result = talib.SAR(self.high, self.low, acceleration=acceleration, maximum=maximum)

        if array:
            return result
        return result[-1]

    def boll(self, n: int, dev_up: float = 2, dev_dn: float = 2, array: bool = False, log: bool = False, matype: int = 0) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[float, float, float]]:
        """
        Bollinger Channel.  布林线指标：其利用统计原理，求出股价的标准差及其信赖区间，从而确定股价的波动范围及未来走势，利用波带显示股价的安全高低价位，因而也被称为布林带。
        """
        if log:
            upperband, middleband, lowerband = talib.BBANDS(np.log(self.close), timeperiod=n, nbdevup=dev_up, nbdevdn=dev_dn, matype=matype)
        else:
            upperband, middleband, lowerband = talib.BBANDS(self.close, timeperiod=n, nbdevup=dev_up, nbdevdn=dev_dn, matype=matype)

        if array:
            return upperband, middleband, lowerband
        return upperband[-1], middleband[-1], lowerband[-1]

    # ======================================================================================================================================================================================================== #
    # ======================================================================================================================================================================================================== #
    # Momentum Indicator 动量指标
    def apo(self, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Absolute Price Oscillator.
        """
        if log:
            result = talib.APO(np.log(self.close), fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
        else:
            result = talib.APO(self.close, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)

        if array:
            return result
        return result[-1]

    def cmo(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Chande Momentum Oscillator. 钱德动量摆动指标：与其他动量指标摆动指标如相对强弱指标（RSI）和随机指标（KDJ）不同，钱德动量指标在计算公式的分子中采用上涨日和下跌日的数据。 计算公式：CMO=（Su－Sd）*100/（Su+Sd）
        其中：Su是今日收盘价与昨日收盘价（上涨日）差值加总。若当日下跌，则增加值为0；Sd是今日收盘价与做日收盘价（下跌日）差值的绝对值加总。若当日上涨，则增加值为0。
        指标应用：本指标类似RSI指标。当本指标下穿-50水平时是买入信号，上穿+50水平是卖出信号。钱德动量摆动指标的取值介于-100和100之间。本指标也能给出良好的背离信号。当股票价格创出新低而本指标未能创出新低时，出现牛市背离；
        当股票价格创出新高而本指标未能创出新高时，当出现熊市背离时。我们可以用移动均值对该指标进行平滑。
        """
        if log:
            result = talib.CMO(np.log(self.close), timeperiod=n)
        else:
            result = talib.CMO(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def mom(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Momentum. 动量，上升动向值：
        """
        if log:
            result = talib.MOM(np.log(self.close), timeperiod=n)
        else:
            result = talib.MOM(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def ppo(self, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Percentage Price Oscillator. 价格震荡百分比指数：PPO标准设定和MACD设定非常相似：12,26,9和PPO，和MACD一样说明了两条移动平均线的差距，但是它们有一个差别是PPO是用百分比说明。
        """
        if log:
            result = talib.PPO(np.log(self.close), fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
        else:
            result = talib.PPO(self.close, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)

        if array:
            return result
        return result[-1]

    def roc(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Rate of change. 变动率指标：ROC是由当天的股价与一定的天数之前的某一天股价比较，其变动速度的大小,来反映股票市变动的快慢程度。
        ROC ＝ (当日收盘价－N天前的收盘价) ÷ N天前的收盘价 * 100% 
        指标研判：
        当ROC向下则表示弱势,以100为中心线,由中心线上下穿小于100时为卖出信号。
        当股价创新高时,ROC未能创新高,出现背离,表示头部形成。
        当股价创新低时,ROC未能创新低,出现背离,表示底部形成。
        """
        if log:
            result = talib.ROC(np.log(self.close), timeperiod=n)
        else:
            result = talib.ROC(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def rocr(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Rate of change ratio. (price/prevPrice)
        """
        if log:
            result = talib.ROCR(np.log(self.close), timeperiod=n)
        else:
            result = talib.ROCR(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def rocp(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Rate of change Percentage. (price-prevPrice)/prevPrice
        """
        if log:
            result = talib.ROCP(np.log(self.close), timeperiod=n)
        else:
            result = talib.ROCP(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def rocr_100(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Rate of change ratio 100 scale. (price/prevPrice)*100
        """
        if log:
            result = talib.ROCR100(np.log(self.close), timeperiod=n)
        else:
            result = talib.ROCR100(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def trix(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        1-day Rate-Of-Change (ROC) of a Triple Smooth EMA.
        """
        if log:
            result = talib.TRIX(np.log(self.close), timeperiod=n)
        else:
            result = talib.TRIX(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def cci(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Commodity Channel Index (CCI). 顺势指标：该指标用来测量股价脱离正常价格范围之变异性，正常波动范围在±100之间。属于超买超卖类指标中较特殊的一种，是专门对付极端行情的。在一般常态行情下，CCI指标不会发生作用，当CCI扫描到异常股价波动时，立求速战速决，胜负瞬间立即分晓，赌输了也必须立刻加速逃逸。
        指标应用
        1.当CCI指标曲线在+100线～-100线的常态区间里运行时,CCI指标参考意义不大，可以用KDJ等其它技术指标进行研判。
        2.当CCI指标曲线从上向下突破+100线而重新进入常态区间时，表明市场价格的上涨阶段可能结束，将进入一个比较长时间的震荡整理阶段，应及时平多做空。
        3.当CCI指标曲线从上向下突破-100线而进入另一个非常态区间（超卖区）时，表明市场价格的弱势状态已经形成，将进入一个比较长的寻底过程，可以持有空单等待更高利润。如果CCI指标曲线在超卖区运行了相当长的一段时间后开始掉头向上，表明价格的短期底部初步探明，可以少量建仓。CCI指标曲线在超卖区运行的时间越长，确认短期的底部的准确度越高。
        4.CCI指标曲线从下向上突破-100线而重新进入常态区间时，表明市场价格的探底阶段可能结束，有可能进入一个盘整阶段，可以逢低少量做多。
        5.CCI指标曲线从下向上突破+100线而进入非常态区间(超买区)时，表明市场价格已经脱离常态而进入强势状态，如果伴随较大的市场交投，应及时介入成功率将很大。
        6.CCI指标曲线从下向上突破+100线而进入非常态区间(超买区)后，只要CCI指标曲线一直朝上运行，表明价格依然保持强势可以继续持有待涨。但是，如果在远离+100线的地方开始掉头向下时，则表明市场价格的强势状态将可能难以维持，涨势可能转弱，应考虑卖出。如果前期的短期涨幅过高同时价格回落时交投活跃，则应该果断逢高卖出或做空。
        CCI主要是在超买和超卖区域发生作用，对急涨急跌的行情检测性相对准确。非常适用于股票、外汇、贵金属等市场的短期操作。
        计算方法：
        TP = (最高价 + 最低价 + 收盘价) ÷ 3
        MA = 最近n日每日TP之和÷n 
        MD = 最近n日 ABS(MATP - 每日TP)累计和 ÷ n
        CCI(n) = (TP－ MA) ÷MD ÷0.015
        """
        if log:
            result = talib.CCI(np.log(self.high), np.log(self.low), np.log(self.close), timeperiod=n)
        else:
            result = talib.CCI(self.high, self.low, self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def rsi(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Relative Strenght Index (RSI). 相对强弱指数：通过比较一段时期内的平均收盘涨数和平均收盘跌数来分析市场买沽盘的意向和实力，从而作出未来市场的走势。
        """
        if log:
            result = talib.RSI(np.log(self.close), timeperiod=n)
        else:
            result = talib.RSI(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def macd(self, fast_period: int, slow_period: int, signal_period: int, array: bool = False, log: bool = False) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[float, float, float]]:
        """
        Moving Average Convergence/Divergence. 平滑异同移动平均线：利用收盘价的短期（常用为12日）指数移动平均线与长期（常用为26日）指数移动平均线之间的聚合与分离状况，对买进、卖出时机作出研判的技术指标。
        """
        if log:
            macd, signal, hist = talib.MACD(np.log(self.close), fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
        else:
            macd, signal, hist = talib.MACD(self.close, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)

        if array:
            return macd, signal, hist
        return macd[-1], signal[-1], hist[-1]
    
    def bop(self, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Balance Of Power. 均势指标
        """
        if log:
            result = talib.BOP(np.log(self.open), np.log(self.high), np.log(self.low), np.log(self.close))
        else:
            result = talib.BOP(self.open, self.high, self.low, self.close)


        if array:
            return result
        return result[-1]

    def dx(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Directional Movement Index. 动向指标或趋向指标：通过分析股票价格在涨跌过程中买卖双方力量均衡点的变化情况，即多空双方的力量的变化受价格波动的影响而发生由均衡到失衡的循环过程，从而提供对趋势判断依据的一种技术指标。
        """
        if log:
            result = talib.DX(np.log(self.high), np.log(self.low), np.log(self.close), timeperiod=n)
        else:
            result = talib.DX(self.high, self.low, self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def adx(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Average Directional Movement Index. 平均趋向指标：使用ADX指标，指标判断盘整、振荡和单边趋势。
        指标应用：
        1、+DI与–DI表示多空相反的二个动向，当据此绘出的两条曲线彼此纠结相缠时，代表上涨力道与下跌力道相当，多空势均力敌。当 +DI与–DI彼此穿越时，由下往上的一方其力道开始压过由上往下的另一方，此时出现买卖讯号。
        2、ADX可作为趋势行情的判断依据，当行情明显朝多空任一方向进行时，ADX数值都会显著上升，趋势走强。若行情呈现盘整格局时，ADX会低于 +DI与–DI二条线。若ADX数值低于20，则不论DI表现如何，均显示市场没有明显趋势。
        3、ADX持续偏高时，代表“超买”（Overbought）或“超卖”（Oversold）的现象，行情反转的机会将增加，此时则不适宜顺势操作。当ADX数值从上升趋势转为下跌时，则代表行情即将反转；若ADX数值由下跌趋势转为上升时，行情将止跌回升。
        4、总言之，DMI指标包含4条线：+DI、-DI、ADX和ADXR。+DI代表买盘的强度、-DI代表卖盘的强度；ADX代表趋势的强度、ADXR则为ADX的移动平均。
        """
        if log:
            result = talib.ADX(np.log(self.high), np.log(self.low), np.log(self.close), timeperiod=n)
        else:
            result = talib.ADX(self.high, self.low, self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def adxr(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Average Directional Movement Index Rating. 平均趋向指数的趋向指数：使用ADXR指标判断ADX趋势，ADXR则为ADX的移动平均。
        """
        if log:
            result = talib.ADXR(np.log(self.high), np.log(self.low), np.log(self.close), timeperiod=n)
        else:
            result = talib.ADXR(self.high, self.low, self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def minus_di(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Minus Directional Indicator. DMI 中的DI指标，负方向指标，下升动向值：通过分析股票价格在涨跌过程中买卖双方力量均衡点的变化情况，即多空双方的力量的变化受价格波动的影响而发生由均衡到失衡的循环过程，从而提供对趋势判断依据的一种技术指标。
        """
        if log:
            result = talib.MINUS_DI(np.log(self.high), np.log(self.low), np.log(self.close), timeperiod=n)
        else:
            result = talib.MINUS_DI(self.high, self.low, self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def minus_dm(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Minus Directional Movement. DMI中的DM代表正趋向变动值，即上升动向值：通过分析股票价格在涨跌过程中买卖双方力量均衡点的变化情况，即多空双方的力量的变化受价格波动的影响而发生由均衡到失衡的循环过程，从而提供对趋势判断依据的一种技术指标。
        """
        if log:
            result = talib.MINUS_DM(np.log(self.high), np.log(self.low), timeperiod=n)
        else:
            result = talib.MINUS_DM(self.high, self.low, timeperiod=n)

        if array:
            return result
        return result[-1]

    def plus_di(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Plus Directional Indicator.
        """
        if log:
            result = talib.PLUS_DI(np.log(self.high), np.log(self.low), np.log(self.close), timeperiod=n)
        else:
            result = talib.PLUS_DI(self.high, self.low, self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def plus_dm(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Plus Directional Movement.
        """
        if log:
            result = talib.PLUS_DM(np.log(self.high), np.log(self.low), timeperiod=n)
        else:
            result = talib.PLUS_DM(self.high, self.low, timeperiod=n)

        if array:
            return result
        return result[-1]

    def willr(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Williams' %R. 威廉指标：WMS表示的是市场处于超买还是超卖状态。
        """
        if log:
            result = talib.WILLR(np.log(self.high), np.log(self.low), np.log(self.close), timeperiod=n)
        else:
            result = talib.WILLR(self.high, self.low, self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def ultosc(self, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Ultimate Oscillator. 终极波动指标：UOS是一种多方位功能的指标，除了趋势确认及超买超卖方面的作用之外，它的“突破”讯号不仅可以提供最适当的交易时机之外，更可以进一步加强指标的可靠度。
        """
        if log:
            result = talib.ULTOSC(np.log(self.high), np.log(self.low), np.log(self.close))
        else:
            result = talib.ULTOSC(self.high, self.low, self.close)

        if array:
            return result
        return result[-1]

    def keltner(self, n: int, dev: float, array: bool = False, log: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
        """
        Keltner Channel. 金肯特纳通道
        """
        if log:
            mid = self.sma(n, array, log)
            atr = self.atr(n, array, log)
        else:
            mid = self.sma(n, array)
            atr = self.atr(n, array)

        up = mid + atr * dev
        down = mid - atr * dev

        return up, down

    def donchian(self, n: int, array: bool = False, log: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
        """
        Donchian Channel. 唐奇安通道，以N周期内的最高价和最低价作为通道上下轨
        """
        if log:
            up = talib.MAX(np.log(self.high), n)
            down = talib.MIN(np.log(self.low), n)
        else:
            up = talib.MAX(self.high, n)
            down = talib.MIN(self.low, n)

        if array:
            return up, down
        return up[-1], down[-1]

    def donchian_close(self, n: int, array: bool = False, log: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
        """
        Donchian Channel. 唐奇安通道变种，以N周期内的收盘价最高最低点作为通道上下轨
        """
        if log:
            close = np.log(self.close)
            up = talib.MAX(close, n)
            down = talib.MIN(close, n)
        else:
            up = talib.MAX(self.close, n)
            down = talib.MIN(self.close, n)

        if array:
            return up, down
        return up[-1], down[-1]

    def aroon(self, n: int, array: bool = False, log: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
        """
        Aroon indicator. 阿隆指标：通过计算自价格达到近期最高值和最低值以来所经过的期间数，阿隆指标帮助你预测价格趋势到趋势区域（或者反过来，从趋势区域到趋势）的变化。
        计算公式：
        Aroon(上升)=[(计算期天数-最高价后的天数)/计算期天数]*100
        Aroon(下降)=[(计算期天数-最低价后的天数)/计算期天数]*100
        """
        if log:
            aroon_up, aroon_down = talib.AROON(np.log(self.high), np.log(self.low), timeperiod=n)
        else:
            aroon_up, aroon_down = talib.AROON(self.high, self.low, timeperiod=n)

        if array:
            return aroon_up, aroon_down
        return aroon_up[-1], aroon_down[-1]

    def aroonosc(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Aroon Oscillator. 阿隆振荡
        """
        if log:
            result = talib.AROONOSC(np.log(self.high), np.log(self.low), timeperiod=n)
        else:
            result = talib.AROONOSC(self.high, self.low, timeperiod=n)

        if array:
            return result
        return result[-1]

    def mfi(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Money Flow Index. 资金流量指标
        """
        if log:
            result = talib.MFI(np.log(self.high), np.log(self.low), np.log(self.close), self.volume, timeperiod=n)
        else:
            result = talib.MFI(self.high, self.low, self.close, self.volume, timeperiod=n)

        if array:
            return result
        return result[-1]

    # ======================================================================================================================================================================================================== #
    # ======================================================================================================================================================================================================== #
    # Volume Indicators 成交量指标
    def ad(self, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Accumulation/Distribution Line. 量价指标（累积/派发线）：平衡交易量指标，以当日的收盘价位来估算成交流量，用于估定一段时间内该证券累积的资金流量。
        计算公式：
        多空对比 = [（收盘价- 最低价） - （最高价 - 收盘价）] / （最高价 - 最低价)
        若最高价等于最低价： 多空对比 = （收盘价 / 昨收盘） - 1
        研判：
        1、A/D测量资金流向，向上的A/D表明买方占优势，而向下的A/D表明卖方占优势
        2、A/D与价格的背离可视为买卖信号，即底背离考虑买入，顶背离考虑卖出
        3、应当注意A/D忽略了缺口的影响，事实上，跳空缺口的意义是不能轻易忽略的
        A/D指标无需设置参数，但在应用时，可结合指标的均线进行分析
        """
        if log:
            result = talib.AD(np.log(self.high), np.log(self.low), np.log(self.close), self.volume)
        else:
            result = talib.AD(self.high, self.low, self.close, self.volume)

        if array:
            return result
        return result[-1]
    
    def obv(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        On Balance Volume 能量潮指标：通过统计成交量变动的趋势推测股价趋势。
        计算公式：以某日为基期，逐日累计每日上市股票总成交量，若隔日指数或股票上涨 ，则基期OBV加上本日成交量为本日OBV。隔日指数或股票下跌， 则基期OBV减去本日成交量为本日OBV
        研判：
        1、以“N”字型为波动单位，一浪高于一浪称“上升潮”，下跌称“跌潮”；上升潮买进，跌潮卖出
        2、须配合K线图走势
        3、用多空比率净额法进行修正，但不知TA-Lib采用哪种方法
        计算公式： 多空比率净额= [（收盘价－最低价）－（最高价-收盘价）] ÷（ 最高价－最低价）× 成交量
        """
        if log:
            result = talib.OBV(np.log(self.close), self.volume)
        else:
            result = talib.OBV(self.close, self.volume)

        if array:
            return result
        return result[-1]

    def adosc(self, fastperiod:int = 3, slowperiod:int = 10, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        ADOSC. 震荡指标：将资金流动情况与价格行为相对比，检测市场中资金流入和流出的情况。
        """
        if log:
            result = talib.ADOSC(np.log(self.high), np.log(self.low), np.log(self.close), self.volume, fastperiod=fastperiod, slowperiod=slowperiod)
        else:
            result = talib.ADOSC(self.high, self.low, self.close, self.volume, fastperiod=fastperiod, slowperiod=slowperiod)

        if array:
            return result
        return result[-1]

    # ======================================================================================================================================================================================================== #
    # ======================================================================================================================================================================================================== #
    # Volatility Indicator 波动率指标
    def trange(self, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        TRANGE. 真实波动幅度 = max(当日最高价, 昨日收盘价) − min(当日最低价, 昨日收盘价)
        """
        if log:
            result = talib.TRANGE(np.log(self.high), np.log(self.low), np.log(self.close))
        else:
            result = talib.TRANGE(self.high, self.low, self.close)

        if array:
            return result
        return result[-1]

    def atr(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Average True Range (ATR). 平均真实波动幅度：真实波动幅度的 N 日 指数移动平均数。
        """
        if log:
            result = talib.ATR(np.log(self.high), np.log(self.low), np.log(self.close), timeperiod=n)
        else:
            result = talib.ATR(self.high, self.low, self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def natr(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Normalized Average True Range. 归一化波动幅度均值
        """
        if log:
            result = talib.NATR(np.log(self.high), np.log(self.low), np.log(self.close), timeperiod=n)
        else:
            result = talib.NATR(self.high, self.low, self.close, timeperiod=n)

        if array:
            return result
        return result[-1]
    # ======================================================================================================================================================================================================== #
    # ======================================================================================================================================================================================================== #
    # Statistic 统计学指标
    def std(self, n: int, dev: int = 1, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Standard deviation. 标准偏差：量度数据分布的分散程度之标准，用以衡量数据值偏离算术平均值的程度。标准偏差越小，这些值偏离平均值就越少，反之亦然。标准偏差的大小可通过标准偏差与平均值的倍率关系来衡量。
        """
        if log:
            result = talib.STDDEV(np.log(self.close), timeperiod=n, nbdev=dev)
        else:
            result = talib.STDDEV(self.close, timeperiod=n, nbdev=dev)

        if array:
            return result
        return result[-1]

    def var(self, n: int, dev: int = 1, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        VAR. 方差：用来计算每一个变量（观察值）与总体均数之间的差异。为避免出现离均差总和为零，离均差平方和受样本含量的影响，统计学采用平均离均差平方和来描述变量的变异程度。
        """
        if log:
            result = talib.VAR(np.log(self.close), timeperiod=n, nbdev=dev)
        else:
            result = talib.VAR(self.close, timeperiod=n, nbdev=dev)

        if array:
            return result
        return result[-1]

    def linearreg(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Linear Regression. 直线回归方程
        """
        if log:
            result = talib.LINEARREG(np.log(self.close), timeperiod=n)
        else:
            result = talib.LINEARREG(self.close, timeperiod=n)
 
        if array:
            return result
        return result[-1]

    def intercept(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Linear Regression Intercept. 线性回归截距
        """
        if log:
            result = talib.LINEARREG_INTERCEPT(np.log(self.close), timeperiod=n)
        else:
            result = talib.LINEARREG_INTERCEPT(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def slope(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Linear Regression Slope. 线性回归斜率
        """
        if log:
            result = talib.LINEARREG_SLOPE(np.log(self.close), timeperiod=n)
        else:
            result = talib.LINEARREG_SLOPE(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def angle(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Linear Regression Angle. 线性回归角度
        """
        if log:
            result = talib.LINEARREG_ANGLE(np.log(self.close), timeperiod=n)
        else:
            result = talib.LINEARREG_ANGLE(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def tsf(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Time Series Forecast. 时间序列预测：一种历史资料延伸预测，也称历史引伸预测法。是以时间数列所能反映的社会经济现象的发展过程和规律性，进行引伸外推，预测其发展趋势的方法。
        """
        if log:
            result = talib.TSF(np.log(self.close), timeperiod=n)
        else:
            result = talib.TSF(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def beta(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Beta. β系数(贝塔系数)：一种风险指数，用来衡量个别股票或 股票基金相对于整个股市的价格波动情况 贝塔系数衡量股票收益相对于业绩评价基准收益的总体波动性，是一个相对指标。
        β 越高，意味着股票相对于业绩评价基准的波动性越大。 β 大于 1 ， 则股票的波动性大于业绩评价基准的波动性。
        """
        if log:
            result = talib.BETA(np.log(self.high), np.log(self.low), timeperiod=n)
        else:
            result = talib.BETA(self.high, self.low, timeperiod=n)

        if array:
            return result
        return result[-1]

    def correl(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Pearson's Correlation Coefficient. 皮尔逊相关系数：用于度量两个变量X和Y之间的相关（线性相关），其值介于-1与1之间皮尔逊相关系数是一种度量两个变量间相关程度的方法。
        它是一个介于 1 和 -1 之间的值， 其中，1 表示变量完全正相关， 0 表示无关，-1 表示完全负相关。
        """
        if log:
            result = talib.CORREL(np.log(self.high), np.log(self.low), timeperiod=n)
        else:
            result = talib.CORREL(self.high, self.low, timeperiod=n)

        if array:
            return result
        return result[-1]

    # ======================================================================================================================================================================================================== #
    # ======================================================================================================================================================================================================== #
    # Pattern Recognition 形态识别
    def CDL2CROWS(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Two Crows 两只乌鸦：三日K线模式，第一天长阳，第二天高开收阴，第三天再次高开继续收阴， 收盘比前一日收盘价低，预示股价下跌。
        """
        result = talib.CDL2CROWS(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDL3BLACKCROWS(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Three Black Crows 三只乌鸦：三日K线模式，连续三根阴线，每日收盘价都下跌且接近最低价， 每日开盘价都在上根K线实体内，预示股价下跌。
        """
        result = talib.CDL3BLACKCROWS(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDL3INSIDE(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Three-Line Strike 三线打击：四日K线模式，前三根阳线，每日收盘价都比前一日高， 开盘价在前一日实体内，第四日市场高开，收盘价低于第一日开盘价，预示股价下跌。
        """
        result = talib.CDL3INSIDE(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDL3OUTSIDE(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Three Outside Up/Down 三外部上涨和下跌：三日K线模式，与三内部上涨和下跌类似，K线为阴阳阳，但第一日与第二日的K线形态相反， 以三外部上涨为例，第一日K线在第二日K线内部，预示着股价上涨。
        """
        result = talib.CDL3OUTSIDE(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDL3STARSINSOUTH(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Three Stars In The South 南方三星：三日K线模式，与大敌当前相反，三日K线皆阴，第一日有长下影线， 第二日与第一日类似，K线整体小于第一日，第三日无下影线实体信号， 成交价格都在第一日振幅之内，预示下跌趋势反转，股价上升。
        """
        result = talib.CDL3STARSINSOUTH(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDL3WHITESOLDIERS(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Three Advancing White Soldiers 三个白兵：三日K线模式，三日K线皆阳， 每日收盘价变高且接近最高价，开盘价在前一日实体上半部，预示股价上升。
        """
        result = talib.CDL3WHITESOLDIERS(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLABANDONEDBABY(self, penetration: int = 0, array: bool = False) -> Union[float, np.ndarray]:
        """
        Abandoned Baby 弃婴：三日K线模式，第二日价格跳空且收十字星（开盘价与收盘价接近， 最高价最低价相差不大），预示趋势反转，发生在顶部下跌，底部上涨。
        """
        result = talib.CDLABANDONEDBABY(self.open, self.high, self.low, self.close, penetration=penetration)
        if array:
            return result
        return result[-1]

    def CDLADVANCEBLOCK(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Advance Block 大敌当前：三日K线模式，三日都收阳，每日收盘价都比前一日高， 开盘价都在前一日实体以内，实体变短，上影线变长。
        """
        result = talib.CDLADVANCEBLOCK(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]
        
    def CDLBELTHOLD(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Belt-hold 捉腰带线：两日K线模式，下跌趋势中，第一日阴线， 第二日开盘价为最低价，阳线，收盘价接近最高价，预示价格上涨。
        """
        result = talib.CDLBELTHOLD(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLBREAKAWAY(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Breakaway 脱离：五日K线模式，以看涨脱离为例，下跌趋势中，第一日长阴线，第二日跳空阴线，延续趋势开始震荡， 第五日长阳线，收盘价在第一天收盘价与第二天开盘价之间，预示价格上涨。
        """
        result = talib.CDLBREAKAWAY(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLCLOSINGMARUBOZU(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Closing Marubozu 收盘缺影线：一日K线模式，以阳线为例，最低价低于开盘价，收盘价等于最高价， 预示着趋势持续。
        """
        result = talib.CDLCLOSINGMARUBOZU(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLCONCEALBABYSWALL(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Concealing Baby Swallow 藏婴吞没：四日K线模式，下跌趋势中，前两日阴线无影线 ，第二日开盘、收盘价皆低于第二日，第三日倒锤头， 第四日开盘价高于前一日最高价，收盘价低于前一日最低价，预示着底部反转。
        """
        result = talib.CDLCONCEALBABYSWALL(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLCOUNTERATTACK(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Counterattack 反击线：二日K线模式，与分离线类似。
        """
        result = talib.CDLCOUNTERATTACK(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLDARKCLOUDCOVER(self, penetration: int = 0, array: bool = False) -> Union[float, np.ndarray]:
        """
        Dark Cloud Cover 乌云压顶：二日K线模式，第一日长阳，第二日开盘价高于前一日最高价， 收盘价处于前一日实体中部以下，预示着股价下跌。
        """
        result = talib.CDLDARKCLOUDCOVER(self.open, self.high, self.low, self.close, penetration=penetration)
        if array:
            return result
        return result[-1]

    def CDLDOJI(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Doji 十字：一日K线模式，开盘价与收盘价基本相同。
        """
        result = talib.CDLDOJI(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLDOJISTAR(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Doji Star 十字星：一日K线模式，开盘价与收盘价基本相同，上下影线不会很长，预示着当前趋势反转。
        """
        result = talib.CDLDOJISTAR(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLDRAGONFLYDOJI(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Dragonfly Doji 蜻蜓十字/T形十字：一日K线模式，开盘后价格一路走低， 之后收复，收盘价与开盘价相同，预示趋势反转。
        """
        result = talib.CDLDRAGONFLYDOJI(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLENGULFING(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Engulfing Pattern 吞噬模式：两日K线模式，分多头吞噬和空头吞噬，以多头吞噬为例，第一日为阴线， 第二日阳线，第一日的开盘价和收盘价在第二日开盘价收盘价之内，但不能完全相同。
        """
        result = talib.CDLENGULFING(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLEVENINGDOJISTAR(self, penetration: int = 0, array: bool = False) -> Union[float, np.ndarray]:
        """
        Evening Doji Star 十字暮星：三日K线模式，基本模式为暮星，第二日收盘价和开盘价相同，预示顶部反转。
        """
        result = talib.CDLEVENINGDOJISTAR(self.open, self.high, self.low, self.close, penetration=penetration)
        if array:
            return result
        return result[-1]

    def CDLEVENINGSTAR(self, penetration: int = 0, array: bool = False) -> Union[float, np.ndarray]:
        """
        Evening Star 暮星：三日K线模式，与晨星相反，上升趋势中, 第一日阳线，第二日价格振幅较小，第三日阴线，预示顶部反转。
        """
        result = talib.CDLEVENINGSTAR(self.open, self.high, self.low, self.close, penetration=penetration)
        if array:
            return result
        return result[-1]

    def CDLGAPSIDESIDEWHITE(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Up/Down-gap side-by-side white lines 向上/下跳空并列阳线：二日K线模式，上升趋势向上跳空，下跌趋势向下跳空, 第一日与第二日有相同开盘价，实体长度差不多，则趋势持续。
        """
        result = talib.CDLGAPSIDESIDEWHITE(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLGRAVESTONEDOJI(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Gravestone Doji 墓碑十字/倒T十字：一日K线模式，开盘价与收盘价相同，上影线长，无下影线，预示底部反转。
        """
        result = talib.CDLGRAVESTONEDOJI(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLHAMMER(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Hammer 锤头：一日K线模式，实体较短，无上影线， 下影线大于实体长度两倍，处于下跌趋势底部，预示反转。
        """
        result = talib.CDLHAMMER(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLHANGINGMAN(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Hanging Man 上吊线：一日K线模式，形状与锤子类似，处于上升趋势的顶部，预示着趋势反转。
        """
        result = talib.CDLHANGINGMAN(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLHARAMI(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Harami Pattern 母子线：二日K线模式，分多头母子与空头母子，两者相反，以多头母子为例，在下跌趋势中，第一日K线长阴， 第二日开盘价收盘价在第一日价格振幅之内，为阳线，预示趋势反转，股价上升。
        """
        result = talib.CDLHARAMI(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLHARAMICROSS(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Harami Cross Pattern 十字孕线：二日K线模式，与母子县类似，若第二日K线是十字线， 便称为十字孕线，预示着趋势反转。
        """
        result = talib.CDLHARAMICROSS(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLHIGHWAVE(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        High-Wave Candle 风高浪大线：三日K线模式，具有极长的上/下影线与短的实体，预示着趋势反转。
        """
        result = talib.CDLHIGHWAVE(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLHIKKAKE(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Hikkake Pattern 陷阱：三日K线模式，与母子类似，第二日价格在前一日实体范围内, 第三日收盘价高于前两日，反转失败，趋势继续。
        """
        result = talib.CDLHIKKAKE(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLHIKKAKEMOD(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Modified Hikkake Pattern 修正陷阱：三日K线模式，与陷阱类似，上升趋势中，第三日跳空高开； 下跌趋势中，第三日跳空低开，反转失败，趋势继续。
        """
        result = talib.CDLHIKKAKEMOD(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLHOMINGPIGEON(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Homing Pigeon 家鸽：二日K线模式，与母子线类似，不同的的是二日K线颜色相同， 第二日最高价、最低价都在第一日实体之内，预示着趋势反转。
        """
        result = talib.CDLHOMINGPIGEON(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLIDENTICAL3CROWS(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Identical Three Crows 三胞胎乌鸦：三日K线模式，上涨趋势中，三日都为阴线，长度大致相等， 每日开盘价等于前一日收盘价，收盘价接近当日最低价，预示价格下跌。
        """
        result = talib.CDLIDENTICAL3CROWS(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLINNECK(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        In-Neck Pattern 颈内线：二日K线模式，下跌趋势中，第一日长阴线， 第二日开盘价较低，收盘价略高于第一日收盘价，阳线，实体较短，预示着下跌继续。
        """
        result = talib.CDLINNECK(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLINVERTEDHAMMER(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Inverted Hammer 倒锤头：一日K线模式，上影线较长，长度为实体2倍以上， 无下影线，在下跌趋势底部，预示着趋势反转。
        """
        result = talib.CDLINVERTEDHAMMER(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLKICKING(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Kicking 反冲形态：二日K线模式，与分离线类似，两日K线为秃线，颜色相反，存在跳空缺口。
        """
        result = talib.CDLKICKING(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLKICKINGBYLENGTH(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Kicking - bull/bear determined by the longer marubozu 由较长缺影线决定的反冲形态：二日K线模式，与反冲形态类似，较长缺影线决定价格的涨跌。
        """
        result = talib.CDLKICKINGBYLENGTH(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLLADDERBOTTOM(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Ladder Bottom 梯底：五日K线模式，下跌趋势中，前三日阴线， 开盘价与收盘价皆低于前一日开盘、收盘价，第四日倒锤头，第五日开盘价高于前一日开盘价， 阳线，收盘价高于前几日价格振幅，预示着底部反转。
        """
        result = talib.CDLLADDERBOTTOM(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLLONGLEGGEDDOJI(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Long Legged Doji 长脚十字：一日K线模式，开盘价与收盘价相同居当日价格中部，上下影线长， 表达市场不确定性。
        """
        result = talib.CDLLONGLEGGEDDOJI(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLLONGLINE(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Long Line Candle 长蜡烛：一日K线模式，K线实体长，无上下影线。
        """
        result = talib.CDLLONGLINE(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLMARUBOZU(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Marubozu 光头光脚/缺影线：一日K线模式，上下两头都没有影线的实体， 阴线预示着熊市持续或者牛市反转，阳线相反。
        """
        result = talib.CDLMARUBOZU(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLMATCHINGLOW(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Matching Low 相同低价：二日K线模式，下跌趋势中，第一日长阴线， 第二日阴线，收盘价与前一日相同，预示底部确认，该价格为支撑位。
        """
        result = talib.CDLMATCHINGLOW(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLMATHOLD(self, penetration: int = 0, array: bool = False) -> Union[float, np.ndarray]:
        """
        Mat Hold 铺垫：五日K线模式，上涨趋势中，第一日阳线，第二日跳空高开影线， 第三、四日短实体影线，第五日阳线，收盘价高于前四日，预示趋势持续。
        """
        result = talib.CDLMATHOLD(self.open, self.high, self.low, self.close, penetration=penetration)
        if array:
            return result
        return result[-1]

    def CDLMORNINGDOJISTAR(self, penetration: int = 0, array: bool = False) -> Union[float, np.ndarray]:
        """
        Morning Doji Star 十字晨星：三日K线模式， 基本模式为晨星，第二日K线为十字星，预示底部反转。
        """
        result = talib.CDLMORNINGDOJISTAR(self.open, self.high, self.low, self.close, penetration=penetration)
        if array:
            return result
        return result[-1]

    def CDLMORNINGSTAR(self, penetration: int = 0, array: bool = False) -> Union[float, np.ndarray]:
        """
        Morning Star 晨星：三日K线模式，下跌趋势，第一日阴线， 第二日价格振幅较小，第三天阳线，预示底部反转。
        """
        result = talib.CDLMORNINGSTAR(self.open, self.high, self.low, self.close, penetration=penetration)
        if array:
            return result
        return result[-1]

    def CDLONNECK(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        On-Neck Pattern 颈上线：二日K线模式，下跌趋势中，第一日长阴线，第二日开盘价较低， 收盘价与前一日最低价相同，阳线，实体较短，预示着延续下跌趋势。
        """
        result = talib.CDLONNECK(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLPIERCING(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Piercing Pattern 刺透形态：两日K线模式，下跌趋势中，第一日阴线，第二日开盘价低于前一日最低价， 收盘价处在第一日实体上部，预示着底部反转。
        """
        result = talib.CDLPIERCING(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLRICKSHAWMAN(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Rickshaw Man 黄包车夫：一日K线模式，与长腿十字线类似， 若实体正好处于价格振幅中点，称为黄包车夫。
        """
        result = talib.CDLRICKSHAWMAN(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLRISEFALL3METHODS(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Rising/Falling Three Methods 上升/下降三法：五日K线模式，以上升三法为例，上涨趋势中， 第一日长阳线，中间三日价格在第一日范围内小幅震荡， 第五日长阳线，收盘价高于第一日收盘价，预示股价上升。
        """
        result = talib.CDLRISEFALL3METHODS(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLSEPARATINGLINES(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Shooting Star 射击之星：一日K线模式，上影线至少为实体长度两倍， 没有下影线，预示着股价下跌。
        """
        result = talib.CDLSEPARATINGLINES(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLSHORTLINE(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Short Line Candle 短蜡烛：一日K线模式，实体短，无上下影线。
        """
        result = talib.CDLSHORTLINE(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLSPINNINGTOP(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Spinning Top 纺锤：一日K线，实体小。
        """
        result = talib.CDLSPINNINGTOP(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLSTALLEDPATTERN(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Stalled Pattern 停顿形态：三日K线模式，上涨趋势中，第二日长阳线， 第三日开盘于前一日收盘价附近，短阳线，预示着上涨结束。
        """
        result = talib.CDLSTALLEDPATTERN(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLSTICKSANDWICH(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Stick Sandwich 条形三明治：三日K线模式，第一日长阴线，第二日阳线，开盘价高于前一日收盘价， 第三日开盘价高于前两日最高价，收盘价于第一日收盘价相同。
        """
        result = talib.CDLSTICKSANDWICH(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLTAKURI(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Takuri (Dragonfly Doji with very long lower shadow) 探水竿：一日K线模式，大致与蜻蜓十字相同，下影线长度长。
        """
        result = talib.CDLTAKURI(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLTASUKIGAP(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Tasuki Gap 跳空并列阴阳线：三日K线模式，分上涨和下跌，以上升为例， 前两日阳线，第二日跳空，第三日阴线，收盘价于缺口中，上升趋势持续。
        """
        result = talib.CDLTASUKIGAP(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLTHRUSTING(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Thrusting Pattern 插入：二日K线模式，与颈上线类似，下跌趋势中，第一日长阴线，第二日开盘价跳空， 收盘价略低于前一日实体中部，与颈上线相比实体较长，预示着趋势持续。
        """
        result = talib.CDLTHRUSTING(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLTRISTAR(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Tristar Pattern 三星：三日K线模式，由三个十字组成， 第二日十字必须高于或者低于第一日和第三日，预示着反转。
        """
        result = talib.CDLTRISTAR(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLUNIQUE3RIVER(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Unique 3 River 奇特三河床：三日K线模式，下跌趋势中，第一日长阴线，第二日为锤头，最低价创新低，第三日开盘价低于第二日收盘价，收阳线， 收盘价不高于第二日收盘价，预示着反转，第二日下影线越长可能性越大。
        """
        result = talib.CDLUNIQUE3RIVER(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLUPSIDEGAP2CROWS(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        TUpside Gap Two Crows 向上跳空的两只乌鸦：三日K线模式，第一日阳线，第二日跳空以高于第一日最高价开盘， 收阴线，第三日开盘价高于第二日，收阴线，与第一日比仍有缺口。
        """
        result = talib.CDLUPSIDEGAP2CROWS(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def CDLXSIDEGAP3METHODS(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Upside/Downside Gap Three Methods 上升/下降跳空三法：五日K线模式，以上升跳空三法为例，上涨趋势中，第一日长阳线，第二日短阳线，第三日跳空阳线，第四日阴线，开盘价与收盘价于前两日实体内， 第五日长阳线，收盘价高于第一日收盘价，预示股价上升。
        """
        result = talib.CDLXSIDEGAP3METHODS(self.open, self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def candle_pattern_recognition(self):
        pass

    # ======================================================================================================================================================================================================== #
    # ======================================================================================================================================================================================================== #
    # Math Operators 数学运算
    def ADD(self, price_1, price_2, array: bool = False):
        """
        Vector Arithmetic Add. 向量加法运算
        """
        result = talib.ADD(price_1, price_2)
        if array:
            return result
        return result[-1]

    def SUB(self, price_1, price_2, array: bool = False):
        """
        Vector Arithmetic Substraction. 向量减法运算
        """
        result = talib.SUB(price_1, price_2)
        if array:
            return result
        return result[-1]

    def MULT(self, price_1, price_2, array: bool = False):
        """
        Vector Arithmetic Mult. 向量乘法运算
        """
        result = talib.MULT(price_1, price_2)
        if array:
            return result
        return result[-1]

    def DIV(self, price_1, price_2, array: bool = False):
        """
        Vector Arithmetic Div. 向量除法运算
        """
        result = talib.DIV(price_1, price_2)
        if array:
            return result
        return result[-1]

    def SUM(self, price, n, array: bool = False):
        """
        Summation. 周期内求和
        """
        result = talib.SUM(price, timeperiod=n)
        if array:
            return result
        return result[-1]

    def MAX(self, price, n, array: bool = False):
        """
        Highest value over a specified period. 周期内最大值（未满足周期返回nan）
        """
        result = talib.MAX(price, timeperiod=n)
        if array:
            return result
        return result[-1]

    def MIN(self, price, n, array: bool = False):
        """
        Lowest value over a specified period. 周期内最小值 （未满足周期返回nan）
        """
        result = talib.MIN(price, timeperiod=n)
        if array:
            return result
        return result[-1]

    def MAXINDEX(self, price, n, array: bool = False):
        """
        Index of highest value over a specified period. 周期内最大值的索引，返回整数，索引为倒数第几个
        """
        result = talib.MAXINDEX(price, timeperiod=n)
        if array:
            return result - self.size
        return result[-1] - self.size

    def MININDEX(self, price, n, array: bool = False):
        """
        Index of lowest value over a specified period. 周期内最小值的索引，返回整数
        """
        result = talib.MININDEX(price, timeperiod=n)
        if array:
            return result - self.size
        return result[-1] - self.size

    def MINMAX(self, price, n, array: bool = False):
        """
        Lowest and highest values over a specified period. 周期内最小值和最大值，返回元组系列
        """
        min_, max_ = talib.MINMAX(price, timeperiod=n)
        if array:
            return min_, max_
        return min_[-1], max_[-1]

    def MINMAXINDEX(self, price, n, array: bool = False):
        """
        Indexes of lowest and highest values over a specified period. 周期内最小值和最大值索引，返回元组整数
        """
        minidx, maxidx = talib.MINMAXINDEX(price, timeperiod=n)
        if array:
            return minidx - self.size, maxidx - self.size
        return minidx[-1] - self.size, maxidx - self.size

    # ======================================================================================================================================================================================================== #
    # ======================================================================================================================================================================================================== #
    # 丰富技术指标
    def bias(self, n: int, array: bool = False, log: bool = False, matype = 0) -> Union[float, np.ndarray]:
        """
        BIAS. 乖离率指标 (乖离率=[(当日收盘价-N日平均价)/N日平均价]*100%)
        乖离率（BIAS）简称Y值也叫偏离率，是反映一定时期内股价与其移动平均数偏离程度的指标。
        """
        if log:
            MA_array = talib.MA(np.log(self.close), timeperiod=n, matype=matype)
        else:
            MA_array = talib.MA(self.close, timeperiod=n, matype=matype)
        
        result = ((self.close - MA_array) / MA_array) * 100
        if array:
            return result
        return result[-1]

    def kdj(self, array: bool = False, fastk_period=9, slowk_period=3, slowd_period=3, matype=1) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[float, float, float]]:
        """
        KDJ随机指标，Stochastic (Momentum Indicators)
        指标计算方法：
        N日RSV = (N日收盘价 - N日内最低价) ÷ (N日内最高价-N日内最低价) × 100 　
        当日K值 = 2/3前1日K值 + 1/3当日RSV；　　
        当日D值 = 2/3前1日D值 + 1/3当日K值； 　　
        当日J值 = 3当日K值 - 2当日D值。            
        若无前一日K 值与D值，则可分别用50来代替。
        """
        df = pd.DataFrame({"high":self.high, "low":self.low})
        hv = df.high.rolling(fastk_period).max().values
        lv = df.low.rolling(fastk_period).min().values
        rsv = np.where(hv == lv, 0, (self.close - lv) / (hv - lv) * 100)
        k = talib.MA(rsv, timeperiod=slowk_period, matype=matype)
        d = talib.MA(k, timeperiod=slowd_period, matype=matype)

        # k, d = talib.STOCH(self.high, self.low, self.close, fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=matype, slowd_period=slowd_period, slowd_matype=matype)
        
        j = 3 * k - 2 * d

        if array:
            return k, d, j
        return k[-1], d[-1], j[-1]

    def countif(self, condition: bool, con_key: str, n: int = 1, size: int = 500) -> int:
        """获取最近N周期满足给定条件的次数(每一次回调都需执行判断，con_key须唯一，可通过countif_dict调取)"""
        self.countif_dict[con_key].append(condition * 1)
        list_len = len(self.countif_dict[con_key])

        if list_len > size:
            self.countif_dict[con_key].pop(0)

        if list_len >= n:
            return sum(self.countif_dict[con_key][-n:])
        else:
            return 0

    def CrossOver(self, PriceArrray_1: np.ndarray, Price_2:  Union[int, float, np.ndarray]) -> bool:
        """判断是否上穿"""
        if isinstance(Price_2, (int, float)):
            if PriceArrray_1[-2] < Price_2 and PriceArrray_1[-1] >= Price_2:
                return True
            else:
                return False
        else:
            if PriceArrray_1[-2] < Price_2[-2] and PriceArrray_1[-1] >= Price_2[-1]:
                return True
            else:
                return False

    def CrossUnder(self, PriceArrray_1: np.ndarray, Price_2: Union[int, float, np.ndarray]) -> bool:
        """判断是否下破"""
        if isinstance(Price_2, (int, float)):
            if PriceArrray_1[-2] > Price_2 and PriceArrray_1[-1] <= Price_2:
                return True
            else:
                return False
        else:
            if PriceArrray_1[-2] > Price_2[-2] and PriceArrray_1[-1] <= Price_2[-1]:
                return True
            else:
                return False

    def RealDate(self, data: Union[TickData, BarData]):
        """"""
        day_offset = timedelta(days=0)
        if data.datetime.hour >= 18:
            if data.datetime.isoweekday() == 5:         # 周五晚上
                day_offset = timedelta(days=3)
            elif data.datetime.isoweekday() == 6:       # 周六晚上
                day_offset = timedelta(days=2)
            elif data.datetime.isoweekday() == 7:       # 周日晚上
                day_offset = timedelta(days=1)

        else:
            if data.datetime.isoweekday() == 6:         # 周六
                day_offset = timedelta(days=2)
            elif data.datetime.isoweekday() == 7:       # 周日
                day_offset = timedelta(days=1)
                
        dt = data.datetime + day_offset

        return dt.date()

    
    # ======================================================================================================================================================================================================== #
    # ======================================================================================================================================================================================================== #
    def ohlc_average(self, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """开高低收四个价格的平均价"""
        if log:
            result = (np.log(self.open) + np.log(self.high) + np.log(self.low) + np.log(self.close)) / 4
        else:
            result = (self.open + self.high + self.low + self.close) / 4

        if array:
            return result
        return result[-1]

    def c_sub_o(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        (Close - Open)差价序列（区分阴阳的K线实体大小）
        """
        result = self.close - self.open

        if array:
            return result
        return result[-1]
        
    def abs_c_sub_o(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        (Close - Open)的绝对值差价序列（K线实体大小）
        """
        result = abs(self.close - self.open)

        if array:
            return result
        return result[-1]

    def upper_shadow_line(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        (High - max(Open, Close))的差价序列（K线上影线大小）
        """
        result = self.high - self.max_o_or_c(array=True)

        if array:
            return result
        return result[-1]

    def lower_shadow_line(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        (High - max(Open, Close))的差价序列（K线上影线大小）
        """
        result = self.min_o_or_c(array=True) - self.low

        if array:
            return result
        return result[-1]

    def min_o_or_c(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        取 Open 或 Close 中较小值的价格序列（K线实体下沿价格）
        """
        result = np.where(self.c_sub_o(array=True) < 0, self.close, self.open)

        if array:
            return result
        return result[-1]

    def max_o_or_c(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        取 Open 或 Close 中较大值的价格序列（K线实体上沿价格）
        """
        result = np.where(self.c_sub_o(array=True) > 0, self.close, self.open)

        if array:
            return result
        return result[-1]

    def lower_edge_high(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        在指定序列长度中， Open 或 Close 中较小值与最低点的距离，所构成的差价序列（K线实体下沿距离最低点的高度）
        """
        result = self.min_o_or_c(array=True)[-n:] - min(self.low[-n:])

        if array:
            return result
        return result[-1]

    def upper_edge_high(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        在指定序列长度中， Open 或 Close 中较大值与最低点的距离，所构成的差价序列（K线实体上沿距离最低点的高度）
        """
        result = self.max_o_or_c(array=True)[-n:] - min(self.low[-n:])

        if array:
            return result
        return result[-1]

    def median_point_high(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        在指定序列长度中， Open与Close的中间点距离最低价的高度（K线实体的中间点距离最低点的高度）
        """
        result = self.min_o_or_c(array=True)[-n:] - min(self.low[-n:]) + self.abs_c_sub_o(array=True)[-n:] * 0.5

        if array:
            return result
        return result[-1]


# ======================================================================================================================================================================================================== #
# ======================================================================================================================================================================================================== #

class DayArrayManager(object):
    """日线价格序列管理器"""

    def __init__(self, size: int = 5, nature_day: bool = False, log: bool = False):
        """Constructor"""
        self.count: int = 0
        self.size: int = size
        self.nature_day: bool = nature_day
        self.log: bool = log
        self.inited: bool = False
        self.new_day: bool = False
        self.last_bar: BarData = None

        self.open_array: np.ndarray = np.zeros(size)
        self.high_array: np.ndarray = np.zeros(size)
        self.low_array: np.ndarray = np.zeros(size)
        self.close_array: np.ndarray = np.zeros(size)
        self.volume_array: np.ndarray = np.zeros(size)
        self.open_interest_array: np.ndarray = np.zeros(size)

    def update_bar(self, bar: BarData) -> None:
        """
        Update new bar data into array manager.
        """
        self.count += 1
        if not self.inited and self.count >= self.size:
            self.inited = True

        if self.nature_day:
            if self.last_bar and self.last_bar.datetime.date() != bar.datetime.date():        # 按自然交易日时间划分新的一天
                self.new_day = True
            else:
                self.new_day = False
        else:
            if self.last_bar and self.last_bar.datetime.date() != self.RealDate(bar):         # 针针对国内夜盘开盘时间为次日开盘时间
                self.new_day = True
            else:
                self.new_day = False

        if self.new_day:
            self.open_array[:-1] = self.open_array[1:]
            self.high_array[:-1] = self.high_array[1:]
            self.low_array[:-1] = self.low_array[1:]
            self.close_array[:-1] = self.close_array[1:]
            self.volume_array[:-1] = self.volume_array[1:]
            self.open_interest_array[:-1] = self.open_interest_array[1:]

            if not self.log:
                self.open_array[-1] = bar.open_price
                self.high_array[-1] = bar.high_price
                self.low_array[-1] = bar.low_price
                self.close_array[-1] = bar.close_price
                self.volume_array[-1] = bar.volume
                self.open_interest_array[-1] = bar.open_interest

            else:
                self.open_array[-1] = np.log(bar.open_price)
                self.high_array[-1] = np.log(bar.high_price)
                self.low_array[-1] = np.log(bar.low_price)
                self.close_array[-1] = np.log(bar.close_price)
                self.volume_array[-1] = np.log(bar.volume)
                self.open_interest_array[-1] = np.log(bar.open_interest)

        else:
            if not self.log:
                self.high_array[-1] = max(self.high_array[-1], bar.high_price)
                self.low_array[-1] = min(self.low_array[-1], bar.low_price)
                self.close_array[-1] = bar.close_price
                self.volume_array[-1] = int(self.volume_array[-1] + bar.volume)
                self.open_interest_array[-1] = bar.open_interest

            else:
                self.high_array[-1] = max(self.high_array[-1], np.log(bar.high_price))
                self.low_array[-1] = min(self.low_array[-1], np.log(bar.low_price))
                self.close_array[-1] = np.log(bar.close_price)
                self.volume_array[-1] = int(self.volume_array[-1] + np.log(bar.volume))
                self.open_interest_array[-1] = np.log(bar.open_interest)

        self.last_bar = bar

    @property
    def open(self) -> np.ndarray:
        """
        Get open price time series.
        """
        return self.open_array

    @property
    def high(self) -> np.ndarray:
        """
        Get high price time series.
        """
        return self.high_array

    @property
    def low(self) -> np.ndarray:
        """
        Get low price time series.
        """
        return self.low_array

    @property
    def close(self) -> np.ndarray:
        """
        Get close price time series.
        """
        return self.close_array

    @property
    def volume(self) -> np.ndarray:
        """
        Get trading volume time series.
        """
        return self.volume_array

    @property
    def open_interest(self) -> np.ndarray:
        """
        Get trading volume time series.
        """
        return self.open_interest_array

    def RealDate(self, data: Union[TickData, BarData]):
        """"""
        day_offset = timedelta(days=0)
        if data.datetime.hour >= 18:
            if data.datetime.isoweekday() == 5:         # 周五晚上
                day_offset = timedelta(days=3)
            elif data.datetime.isoweekday() == 6:       # 周六晚上
                day_offset = timedelta(days=2)
            elif data.datetime.isoweekday() == 7:       # 周日晚上
                day_offset = timedelta(days=1)

        else:
            if data.datetime.isoweekday() == 6:         # 周六
                day_offset = timedelta(days=2)
            elif data.datetime.isoweekday() == 7:       # 周日
                day_offset = timedelta(days=1)
        
        dt = data.datetime + day_offset

        return dt.date()

    # ======================================================================================================================================================================================================== #
    # Overlap Studies 重叠研究指标
    def sma(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Simple moving average. 简单移动平均线
        """
        if log:
            result = talib.SMA(np.log(self.close), timeperiod=n)
        else:
            result = talib.SMA(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def ema(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Exponential moving average.  指数移动平均线：趋向类指标，其构造原理是仍然对价格收盘价进行算术平均，并根据计算结果来进行分析，用于判断价格未来走势的变动趋势。
        """
        if log:
            result = talib.EMA(np.log(self.close), timeperiod=n)
        else:
            result = talib.EMA(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def ma(self, n: int, matype: int = 0, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Moving average.  移动平均线：matype: 0=SMA(默认), 1=EMA(指数移动平均线), 2=WMA(加权移动平均线), 3=DEMA(双指数移动平均线), 4=TEMA(三重指数移动平均线), 5=TRIMA, 6=KAMA(考夫曼自适应移动平均线), 7=MAMA, 8=T3(三重指数移动平均线)
        """
        if log:
            result = talib.MA(np.log(self.close), timeperiod=n, matype=matype)
        else:
            result = talib.MA(self.close, timeperiod=n, matype=matype)

        if array:
            return result
        return result[-1]

    def rsi(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Relative Strenght Index (RSI). 相对强弱指数：通过比较一段时期内的平均收盘涨数和平均收盘跌数来分析市场买沽盘的意向和实力，从而作出未来市场的走势。
        """
        if log:
            result = talib.RSI(np.log(self.close), timeperiod=n)
        else:
            result = talib.RSI(self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

    def macd(self, fast_period: int, slow_period: int, signal_period: int, array: bool = False, log: bool = False) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[float, float, float]]:
        """
        Moving Average Convergence/Divergence. 平滑异同移动平均线：利用收盘价的短期（常用为12日）指数移动平均线与长期（常用为26日）指数移动平均线之间的聚合与分离状况，对买进、卖出时机作出研判的技术指标。
        """
        if log:
            macd, signal, hist = talib.MACD(np.log(self.close), fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
        else:
            macd, signal, hist = talib.MACD(self.close, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)

        if array:
            return macd, signal, hist
        return macd[-1], signal[-1], hist[-1]

    # ======================================================================================================================================================================================================== #
    # Volatility Indicator 波动率指标
    def trange(self, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        TRANGE. 真实波动幅度 = max(当日最高价, 昨日收盘价) − min(当日最低价, 昨日收盘价)
        """
        if log:
            result = talib.TRANGE(np.log(self.high), np.log(self.low), np.log(self.close))
        else:
            result = talib.TRANGE(self.high, self.low, self.close)

        if array:
            return result
        return result[-1]

    def atr(self, n: int, array: bool = False, log: bool = False) -> Union[float, np.ndarray]:
        """
        Average True Range (ATR). 平均真实波动幅度：真实波动幅度的 N 日 指数移动平均数。
        """
        if log:
            result = talib.ATR(np.log(self.high), np.log(self.low), np.log(self.close), timeperiod=n)
        else:
            result = talib.ATR(self.high, self.low, self.close, timeperiod=n)

        if array:
            return result
        return result[-1]

# ======================================================================================================================================================================================================== #
# ======================================================================================================================================================================================================== #


def virtual(func: Callable) -> Callable:
    """
    mark a function as "virtual", which means that this function can be override.
    any base class should use this or @abstractmethod to decorate all functions
    that can be (re)implemented by subclasses.
    """
    return func


file_handlers: Dict[str, logging.FileHandler] = {}


def _get_file_logger_handler(filename: str) -> logging.FileHandler:
    handler = file_handlers.get(filename, None)
    if handler is None:
        handler = logging.FileHandler(filename)
        file_handlers[filename] = handler  # Am i need a lock?
    return handler


def get_file_logger(filename: str) -> logging.Logger:
    """
    return a logger that writes records into a file.
    """
    logger = logging.getLogger(filename)
    handler = _get_file_logger_handler(filename)  # get singleton handler.
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)  # each handler will be added only once.
    return logger
