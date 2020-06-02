"""
General utility functions.
"""

import json
import logging
import sys
import datetime
from pathlib import Path
from typing import Callable, Dict, Tuple, Union
from decimal import Decimal
from math import floor, ceil

import numpy as np
import talib

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
    if not folder_path.exists():
        folder_path.mkdir()
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

    if "." not in value_str:
        return 0
    else:
        _, buf = value_str.split(".")
        return len(buf)


class BarGenerator:
    """
    For:
    1. generating 1 minute bar data from tick data
    2. generating x minute bar/x hour bar data from 1 minute data

    Notice:
    1. for x minute bar, x must be able to divide 60: 2, 3, 5, 6, 10, 15, 20, 30
    2. for x hour bar, x can be any number
    """

    def __init__(
        self,
        on_bar: Callable,
        window: int = 0,
        on_window_bar: Callable = None,
        interval: Interval = Interval.MINUTE
    ):
        """Constructor"""
        self.bar: BarData = None
        self.on_bar: Callable = on_bar

        self.interval: Interval = interval
        self.interval_count: int = 0

        self.window: int = window
        self.window_bar: BarData = None
        self.on_window_bar: Callable = on_window_bar

        self.last_tick: TickData = None
        self.last_bar: BarData = None

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

            self.bar.datetime = self.bar.datetime.replace(
                second=0, microsecond=0
            )
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

    def update_bar(self, bar: BarData) -> None:
        """
        Update 1 minute bar into generator
        """
        # If not inited, creaate window bar object
        if not self.window_bar:
            # Generate timestamp for bar data
            if self.interval == Interval.MINUTE:
                dt = bar.datetime.replace(second=0, microsecond=0)
            else:
                dt = bar.datetime.replace(minute=0, second=0, microsecond=0)

            self.window_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
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

        if self.interval == Interval.MINUTE:
            # x-minute bar

            # 官方版本合成方式。整除切分法进行分钟K线合成，只能合成 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60分钟K线
            # if not (bar.datetime.minute + 1) % self.window:
            #     finished = True
            
            # 计数切分法进行分钟K线合成，可以合成任意分钟K线
            if self.last_bar and bar.datetime.minute != self.last_bar.datetime.minute:
                self.interval_count += 1

                if not self.interval_count % self.window:
                    finished = True
                    self.interval_count = 0

        # 计数切分法进行N小时K线合成，可以合成任意小时K线
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

        if finished:
            self.on_window_bar(self.window_bar)
            self.window_bar = None

        # Cache last bar object
        self.last_bar = bar

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

class NewBarGenerator:
    """ 增强版K线合成器。可合成 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60秒钟K线。 """

    def __init__(
        self,
        on_second_bar: Callable = None,
        second_window: int = 60,
        on_bar: Callable = None,
        minute_window: int = 1,
        on_window_bar: Callable = None
    ):
        
        self.second_window: int = second_window
        self.second_bar_range = [i for i in range(0, 60, self.second_window)]
        
        self.second_bar: BarData = None
        self.on_second_bar: Callable = on_second_bar
        self.last_tick: TickData = None

        self.second_window_bar: BarData = None
        self.on_bar: Callable = on_bar

        self.minute_window_bar: BarData = None        
        self.interval_count: int = 0
        
        self.last_bar: BarData = None
        self.minute_window: int = minute_window
        self.on_window_bar: Callable = on_window_bar

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
            bar_second = (tick.datetime - datetime.timedelta(seconds=self.second_window)).second
            self.second_bar.datetime = self.second_bar.datetime.replace(second=bar_second, microsecond=0)
            self.on_second_bar(self.second_bar)
            new_second_bar = True

        if new_second_bar:
            self.second_bar = BarData(
                symbol=tick.symbol,
                exchange=tick.exchange,
                interval=Interval.SECOND,
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
        策略初始化时载入行情数据，将X秒钟的second_bar，合成1分钟bar,并回调on_bar。
        """
        # If not inited, creaate second window bar object
        if not self.second_window_bar:
            # Generate timestamp for bar data
            dt = second_bar.datetime.replace(second=0, microsecond=0)

            self.second_window_bar = BarData(
                symbol=second_bar.symbol,
                exchange=second_bar.exchange,
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
        if (second_bar.datetime + datetime.timedelta(seconds=self.second_window)).second == 0:
            new_minute = True

        if new_minute:
            self.on_bar(self.second_window_bar)
            self.second_window_bar = None

    def update_x_minute_bar(self, bar: BarData) -> None:
        """
        Update 1 minute bar into generator
        """
        # If not inited, creaate window bar object
        if not self.minute_window_bar:
            # Generate timestamp for bar data
            dt = bar.datetime.replace(second=0, microsecond=0)

            self.minute_window_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=dt,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price
            )

        # Otherwise, update high/low price into window bar
        else:
            self.minute_window_bar.high_price = max(self.minute_window_bar.high_price, bar.high_price)
            self.minute_window_bar.low_price  = min(self.minute_window_bar.low_price, bar.low_price)

        # Update close price/volume into window bar
        self.minute_window_bar.close_price = bar.close_price
        self.minute_window_bar.volume += int(bar.volume)
        self.minute_window_bar.open_interest = bar.open_interest

        # Check if minute window bar completed
        finished = False

        if self.last_bar and bar.datetime.minute != self.last_bar.datetime.minute:
            if self.minute_window == 1:
                finished = True
            
            else:
                self.interval_count += 1
                if not self.interval_count % self.minute_window:
                    finished = True
                    self.interval_count = 0

        if finished:
            self.on_window_bar(self.minute_window_bar)
            self.minute_window_bar = None

        # Cache last bar object
        self.last_bar = bar

    def generate(self) -> None:
        """
        Generate the second_bar data and call callback immediately.
        """
        second_bar = self.second_bar

        if self.second_bar:
            second_bar.datetime = second_bar.datetime.replace(microsecond=0)
            self.on_second_bar(second_bar)

        self.second_bar = None
        return second_bar

class ArrayManager(object):
    """
    For:
    1. time series container of bar data
    2. calculating technical indicator value
    """

    def __init__(self, size: int = 100):
        """Constructor"""
        self.count: int = 0
        self.size: int = size
        self.inited: bool = False

        self.open_array: np.ndarray = np.zeros(size)
        self.high_array: np.ndarray = np.zeros(size)
        self.low_array: np.ndarray = np.zeros(size)
        self.close_array: np.ndarray = np.zeros(size)
        self.volume_array: np.ndarray = np.zeros(size)
        self.open_interest_array: np.ndarray = np.zeros(size)

        # 初始化每日OHLC价格数组，用于后续记录日K线数据
        self.open_price_day = np.zeros(size)
        self.high_price_day = np.zeros(size)
        self.low_price_day = np.zeros(size)
        self.close_price_day = np.zeros(size)

    def update_bar(self, bar: BarData) -> None:
        """
        Update new bar data into array manager.
        """
        self.count += 1
        if not self.inited and self.count >= self.size:
            self.inited = True

        self.open_array[:-1] = self.open_array[1:]
        self.high_array[:-1] = self.high_array[1:]
        self.low_array[:-1] = self.low_array[1:]
        self.close_array[:-1] = self.close_array[1:]
        self.volume_array[:-1] = self.volume_array[1:]
        self.open_interest_array[:-1] = self.open_interest_array[1:]

        self.open_array[-1] = bar.open_price
        self.high_array[-1] = bar.high_price
        self.low_array[-1] = bar.low_price
        self.close_array[-1] = bar.close_price
        self.volume_array[-1] = bar.volume
        self.open_interest_array[-1] = bar.open_interest

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

    def sma(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Simple moving average.
        """
        result = talib.SMA(self.close, n)
        if array:
            return result
        return result[-1]

    def ema(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Exponential moving average.
        """
        result = talib.EMA(self.close, n)
        if array:
            return result
        return result[-1]

    def kama(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        KAMA.
        """
        result = talib.KAMA(self.close, n)
        if array:
            return result
        return result[-1]

    def wma(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        WMA.
        """
        result = talib.WMA(self.close, n)
        if array:
            return result
        return result[-1]

    def apo(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        APO.
        """
        result = talib.APO(self.close, n)
        if array:
            return result
        return result[-1]

    def cmo(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        CMO.
        """
        result = talib.CMO(self.close, n)
        if array:
            return result
        return result[-1]

    def mom(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        MOM.
        """
        result = talib.MOM(self.close, n)
        if array:
            return result
        return result[-1]

    def ppo(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        PPO.
        """
        result = talib.PPO(self.close, n)
        if array:
            return result
        return result[-1]

    def roc(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        ROC.
        """
        result = talib.ROC(self.close, n)
        if array:
            return result
        return result[-1]

    def rocr(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        ROCR.
        """
        result = talib.ROCR(self.close, n)
        if array:
            return result
        return result[-1]

    def rocp(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        ROCP.
        """
        result = talib.ROCP(self.close, n)
        if array:
            return result
        return result[-1]

    def rocr_100(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        ROCR100.
        """
        result = talib.ROCR100(self.close, n)
        if array:
            return result
        return result[-1]

    def trix(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        TRIX.
        """
        result = talib.TRIX(self.close, n)
        if array:
            return result
        return result[-1]

    def std(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Standard deviation.
        """
        result = talib.STDDEV(self.close, n)
        if array:
            return result
        return result[-1]

    def obv(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        OBV.
        """
        result = talib.OBV(self.close, self.volume)
        if array:
            return result
        return result[-1]

    def cci(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Commodity Channel Index (CCI).
        """
        result = talib.CCI(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def atr(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Average True Range (ATR).
        """
        result = talib.ATR(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def natr(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        NATR.
        """
        result = talib.NATR(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def rsi(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Relative Strenght Index (RSI).
        """
        result = talib.RSI(self.close, n)
        if array:
            return result
        return result[-1]

    def macd(
        self,
        fast_period: int,
        slow_period: int,
        signal_period: int,
        array: bool = False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[float, float, float]
    ]:
        """
        MACD.
        """
        macd, signal, hist = talib.MACD(
            self.close, fast_period, slow_period, signal_period
        )
        if array:
            return macd, signal, hist
        return macd[-1], signal[-1], hist[-1]

    def adx(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        ADX.
        """
        result = talib.ADX(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def adxr(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        ADXR.
        """
        result = talib.ADXR(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def dx(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        DX.
        """
        result = talib.DX(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def minus_di(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        MINUS_DI.
        """
        result = talib.MINUS_DI(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def plus_di(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        PLUS_DI.
        """
        result = talib.PLUS_DI(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def willr(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        WILLR.
        """
        result = talib.WILLR(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def ultosc(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        Ultimate Oscillator.
        """
        result = talib.ULTOSC(self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def trange(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        TRANGE.
        """
        result = talib.TRANGE(self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def boll(
        self,
        n: int,
        dev: float,
        array: bool = False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[float, float]
    ]:
        """
        Bollinger Channel.
        """
        mid = self.sma(n, array)
        std = self.std(n, array)

        up = mid + std * dev
        down = mid - std * dev

        return up, down

    def keltner(
        self,
        n: int,
        dev: float,
        array: bool = False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[float, float]
    ]:
        """
        Keltner Channel.
        """
        mid = self.sma(n, array)
        atr = self.atr(n, array)

        up = mid + atr * dev
        down = mid - atr * dev

        return up, down

    def donchian(
        self, n: int, array: bool = False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[float, float]
    ]:
        """
        Donchian Channel.
        """
        up = talib.MAX(self.high, n)
        down = talib.MIN(self.low, n)

        if array:
            return up, down
        return up[-1], down[-1]

    def aroon(
        self,
        n: int,
        array: bool = False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[float, float]
    ]:
        """
        Aroon indicator.
        """
        aroon_up, aroon_down = talib.AROON(self.high, self.low, n)

        if array:
            return aroon_up, aroon_down
        return aroon_up[-1], aroon_down[-1]

    def aroonosc(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Aroon Oscillator.
        """
        result = talib.AROONOSC(self.high, self.low, n)

        if array:
            return result
        return result[-1]

    def minus_dm(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        MINUS_DM.
        """
        result = talib.MINUS_DM(self.high, self.low, n)

        if array:
            return result
        return result[-1]

    def plus_dm(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        PLUS_DM.
        """
        result = talib.PLUS_DM(self.high, self.low, n)

        if array:
            return result
        return result[-1]

    def mfi(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Money Flow Index.
        """
        result = talib.MFI(self.high, self.low, self.close, self.volume, n)
        if array:
            return result
        return result[-1]

    def ad(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        AD.
        """
        result = talib.AD(self.high, self.low, self.close, self.volume, n)
        if array:
            return result
        return result[-1]

    def adosc(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        ADOSC.
        """
        result = talib.ADOSC(self.high, self.low, self.close, self.volume, n)
        if array:
            return result
        return result[-1]

    def bop(self, array: bool = False) -> Union[float, np.ndarray]:
        """
        BOP.
        """
        result = talib.BOP(self.open, self.high, self.low, self.close)

        if array:
            return result
        return result[-1]


    # 参考TB函数及公式建立工具箱
    def Highest(self, PriceArrray:"Array", Length:"周期"=5) -> "HighestValue":
        """求最高"""
        HighestValue = PriceArrray[-1]
        for i in range(-2, -Length-1, -1):
            if PriceArrray[i] > HighestValue:
                HighestValue = PriceArrray[i]
        return HighestValue
        # PriceArrray = PriceArrray[-Length:]
        # return np.max(PriceArrray)

    def Lowest(self, PriceArrray:"Array", Length:"周期"=5) -> "LowestValue":
        """求最低"""
        LowestValue = PriceArrray[-1]
        for i in range(-2, -Length-1, -1):
            if PriceArrray[i] > LowestValue:
                LowestValue = PriceArrray[i]
        return LowestValue
        # PriceArrray = PriceArrray[-Length:]
        # return np.min(PriceArrray)

    def CrossOver(self, PriceArrray_1:"Array", PriceArrray_2:"Array") -> bool:
        """求是否上穿"""
        if PriceArrray_1[-2] < PriceArrray_2[-2] and PriceArrray_1[-1] >= PriceArrray_2[-1]:
            return True
        else:
            return False

    def CrossUnder(self, PriceArrray_1:"Array", PriceArrray_2:"Array") -> bool:
        """求是否下破"""
        if PriceArrray_1[-2] > PriceArrray_2[-2] and PriceArrray_1[-1] <= PriceArrray_2[-1]:
            return True
        else:
            return False

    def TrueDate(self, bar:BarData):
        """"""
        day_offset = datetime.timedelta(days=0)
        if bar.datetime.hour >= 18:
            if bar.datetime.isoweekday() == 5:         # 周五晚上
                day_offset = datetime.timedelta(days=3)
            elif bar.datetime.isoweekday() == 6:       # 周六晚上
                day_offset = datetime.timedelta(days=2)
            else:                                      # 周日晚上
                day_offset = datetime.timedelta(days=1)

        else:
            if bar.datetime.isoweekday() == 6:         # 周六
                day_offset = datetime.timedelta(days=2)
            elif bar.datetime.isoweekday() == 7:       # 周日
                day_offset = datetime.timedelta(days=1)
        bar.datetime = bar.datetime + day_offset
        return bar.datetime.date()

    def OpenD(self, bar:BarData):
        """记录每日开盘价"""
        if bar.datetime.date() != self.TrueDate(bar):
            self.open_price_day[:-1] = self.open_price_day[1:]
            self.open_price_day[-1] = bar.open_price
        else:
            pass
        return self.open_price_day

    def HighD(self, bar:BarData):
        """记录每日最高价"""
        if bar.datetime.date() != self.TrueDate(bar):
            self.high_price_day[:-1] = self.high_price_day[1:]
            self.high_price_day[-1] = bar.high_price
        else:
            if self.high_price_day[-1] < bar.high_price:
                self.high_price_day[-1] = bar.high_price
            else:
                pass
        return self.high_price_day

    def LowD(self, bar:BarData):
        """记录每日最低价"""
        if bar.datetime.date() != self.TrueDate(bar):
            self.low_price_day[:-1] = self.low_price_day[1:]
            self.low_price_day[-1] = bar.low_price
        else:
            if self.low_price_day[-1] > bar.low_price:
                self.low_price_day[-1] = bar.low_price
            else:
                pass
        return self.low_price_day

    def CloseD(self, bar:BarData):
        """记录每日收盘价"""
        if bar.datetime.date() != self.TrueDate(bar):
            self.close_price_day[:-1] = self.close_price_day[1:]
            self.close_price_day[-1] = bar.close_price
        else:
            self.close_price_day[-1] = bar.close_price
        return self.close_price_day


def virtual(func: Callable) -> Callable:
    """
    mark a function as "virtual", which means that this function can be override.
    any base class should use this or @abstractmethod to decorate all functions
    that can be (re)implemented by subclasses.
    """
    return func


def run_time(func):
    '''计算函数运行耗时装饰器'''
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        print (f"{func.__name__} 运行耗时： {(end_time - start_time).total_seconds()}秒")
        return result
    return wrapper


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
