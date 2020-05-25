from datetime import timedelta
from typing import List, Optional
from pytz import timezone

import pandas as pd
from tqsdk import TqApi

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.datasource.dataapi import DataSourceApi
from vnpy.trader.object import BarData, HistoryRequest


TIME_UTC8 = 8 * 60 * 60 * 1000000000     # 转换为东八区时间

INTERVAL_VT2TQ = {
    Interval.MINUTE: 60,
    Interval.HOUR: 60 * 60,
    Interval.DAILY: 60 * 60 * 24,
}

INTERVAL_ADJUSTMENT_MAP = {
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
    Interval.DAILY: timedelta()         # no need to adjust for daily bar
}

CHINA_TZ = timezone("Asia/Shanghai")

class TqdataClient(DataSourceApi):
    """天勤TQData客户端封装类"""

    def __init__(self):
        """"""
        self.inited: bool = False
        self.symbols: set = set()
        self.api = None

    def init(self) -> bool:
        """"""
        if self.inited:
            return True

        try:
            self.api = TqApi()
            # 获得全部合约
            self.symbols = [k for k, v in self.api._data["quotes"].items()]
        except:
            return False

        self.inited = True
        return True

    def to_tq_symbol(self, symbol: str, exchange: Exchange) -> str:
        """
        TQSdk exchange first
        """
        for count, word in enumerate(symbol):
            if word.isdigit():
                break

        # Check for index symbol
        time_str = symbol[count:]
        if time_str in ["88"]:
            return f"KQ.m@{exchange}.{symbol[:count]}"
        if time_str in ["99"]:
            return f"KQ.i@{exchange}.{symbol[:count]}"

        return f"{exchange.value}.{symbol}"

    def query_history(self, req: HistoryRequest, tq_interval: int) -> Optional[List[BarData]]:
        """
        Query history bar data from TqSdk.
        """
        if self.symbols is None:
            return None
            
        symbol = req.symbol
        exchange = req.exchange
        interval = req.interval
        start = req.start
        end = req.end

        tq_symbol = self.to_tq_symbol(symbol, exchange)
        if tq_symbol not in self.symbols:
            return None

        # 若未从上层load_bar传入tq_interval，则返回空值
        if not tq_interval:
            return None

        # For adjust timestamp from bar close point (TQData) to open point (VN Trader)
        # 调整 K线时间戳为 K线起始时间
        adjustment = INTERVAL_ADJUSTMENT_MAP[interval]

        # For querying night trading period data
        end += timedelta(minutes=1)

        # 获取最新的数据，无法指定日期
        df = self.api.get_kline_serial(tq_symbol, tq_interval, 10000).sort_values(by=["datetime"])

        # 转换为东八区时间
        df["datetime"] = pd.to_datetime(df["datetime"] + TIME_UTC8)

        # 过滤开始结束时间
        # df = df[(df['datetime'] >= start - timedelta(days=1)) & (df['datetime'] < end)]

        data: List[BarData] = []

        if df is not None:
            for ix, row in df.iterrows():
                dt = row.datetime - adjustment
                dt = dt.replace(tzinfo=CHINA_TZ)

                bar = BarData(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    datetime=dt,
                    open_price=row["open"],
                    high_price=row["high"],
                    low_price=row["low"],
                    close_price=row["close"],
                    volume=row["volume"],
                    open_interest=row.get("open_oi", 0),
                    gateway_name="TQ",
                )
                data.append(bar)

        return data


tqdata_client = TqdataClient()