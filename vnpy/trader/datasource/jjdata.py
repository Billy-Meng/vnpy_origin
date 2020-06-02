from datetime import timedelta
from typing import List, Optional
from pytz import timezone

import pandas as pd
from gm.api import set_token, history, history_n, get_instrumentinfos

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.datasource.dataapi import DataSourceApi
from vnpy.trader.object import TickData, BarData, HistoryRequest

import numpy as np


JJ_TOKEN = ""

CHINA_TZ = timezone("Asia/Shanghai")

class JjdataClient(DataSourceApi):
    """掘金JjData客户端封装类"""

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
            set_token(JJ_TOKEN)

            # 获取全部合约
            instrumentinfos = get_instrumentinfos(symbols=None, exchanges=None, sec_types=None, names=None, fields=None, df=True)
            self.symbols = [k for k in instrumentinfos["symbol"]]

        except:
            return False

        self.inited = True
        return True

    def to_jj_symbol(self, symbol: str, exchange: Exchange) -> str:
        """
        转换为掘金的标的代码
        """
        return f"{exchange.value}.{symbol}"

    def query_history(self, req: HistoryRequest, frequency: str) -> Optional[List[BarData]]:
        """
        Query history bar data from JJSdk.
        """
        if self.symbols is None:
            return None
            
        symbol = req.symbol
        exchange = req.exchange
        interval = req.interval
        start = req.start
        end = req.end

        jj_symbol = self.to_jj_symbol(symbol, exchange)
        if jj_symbol not in self.symbols:
            return None

        # 若未从上层load_bar传入frequency，则返回空值
        if not frequency:
            return None

        # For querying night trading period data
        end += timedelta(minutes=1)

        # 查询历史行情最新 n条
        if frequency == "tick":
            fields = "created_at, open, high, low, price, cum_volume, cum_amount, trade_type, last_volume, cum_position, last_amount, quotes"
            df = history_n(symbol=jj_symbol, frequency=frequency, end_time=end, count=33000, fields=fields, df=True).sort_values(by=["created_at"])
            df["datetime"] = pd.to_datetime(df["created_at"])

        else:
            fields = "bob, open, high, low, close, volume, position"
            df = history_n(symbol=jj_symbol, frequency=frequency, end_time=end, count=33000, fields=fields, df=True).sort_values(by=["bob"])
            df["datetime"] = pd.to_datetime(df["bob"])

        # 过滤开始结束时间
        # df = df[(df['datetime'] >= start.replace(tzinfo=CHINA_TZ) - timedelta(days=1)) & (df['datetime'] < end.replace(tzinfo=CHINA_TZ))]

        if frequency == "tick":
            data: List[TickData] = []

            if df is not None:
                for ix, row in df.iterrows():
                    dt = row.datetime.replace(tzinfo=CHINA_TZ)

                    tick = TickData(
                        symbol = symbol,
                        exchange = exchange,
                        datetime = dt,

                        name = symbol,
                        volume = row.cum_volume,
                        open_interest = row.cum_position,
                        last_price = row.price,
                        last_volume = row.last_volume,
                        limit_up  = 0,
                        limit_down  = 0,

                        open_price = row.open,
                        high_price = row.high,
                        low_price = row.low,
                        pre_close = 0,

                        bid_price_1 = row.quotes[0]["bid_p"],
                        bid_price_2 = 0,
                        bid_price_3 = 0,
                        bid_price_4 = 0,
                        bid_price_5 = 0,

                        ask_price_1 = row.quotes[0]["ask_p"],
                        ask_price_2 = 0,
                        ask_price_3 = 0,
                        ask_price_4 = 0,
                        ask_price_5 = 0,

                        bid_volume_1 = row.quotes[0]["bid_v"],
                        bid_volume_2 = 0,
                        bid_volume_3 = 0,
                        bid_volume_4 = 0,
                        bid_volume_5 = 0,

                        ask_volume_1 = row.quotes[0]["ask_v"],
                        ask_volume_2 = 0,
                        ask_volume_3 = 0,
                        ask_volume_4 = 0,
                        ask_volume_5 = 0,
                        gateway_name = "jj",
                    )
                    data.append(tick)

        else:
            data: List[BarData] = []

            if df is not None:
                for ix, row in df.iterrows():
                    dt = row.datetime.replace(tzinfo=CHINA_TZ)

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
                        open_interest=row.get("position", 0),
                        gateway_name="jj",
                    )
                    data.append(bar)

        return data


jjdata_client = JjdataClient()