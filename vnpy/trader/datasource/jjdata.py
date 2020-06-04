from datetime import timedelta
from typing import List, Optional
from pytz import timezone
from enum import Enum

import pandas as pd
from gm.api import set_token, history, history_n, get_instrumentinfos

from vnpy.trader.constant import Exchange
from vnpy.trader.datasource.dataapi import DataSourceApi
from vnpy.trader.object import TickData, BarData, HistoryRequest
from vnpy.trader.setting import SETTINGS


JJ_TOKEN = SETTINGS["jjdata.token"]

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
        start = req.start
        end = req.end

        jj_symbol = self.to_jj_symbol(symbol, exchange)
        if jj_symbol not in self.symbols:
            return None

        # 若未从上层load_bar传入frequency，则返回空值
        if not frequency:
            return None

        frequency = frequency.value if isinstance(frequency, Enum) else frequency

        end_time = end + timedelta(minutes=1)
        first_df = True

        # 查询历史行情最新 n条
        if frequency == "tick":
            fields = "created_at, open, high, low, price, cum_volume, cum_amount, trade_type, last_volume, cum_position, last_amount, quotes"

            print("开始从掘金获取Tick数据……")
            for i in range(10000):

                if first_df:
                    df = history_n(symbol=jj_symbol, frequency=frequency, end_time=end_time, count=33000, fields=fields, df=True)
                    bacd_time = df["created_at"][1]
                    first_df  = False

                else:
                    bar_data = history_n(symbol=jj_symbol, frequency=frequency, end_time=bacd_time, count=33000, fields=fields, df=True)
                    df = pd.concat([bar_data, df[1:]], ignore_index=True)
                    bacd_time = df["created_at"][1]

                print(f"第 {i+1} 次循环获取数据，数据起始时间为：{bacd_time}")

                if CHINA_TZ.localize(df["created_at"][0].to_pydatetime().replace(tzinfo=None)) <= start:
                    break


            df.rename(columns={"created_at":"datetime"}, inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values(by=["datetime"])
            df = df[2:]
            print("数据获取处理完毕。")

        else:
            fields = "bob, open, high, low, close, volume, position"
            
            print(f"开始从掘金获取 {frequency} K线数据……")
            for i in range(10000):

                if first_df:
                    df = history_n(symbol=jj_symbol, frequency=frequency, end_time=end_time, count=33000, fields=fields, df=True)
                    bacd_time = df["bob"][1]
                    first_df  = False

                else:
                    bar_data = history_n(symbol=jj_symbol, frequency=frequency, end_time=bacd_time, count=33000, fields=fields, df=True)
                    df = pd.concat([bar_data, df[1:]], ignore_index=True)
                    bacd_time = df["bob"][1]

                print(f"第 {i+1} 次循环获取数据，数据起始时间为：{bacd_time}")
                
                if CHINA_TZ.localize(df["bob"][0].to_pydatetime().replace(tzinfo=None)) <= start:
                    break


            df.rename(columns={"bob":"datetime"}, inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values(by=["datetime"])
            df = df[2:]
            print("数据获取处理完毕。")


        if frequency == "tick":
            data: List[TickData] = []

            if df is not None:
                for ix, row in df.iterrows():
                    dt = CHINA_TZ.localize(row.datetime.to_pydatetime().replace(tzinfo=None))

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
                        gateway_name = "JJ",
                    )
                    data.append(tick)

        else:
            data: List[BarData] = []
            
            interval = frequency

            if df is not None:
                for ix, row in df.iterrows():
                    dt = CHINA_TZ.localize(row.datetime.to_pydatetime().replace(tzinfo=None))

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
                        gateway_name="JJ",
                    )
                    data.append(bar)

        return data


jjdata_client = JjdataClient()