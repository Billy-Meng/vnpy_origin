from datetime import timedelta, datetime
from typing import List, Optional
from pytz import timezone

from numpy import ndarray
import jqdatasdk as jq

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.datasource.dataapi import DataSourceApi
from vnpy.trader.object import BarData, HistoryRequest
from vnpy.trader.setting import SETTINGS

INTERVAL_VT2JQ = {
    Interval.MINUTE: "1m",
    Interval.HOUR: "60m",
    Interval.DAILY: "1d",
}

INTERVAL_ADJUSTMENT_MAP_JQ = {
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
    Interval.DAILY: timedelta()  # no need to adjust for daily bar
}

CHINA_TZ = timezone("Asia/Shanghai")


class JqdataClient(DataSourceApi):
    """聚宽JQData客户端封装类"""

    def __init__(self):
        """"""
        self.username: str = SETTINGS["jqdata.username"]
        self.password: str = SETTINGS["jqdata.password"]

        self.inited: bool = False

    def init(self, username: str = "", password: str = "") -> bool:
        """"""
        if self.inited:
            return True

        if username and password:
            self.username = username
            self.password = password

        if not self.username or not self.password:
            return False

        try:
            jq.auth(self.username, self.password)
        except Exception as ex:
            print("聚宽接口初始化失败:" + repr(ex))
            return False

        self.inited = True
        return True

    def to_jq_symbol(self, symbol: str, exchange: Exchange) -> str:
        """
        CZCE product of RQData has symbol like "TA1905" while
        vt symbol is "TA905.CZCE" so need to add "1" in symbol.
        """
        if exchange in [Exchange.SSE, Exchange.SZSE]:
            if exchange == Exchange.SSE:
                jq_symbol = f"{symbol}.XSHG"  # 上海证券交易所
            else:
                jq_symbol = f"{symbol}.XSHE"  # 深圳证券交易所
        elif exchange == Exchange.SHFE:
            jq_symbol = f"{symbol}.XSGE"  # 上期所
        elif exchange == Exchange.CFFEX:
            jq_symbol = f"{symbol}.CCFX"  # 中金所
        elif exchange == Exchange.DCE:
            jq_symbol = f"{symbol}.XDCE"  # 大商所
        elif exchange == Exchange.INE:
            jq_symbol = f"{symbol}.XINE"  # 上海国际能源期货交易所
        elif exchange == Exchange.CZCE:
            # 郑商所 的合约代码年份只有三位 需要特殊处理
            for count, word in enumerate(symbol):
                if word.isdigit():
                    break

            # Check for index symbol
            time_str = symbol[count:]
            if time_str in ["88", "888", "8888", "99", "9999", "999"]:
                jq_symbol = f"{symbol}.XZCE"
                return jq_symbol

            else:
                # noinspection PyUnboundLocalVariable
                product = symbol[:count]
                year = symbol[count]
                month = symbol[count + 1:]

                if year == "9":
                    year = "1" + year       # 2019年有效
                else:
                    year = "2" + year       # 2020~2028年有效

                jq_symbol = f"{product}{year}{month}.XZCE"
                return jq_symbol

        return jq_symbol.upper()

    def query_history(self, req: HistoryRequest) -> Optional[List[BarData]]:
        """
        Query history bar data from JQData.
        """

        symbol = req.symbol
        exchange = req.exchange
        interval = req.interval
        start = req.start
        end = req.end

        jq_symbol = self.to_jq_symbol(symbol, exchange)

        jq_interval = INTERVAL_VT2JQ.get(interval)
        if not jq_interval:
            return None

        # For adjust timestamp from bar close point (JQData) to open point (VN Trader)
        adjustment = INTERVAL_ADJUSTMENT_MAP_JQ.get(interval)

        # 聚宽传入未来时间仍会生成数据，开高低收为最近收盘价，为避免冗余数据，调整为取当前时间
        now = datetime.now()

        df = jq.get_price(
            jq_symbol,
            frequency=jq_interval,
            fields=["open", "high", "low", "close", "volume"],
            start_date=start,
            end_date=now,
            fq=None,
            panel=True,
            skip_paused=True
        )

        data: List[BarData] = []

        if df is not None:
            for ix, row in df.iterrows():
                dt = row.name.to_pydatetime() - adjustment
                dt = CHINA_TZ.localize(dt.replace(tzinfo=None))

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
                    gateway_name="JQ"
                )
                data.append(bar)

        return data

jqdata_client = JqdataClient()