"""
Please install futu-api before use.
"""

from copy import copy
from collections import OrderedDict
from datetime import datetime
from threading import Thread
from time import sleep

from futu import (
    KLType,
    ModifyOrderOp,
    TrdSide,
    TrdEnv,
    OpenHKTradeContext,
    OpenQuoteContext,
    OpenUSTradeContext,
    OrderBookHandlerBase,
    OrderStatus,
    OrderType,
    RET_ERROR,
    RET_OK,
    StockQuoteHandlerBase,
    TradeDealHandlerBase,
    TradeOrderHandlerBase
)

from vnpy.trader.constant import Direction, Exchange, Product, Status
from vnpy.trader.event import EVENT_TIMER
from vnpy.trader.gateway import BaseGateway, LocalOrderManager
from vnpy.trader.object import (
    BarData,
    TickData,
    OrderData,
    TradeData,
    AccountData,
    ContractData,
    PositionData,
    SubscribeRequest,
    OrderRequest,
    CancelRequest,
    HistoryRequest,
    Interval
)

EXCHANGE_VT2FUTU = {
    Exchange.SMART: "US",
    Exchange.SEHK: "HK",
    Exchange.HKFE: "HK_FUTURE",
}
EXCHANGE_FUTU2VT = {v: k for k, v in EXCHANGE_VT2FUTU.items()}

PRODUCT_VT2FUTU = {
    Product.EQUITY: "STOCK",
    Product.INDEX: "IDX",
    Product.ETF: "ETF",
    Product.WARRANT: "WARRANT",
    Product.BOND: "BOND",
}

DIRECTION_VT2FUTU = {
    Direction.LONG: TrdSide.BUY,
    Direction.SHORT: TrdSide.SELL,
}
DIRECTION_FUTU2VT = {v: k for k, v in DIRECTION_VT2FUTU.items()}

STATUS_FUTU2VT = {
    OrderStatus.NONE: Status.SUBMITTING,
    OrderStatus.SUBMITTING: Status.SUBMITTING,
    OrderStatus.SUBMITTED: Status.NOTTRADED,
    OrderStatus.FILLED_PART: Status.PARTTRADED,
    OrderStatus.FILLED_ALL: Status.ALLTRADED,
    OrderStatus.CANCELLED_ALL: Status.CANCELLED,
    OrderStatus.CANCELLED_PART: Status.CANCELLED,
    OrderStatus.SUBMIT_FAILED: Status.REJECTED,
    OrderStatus.FAILED: Status.REJECTED,
    OrderStatus.DISABLED: Status.CANCELLED,
}

KLTYPE_MINUTES = [1, 3, 5, 15, 30, 60]


class FutuGateway(BaseGateway):
    """
    富途证券API
    # 网络访问路径： vnpy=>FutuGateway=>FutuOpenD 本地客户端[端口11111] => 富途证券
    # FutuOpenD下载地址 https://www.futunn.com/download/openAPI?lang=zh-CN
    # windows： 安装完毕后，使用客户端登录=》短信验证=》建立本地11111端口侦听
    """

    default_setting = {
        "密码": "",  # 交易密码
        "地址": "127.0.0.1",
        "端口": 11111,
        "市场": ["HK", "US"],
        "环境": [TrdEnv.REAL, TrdEnv.SIMULATE],
    }

    # 支持的交易所清单
    exchanges = list(EXCHANGE_FUTU2VT.values())

    def __init__(self, event_engine, gateway_name="FUTU"):
        """Constructor"""
        super(FutuGateway, self).__init__(event_engine, gateway_name)

        self.quote_ctx = None
        self.trade_ctx = None

        self.host = ""
        self.port = 0
        self.market = ""
        self.password = ""
        self.env = TrdEnv.SIMULATE

        self.ticks = {}
        self.trades = set()
        self.contracts = {}

        # 引入本地委托单号《=》接口委托单号的管理
        self.order_manager = LocalOrderManager(gateway=self, order_prefix='', order_rjust=4)

        self.thread = Thread(target=self.query_data)

        # For query function.
        self.count = 0
        self.interval = 1
        self.query_funcs = [self.query_account, self.query_position]

    def connect(self, setting: dict):
        """"""
        self.host = setting["地址"]
        self.port = setting["端口"]
        self.market = setting["市场"]
        self.password = setting["密码"]
        self.env = setting["环境"]

        self.connect_quote()
        self.connect_trade()

        self.thread.start()

    def query_data(self):
        """
        使用异步线程单独查询
        Query all data necessary.
        """
        sleep(2.0)  # Wait 2 seconds till connection completed.

        self.query_contract()
        self.query_trade()
        self.query_order()
        self.query_position()
        self.query_account()

        # Start fixed interval query.
        self.event_engine.register(EVENT_TIMER, self.process_timer_event)

    def process_timer_event(self, event):
        """定时器"""
        self.count += 1
        if self.count < self.interval:
            return
        self.count = 0
        func = self.query_funcs.pop(0)
        func()
        self.query_funcs.append(func)

    def connect_quote(self):
        """
        Connect to market data server.
        连接行情服务器
        """

        self.quote_ctx = OpenQuoteContext(self.host, self.port)

        # 股票行情处理的实现
        class QuoteHandler(StockQuoteHandlerBase):
            gateway = self

            # 处理信息回调 =》 gateway.process_quote
            def on_recv_rsp(self, rsp_str):
                ret_code, content = super(QuoteHandler, self).on_recv_rsp(
                    rsp_str
                )
                if ret_code != RET_OK:
                    return RET_ERROR, content
                self.gateway.process_quote(content)
                return RET_OK, content

        # 订单簿的实现
        class OrderBookHandler(OrderBookHandlerBase):
            gateway = self

            # 处理订单簿信息流回调 => gateway.process_orderbook
            def on_recv_rsp(self, rsp_str):
                ret_code, content = super(OrderBookHandler, self).on_recv_rsp(
                    rsp_str
                )
                if ret_code != RET_OK:
                    return RET_ERROR, content
                self.gateway.process_orderbook(content)
                return RET_OK, content

        # 绑定两个实现方法
        self.quote_ctx.set_handler(QuoteHandler())
        self.quote_ctx.set_handler(OrderBookHandler())
        self.quote_ctx.start()

        self.write_log("行情接口连接成功")

    def connect_trade(self):
        """
        Connect to trade server.
        连接交易服务器
        """
        # Initialize context according to market.
        if self.market == "US":
            self.trade_ctx = OpenUSTradeContext(self.host, self.port)
        else:
            self.trade_ctx = OpenHKTradeContext(self.host, self.port)

        # Implement handlers.
        # 订单回报的实现
        class OrderHandler(TradeOrderHandlerBase):
            gateway = self

            # 订单回报流 =》gateway.process_order
            def on_recv_rsp(self, rsp_str):
                ret_code, content = super(OrderHandler, self).on_recv_rsp(
                    rsp_str
                )
                if ret_code != RET_OK:
                    return RET_ERROR, content
                self.gateway.process_order(content)
                return RET_OK, content

        # 交易回报的实现
        class DealHandler(TradeDealHandlerBase):
            gateway = self

            # 成交回报流 =》 gateway.process_deal
            def on_recv_rsp(self, rsp_str):
                ret_code, content = super(DealHandler, self).on_recv_rsp(
                    rsp_str
                )
                if ret_code != RET_OK:
                    return RET_ERROR, content
                self.gateway.process_deal(content)
                return RET_OK, content

        # Unlock to allow trading.
        # 解锁交易接口
        code, data = self.trade_ctx.unlock_trade(self.password)
        if code == RET_OK:
            self.write_log("交易接口解锁成功")
        else:
            self.write_log(f"交易接口解锁失败，原因：{data}")

        # Start context.
        # 绑定订单回报、成交回报
        self.trade_ctx.set_handler(OrderHandler())
        self.trade_ctx.set_handler(DealHandler())
        self.trade_ctx.start()
        self.write_log("交易接口连接成功")

    def subscribe(self, req: SubscribeRequest):
        """订阅行情"""
        for data_type in ["QUOTE", "ORDER_BOOK"]:
            futu_symbol = convert_symbol_vt2futu(req.symbol, req.exchange)
            code, data = self.quote_ctx.subscribe(futu_symbol, data_type, True)

            if code:
                self.write_log(f"订阅行情失败：{data}")

    def query_history(self, req: HistoryRequest):
        """查询某只股票的历史K线数据"""
        history = []
        limit = 60

        if req.interval not in [Interval.MINUTE, Interval.DAILY]:
            self.write_error(f'查询股票历史范围，本接口只支持分钟/日线')
            return history

        futu_code = '{}.{}'.format(EXCHANGE_VT2FUTU.get(req.exchange), req.symbol)

        if req.interval == Interval.MINUTE:
            if req.interval_num not in KLTYPE_MINUTES:
                self.write_error(f'查询股票历史范围，请求分钟数{req.interval_num}不在范围:{KLTYPE_MINUTES}')
                return history
            k_type = f'K_{req.interval_num}M'
        else:
            if req.interval_num != 1:
                self.write_error(f'查询股票历史范围，请求日线{req.interval_num}只能是1')
                return history
            k_type = KLType.K_DAY
        start_date = req.start.strftime('%Y-%m-%d')
        end_date = req.end.strftime('%Y-%m-%d') if req.end else None

        ret, df, page_req_key = self.quote_ctx.request_history_kline(
            code=futu_code,
            ktype=k_type,
            start=start_date,
            end=end_date,
            max_count=limit)  # 每页5个，请求第一页
        if ret == RET_OK:
            for index, row in df.iterrows():
                symbol = row['code']
                str_time = row['time_key']
                dt = datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S')
                bar = BarData(
                    gateway_name=self.gateway_name,
                    symbol=row['code'],
                    exchange=req.exchange,
                    datetime=dt,
                    trading_day=dt.strftime('%Y-%m-%d'),
                    interval=req.interval,
                    interval_num=req.interval_num,
                    volume=row['volume'],
                    open_price=float(row['open']),
                    high_price=float(row['high']),
                    low_price=float(row['low']),
                    close_price=float(row['close'])
                )
                history.append(bar)
        else:
            return history
        while page_req_key != None:  # 请求后面的所有结果
            ret, df, page_req_key = self.quote_ctx.request_history_kline(
                code=futu_code,
                ktype=k_type,
                start=start_date,
                end=end_date,
                page_req_key=page_req_key)  # 请求翻页后的数据
            if ret == RET_OK:
                for index, row in df.iterrows():
                    symbol = row['code']
                    str_time = row['time_key']
                    dt = datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S')
                    bar = BarData(
                        gateway_name=self.gateway_name,
                        symbol=row['code'],
                        exchange=req.exchange,
                        datetime=dt,
                        trading_day=dt.strftime('%Y-%m-%d'),
                        interval=req.interval,
                        interval_num=req.interval_num,
                        volume=row['volume'],
                        open_price=float(row['open']),
                        high_price=float(row['high']),
                        low_price=float(row['low']),
                        close_price=float(row['close'])
                    )
                    history.append(bar)

        return history

    def download_bars(self, req: HistoryRequest):
        """获取某只股票的历史K线数据"""
        history = []
        limit = 60

        if req.interval not in [Interval.MINUTE, Interval.DAILY]:
            self.write_error(f'查询股票历史范围，本接口只支持分钟/日线')
            return history

        futu_code = '{}.{}'.format(EXCHANGE_VT2FUTU.get(req.exchange), req.symbol)

        if req.interval == Interval.MINUTE:
            if req.interval_num not in KLTYPE_MINUTES:
                self.write_error(f'查询股票历史范围，请求分钟数{req.interval_num}不在范围:{KLTYPE_MINUTES}')
                return history
            k_type = f'K_{req.interval_num}M'
        else:
            if req.interval_num != 1:
                self.write_error(f'查询股票历史范围，请求日线{req.interval_num}只能是1')
                return history
            k_type = KLType.K_DAY
        start_date = req.start.strftime('%Y-%m-%d')
        end_date = req.end.strftime('%Y-%m-%d') if req.end else None

        ret, df, page_req_key = self.quote_ctx.request_history_kline(
            code=futu_code,
            ktype=k_type,
            start=start_date,
            end=end_date,
            max_count=limit)  # 每页5个，请求第一页
        if ret == RET_OK:
            for index, row in df.iterrows():
                symbol = row['code']
                str_time = row['time_key']
                dt = datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S')
                bar = OrderedDict({
                    "datetime": str_time,
                    "open": float(row['open']),
                    "close": float(row['close']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "volume": row['volume'],
                    "amount": row['turnover'],
                    "symbol": row['code'],
                    "trading_date": dt.strftime('%Y-%m-%d'),
                    "date": dt.strftime('%Y-%m-%d'),
                    "time": dt.strftime('%H:%M:%S'),
                    "pre_close": float(row['last_close']),
                    "turnover_rate": float(row.get('turnover_rate', 0)),
                    "change_rate": float(row.get('change_rate', 0))

                })
                history.append(bar)
        else:
            return history
        while page_req_key != None:  # 请求后面的所有结果
            ret, df, page_req_key = self.quote_ctx.request_history_kline(
                code=futu_code,
                ktype=k_type,
                start=start_date,
                end=end_date,
                page_req_key=page_req_key)  # 请求翻页后的数据
            if ret == RET_OK:
                for index, row in df.iterrows():
                    symbol = row['code']
                    str_time = row['time_key']
                    dt = datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S')
                    bar = OrderedDict({
                        "datetime": str_time,
                        "open": float(row['open']),
                        "close": float(row['close']),
                        "high": float(row['high']),
                        "low": float(row['low']),
                        "volume": row['volume'],
                        "amount": row['turnover'],
                        "symbol": row['code'],
                        "trading_date": dt.strftime('%Y-%m-%d'),
                        "date": dt.strftime('%Y-%m-%d'),
                        "time": dt.strftime('%H:%M:%S'),
                        "pre_close": float(row['last_close']),
                        "turnover_rate": float(row.get('turnover_rate', 0)),
                        "change_rate": float(row.get('change_rate', 0))
                    })
                    history.append(bar)

        return history

    def send_order(self, req: OrderRequest):
        """发送委托"""
        side = DIRECTION_VT2FUTU[req.direction]
        futu_order_type = OrderType.NORMAL  # Only limit order is supported.

        # Set price adjustment mode to inside adjustment.
        if req.direction is Direction.LONG:
            adjust_limit = 0.05
        else:
            adjust_limit = -0.05

        futu_symbol = convert_symbol_vt2futu(req.symbol, req.exchange)

        # 港股交易手数为整数
        if req.exchange == Exchange.SEHK:
            self.write_log(f'交易手数:{req.volume}=>{int(req.volume)}')
            req.volume = int(req.volume)

        local_orderid = self.order_manager.new_local_orderid()
        order = req.create_order_data(local_orderid, self.gateway_name)

        # 发出委托确认
        order.status = Status.SUBMITTING
        self.order_manager.on_order(order)

        code, data = self.trade_ctx.place_order(
            req.price,
            req.volume,
            futu_symbol,
            side,
            futu_order_type,
            trd_env=self.env,
            adjust_limit=adjust_limit,
        )

        if code:
            self.write_log(f"委托失败：{data}")
            order.status = Status.REJECTED
            self.order_manager.on_order(order)
            return ""

        sys_orderid = ""
        for ix, row in data.iterrows():
            sys_orderid = str(row.get("order_id",""))
            if len(sys_orderid) > 0:
                self.write_log(f'系统委托号:{sys_orderid}')
                break

        if len(sys_orderid) == 0:
            order.status = Status.REJECTED
            self.order_manager.on_order(order)
            return ""

        # 绑定 系统委托号
        order.sys_orderid = sys_orderid
        order.status = Status.NOTTRADED
        self.order_manager.update_orderid_map(local_orderid, sys_orderid)
        # 更新订单为已委托
        self.order_manager.on_order(copy(order))

        return order.vt_orderid

    def cancel_order(self, req: CancelRequest):
        """"""

        order = self.order_manager.get_order_with_local_orderid(req.orderid)

        # 更新订单委托状态为正在撤销
        if order:
            if order.status in [Status.REJECTED, Status.ALLTRADED, Status.CANCELLED]:
                self.write_error(f'委托单:{req.orderid}，状态已经是:{order.status}，不能撤单')
                return False

            order.status = Status.CANCELLING
            self.order_manager.on_order(order)
            sys_orderid = order.sys_orderid
        else:
            sys_orderid = req.orderid

        # 向接口发出撤单请求
        code, data = self.trade_ctx.modify_order(
            ModifyOrderOp.CANCEL, sys_orderid, 0, 0, trd_env=self.env
        )

        if code:
            self.write_log(f"撤单失败：{data}")
            return False
        else:
            self.write_log(f'成功发出撤单请求:orderid={req.orderid},sys_orderid:{sys_orderid}')
            return True

    def query_contract(self):
        """"""
        for product, futu_product in PRODUCT_VT2FUTU.items():
            code, data = self.quote_ctx.get_stock_basicinfo(
                self.market, futu_product
            )

            self.write_log(f'开始查询{futu_product}市场的合约清单')

            if code:
                self.write_log(f"查询合约信息失败：{data}")
                return

            for ix, row in data.iterrows():
                symbol, exchange = convert_symbol_futu2vt(row["code"])
                contract = ContractData(
                    symbol=symbol,
                    exchange=exchange,
                    name=row["name"],
                    product=product,
                    size=1,
                    pricetick=0.001,
                    net_position=True,
                    history_data=True,
                    gateway_name=self.gateway_name,
                )
                self.on_contract(contract)
                self.contracts[contract.vt_symbol] = contract

        self.write_log("合约信息查询成功")

    def query_account(self):
        """"""
        code, data = self.trade_ctx.accinfo_query(trd_env=self.env, acc_id=0)

        if code:
            self.write_log(f"查询账户资金失败：{data}")
            return

        for ix, row in data.iterrows():
            account = AccountData(
                accountid=f"{self.gateway_name}_{self.market}",
                balance=float(row["total_assets"]),
                frozen=(float(row["total_assets"]) - float(row["avl_withdrawal_cash"])),
                gateway_name=self.gateway_name,
            )
            self.on_account(account)

    def query_position(self):
        """"""
        code, data = self.trade_ctx.position_list_query(
            trd_env=self.env, acc_id=0
        )

        if code:
            self.write_log(f"查询持仓失败：{data}")
            return

        for ix, row in data.iterrows():
            symbol, exchange = convert_symbol_futu2vt(row["code"])
            pos = PositionData(
                symbol=symbol,
                exchange=exchange,
                direction=Direction.LONG,
                volume=row["qty"],
                frozen=(float(row["qty"]) - float(row["can_sell_qty"])),
                price=float(row["cost_price"]),
                pnl=float(row["pl_val"]),
                gateway_name=self.gateway_name,
            )

            self.on_position(pos)

    def query_order(self):
        """"""
        code, data = self.trade_ctx.order_list_query("", trd_env=self.env)

        if code:
            self.write_log(f"查询委托失败：{data}")
            return

        self.process_order(data)
        self.write_log("委托查询成功")

    def query_trade(self):
        """"""
        code, data = self.trade_ctx.deal_list_query("", trd_env=self.env)

        if code:
            self.write_log(f"查询成交失败：{data}")
            return

        self.process_deal(data)
        self.write_log("成交查询成功")

    def close(self):
        """"""
        if self.quote_ctx:
            self.quote_ctx.close()

        if self.trade_ctx:
            self.trade_ctx.close()

    def get_tick(self, code):
        """
        Get tick buffer.
        """
        tick = self.ticks.get(code, None)
        symbol, exchange = convert_symbol_futu2vt(code)
        if not tick:
            tick = TickData(
                symbol=symbol,
                exchange=exchange,
                datetime=datetime.now(),
                gateway_name=self.gateway_name,
            )
            self.ticks[code] = tick

        contract = self.contracts.get(tick.vt_symbol, None)
        if contract:
            tick.name = contract.name

        return tick

    def process_quote(self, data):
        """报价推送"""
        for ix, row in data.iterrows():
            symbol = row["code"]

            tick = self.get_tick(symbol)

            date = row["data_date"].replace("-", "")
            time = row["data_time"]
            tick.datetime = datetime.strptime(
                f"{date} {time}", "%Y%m%d %H:%M:%S")
            tick.open_price = row["open_price"]
            tick.high_price = row["high_price"]
            tick.low_price = row["low_price"]
            tick.pre_close = row["prev_close_price"]
            tick.last_price = row["last_price"]
            tick.volume = row["volume"]

            if "price_spread" in row:
                spread = row["price_spread"]
                tick.limit_up = tick.last_price + spread * 10
                tick.limit_down = tick.last_price - spread * 10

            self.on_tick(copy(tick))

    def process_orderbook(self, data):
        """"""
        symbol = data["code"]
        tick = self.get_tick(symbol)

        d = tick.__dict__
        for i in range(5):
            bid_data = data["Bid"][i]
            ask_data = data["Ask"][i]
            n = i + 1

            d["bid_price_%s" % n] = bid_data[0]
            d["bid_volume_%s" % n] = bid_data[1]
            d["ask_price_%s" % n] = ask_data[0]
            d["ask_volume_%s" % n] = ask_data[1]

        if tick.datetime:
            self.on_tick(copy(tick))

    def process_order(self, data):
        """
        Process order data for both query and update.
        """
        for ix, row in data.iterrows():
            # Ignore order with status DELETED
            if row["order_status"] == OrderStatus.DELETED:
                continue

            symbol, exchange = convert_symbol_futu2vt(row["code"])

            # 获取系统委托编号
            sys_orderid = str(row["order_id"])

            # 系统委托变化=》 缓存 order
            order = self.order_manager.get_order_with_sys_orderid(sys_orderid)

            if order is None:
                # 本地委托 《=》系统委托号
                local_orderid = self.order_manager.get_local_orderid(sys_orderid)

                # 创建本地order缓存
                order = OrderData(
                    symbol=symbol,
                    exchange=exchange,
                    orderid=local_orderid,
                    sys_orderid=sys_orderid,
                    direction=DIRECTION_FUTU2VT[row["trd_side"]],
                    price=float(row["price"]),
                    volume=row["qty"],
                    traded=row["dealt_qty"],
                    status=STATUS_FUTU2VT[row["order_status"]],
                    time=row["create_time"].split(" ")[-1],
                    gateway_name=self.gateway_name,
                )
                self.write_log(f'新建委托单缓存=>{order.__dict__}')
                self.order_manager.on_order(copy(order))
            else:
                # 缓存order存在，判断状态、成交数量是否发生变化
                changed = False
                order_status = STATUS_FUTU2VT[row["order_status"]]
                if order.status != order_status:
                    order.status = order_status
                    changed = True
                if order.traded != row["dealt_qty"]:
                    order.traded = row["dealt_qty"]
                    changed = True
                if changed:
                    self.write_log(f'委托单更新=>{order.__dict__}')
                    self.order_manager.on_order(copy(order))

    def process_deal(self, data):
        """
        Process trade data for both query and update.
        """
        for ix, row in data.iterrows():
            # 系统委托编号
            tradeid = str(row["deal_id"])
            if tradeid in self.trades:
                continue

            self.trades.add(tradeid)

            symbol, exchange = convert_symbol_futu2vt(row["code"])

            # 系统委托号
            sys_orderid = row["order_id"]
            # 本地委托号
            local_orderid = self.order_manager.get_local_orderid(sys_orderid)

            trade = TradeData(
                symbol=symbol,
                exchange=exchange,
                direction=DIRECTION_FUTU2VT[row["trd_side"]],
                tradeid=tradeid,
                orderid=local_orderid,
                sys_orderid=sys_orderid,
                price=float(row["price"]),
                volume=row["qty"],
                time=row["create_time"].split(" ")[-1],
                gateway_name=self.gateway_name,
            )

            self.on_trade(trade)


def convert_symbol_futu2vt(code):
    """
    Convert symbol from futu to vt.
    """
    code_list = code.split(".")
    futu_exchange = code_list[0]
    futu_symbol = ".".join(code_list[1:])
    exchange = EXCHANGE_FUTU2VT[futu_exchange]
    return futu_symbol, exchange


def convert_symbol_vt2futu(symbol, exchange):
    """
    Convert symbol from vt to futu.
    """
    futu_exchange = EXCHANGE_VT2FUTU[exchange]
    return f"{futu_exchange}.{symbol}"
