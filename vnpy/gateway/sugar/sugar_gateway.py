"""
食糖购销交易接口
"""

import json
import hashlib
import sys
from copy import copy
from datetime import datetime
import pytz
from typing import Dict, List, Any, Callable, Type, Union
from types import TracebackType
from functools import lru_cache
import requests
import wmi

from vnpy.api.rest import RestClient, Request
from vnpy.trader.constant import (
    Direction,
    Offset,
    Exchange,
    Product,
    Status
)
from vnpy.trader.gateway import BaseGateway
from vnpy.trader.event import EVENT_TIMER
from vnpy.trader.object import (
    TickData,
    OrderData,
    TradeData,
    AccountData,
    PositionData,
    ContractData,
    BarData,
    OrderRequest,
    CancelRequest,
    SubscribeRequest,
    HistoryRequest
)


# REST_HOST = "http://seedtest.ap-ec.cn/CT-OPEN-SERVER"       # 测试环境访问主路径
REST_HOST = "http://seed.ap-ec.cn/CT-OPEN-SERVER"           # 生产环境访问主路径

CALLBACK_TYPE = Callable[[dict, "Request"], Any]
ON_FAILED_TYPE = Callable[[int, "Request"], Any]
ON_ERROR_TYPE = Callable[[Type, Exception, TracebackType, "Request"], Any]

STATUS_SUGAR2VT = {
    "已报": Status.NOTTRADED,
    "部撤": Status.CANCELLED,
    "已撤": Status.CANCELLED,
    "部成": Status.PARTTRADED,
    "已成": Status.ALLTRADED,
    "废单": Status.REJECTED,
}

DIRECTION_VT2SUGAR = {Direction.LONG: "1", Direction.SHORT: "2"}
DIRECTION_SUGAR2VT = {v: k for k, v in DIRECTION_VT2SUGAR.items()}

OFFSET_VT2SUGAR = {Offset.OPEN: "1", Offset.CLOSE: "2"}
OFFSET_SUGAR2VT = {v: k for k, v in OFFSET_VT2SUGAR.items()}

CHINA_TZ = pytz.timezone("Asia/Shanghai")


class SugarGateway(BaseGateway):
    """
    食糖购销交易接口
    """

    default_setting: Dict[str, Any] = {
        "开放账号": "",
        "加密KEY": "",
        "TOKEN": "",
        "会话线程": 8,
    }

    exchanges: List[Exchange] = [Exchange.SR]

    def __init__(self, event_engine):
        """Constructor"""
        super().__init__(event_engine, "SUGAR")
        self.rest_api = SugarRestApi(self)        
        self.orders: Dict[str, OrderData] = {}
    
    def get_order(self, orderid: str) -> OrderData:
        """"""
        return self.orders.get(orderid, None)

    def on_order(self, order: OrderData) -> None:
        """"""
        self.orders[order.orderid] = order
        super().on_order(order)

    def connect(self, setting: dict) -> None:
        """"""
        open_account = setting["开放账号"]
        key = setting["加密KEY"]
        token = setting["TOKEN"]
        session_number = setting["会话线程"]

        self.rest_api.connect(open_account, key, token, session_number)

        self.init_query()

    def subscribe(self, req: SubscribeRequest) -> None:
        """"""
        self.rest_api.subscribe(req)

    def send_order(self, req: OrderRequest) -> str:
        """"""
        return self.rest_api.send_order(req)

    def cancel_order(self, req: CancelRequest) -> None:
        """"""
        self.rest_api.cancel_order(req)
    
    def query_trade(self) -> None:
        """"""
        self.rest_api.query_trade()

    def query_order(self) -> None:
        """"""
        self.rest_api.query_order()

    def query_quotes(self) -> None:
        """"""
        self.rest_api.query_quotes()

    def query_position(self) -> None:
        """"""
        self.rest_api.query_position()

    def query_account(self) -> None:
        """"""
        self.rest_api.query_account()

    def query_history(self, req: HistoryRequest):
        """"""
        pass

    def close(self) -> None:
        """"""
        self.rest_api.stop()
        self.rest_api.join()

    def process_timer_event(self, event):
        """"""
        for func in self.query_functions:
            func()

    def init_query(self):
        """"""        
        self.query_functions = [self.query_order, self.query_trade, self.query_quotes, self.query_position, self.query_account]

        self.event_engine._interval = 1
        self.event_engine.register(EVENT_TIMER, self.process_timer_event)


class SugarRestApi(RestClient):
    """
    SUGAR REST API
    """

    def __init__(self, gateway: BaseGateway):
        """"""
        super().__init__()

        self.gateway: SugarGateway = gateway
        self.gateway_name: str = gateway.gateway_name

        self.open_account: str = ""
        self.key: str = ""
        self.token: str = ""
        self.ip_address: str = self.get_ip_address()
        self.mac_address: str = self.get_mac_address()

        self.subscribe_symbol: set = set()
        self.trade_businessNo: set = set()

        self.order_count = 0

        self.callback_dt = None
        self.up_dn_limit = {}

    def connect(
        self,
        open_account: str,
        key: str,
        token: str,
        session_number: int,
    ) -> None:
        """
        Initialize connection to REST server.
        """
        self.open_account = open_account
        self.key = key
        self.token = token

        self.init(REST_HOST)
        self.start(session_number)

        self.gateway.write_log("食糖购销接口启动成功")

        self.query_contract()

    def query_contract(self) -> None:
        """"""
        requestBody={
            "marketId": "000"
        }

        self.add_request(
            method="POST",
            path="/quotes/getQuotes.magpie",
            data=requestBody,
            callback=self.on_query_contract
        )

    def query_account(self) -> None:
        """"""
        requestBody = {
            "marketId":"000"
        }

        self.add_request(
            method="POST",
            path="/hps/getAccountInfo.magpie",
            data=requestBody,
            callback=self.on_query_account
        )

    def query_position(self) -> None:
        """"""
        requestBody={
            "marketId": "000",
        }

        self.add_request(
            method="POST",
            path="/ct/postionListByGoodsCode.magpie",
            data=requestBody,
            callback=self.on_query_position
        )

    def query_quotes(self) -> None:
        """"""
        requestBody = {
            "marketId": "000"
        }

        self.add_request(
            method="POST",
            path="/quotes/getQuotes.magpie",
            data=requestBody,
            callback=self.on_query_quotes
        )

    def query_order(self) -> None:
        """"""
        requestBody = {
            "marketId":"000",
            "currentPage":"1",
            "pageSize":"10000",
        }

        self.add_request(
            method="POST",
            path="/ct/entrustGridList.magpie",
            data=requestBody,
            callback=self.on_query_order
        )

        if self.callback_dt and (datetime.now() - self.callback_dt).seconds > 5:
            self.gateway.write_log("接口请求响应间隔超过5秒，发生阻塞！")
            print("接口请求响应间隔超过5秒，发生阻塞！")

    def query_trade(self) -> None:
        """"""
        requestBody = {
            "marketId":"000",
            "currentPage":"1",
            "pageSize":"10000",
        }

        self.add_request(
            method="POST",
            path="/ct/tradeGridDetailList.magpie",
            data=requestBody,
            callback=self.on_query_trade
        )

    def query_history(self, req: HistoryRequest) -> List[BarData]:
        """"""
        pass

    def subscribe(self, req: SubscribeRequest) -> None:
        """"""
        self.subscribe_symbol.add(req.symbol)

    def send_order(self, req: OrderRequest) -> str:
        """"""
        orderid = self.new_orderid()
        order: OrderData = req.create_order_data(orderid, self.gateway_name)
        order.datetime = CHINA_TZ.localize(datetime.now())
        order.price = int(order.price)
        order.volume = int(order.volume)

        requestBody = {
            "goodscode": req.symbol,
            "entrustPrice": int(req.price * 100),
            "entrustAmountH": int(req.volume),
            "entrustBs": DIRECTION_VT2SUGAR.get(req.direction, ""),
            "entrustTs": OFFSET_VT2SUGAR.get(req.offset, ""),
            "opEntrustWay": "0",
            "entrustWay": "1",
            "macAddress": self.mac_address,
            "ipAddress": self.ip_address
        }

        self.add_request(
            method="POST",
            path="/ct/tradeEntrustOrder.magpie",
            callback=self.on_send_order,
            data=requestBody,
            extra=order,
            on_error=self.on_send_order_error,
            on_failed=self.on_send_order_failed
        )
        self.gateway.on_order(order)

        return order.vt_orderid

    def cancel_order(self, req: CancelRequest) -> None:
        """"""
        order = self.gateway.orders.get(req.orderid, None)

        if not order or not order.entrustNo:
            self.gateway.write_log("未找到对应委托，无法撤销。")
            return

        requestBody = {
            "entrustNo": order.entrustNo,
            "macAddress": self.mac_address,
            "ipAddress": self.ip_address,
            "opEntrustWay": "0"
        }

        self.add_request(
            method="POST",
            path="/ct/cancelEntrustOrder.magpie",
            data=requestBody,
            callback=self.on_cancel_order,
            extra=req
        )

    def on_query_contract(self, data: dict, request: Request) -> None:
        """"""
        if self.check_error(data, "查询合约"):
            return

        for d in data["data"]["responseBody"]:
            if not d["goodsCode"].startswith("GS"):
                continue

            contract = ContractData(
                symbol=d["goodsCode"],
                exchange=Exchange.SR,
                name=d["goodsName"],
                pricetick=1,
                size=10,
                min_volume=1,
                margin_ratio=0.1,
                product=Product.SPOT,
                stop_supported=False,
                net_position=False,
                history_data=False,
                gateway_name=self.gateway_name,
            )

            self.gateway.on_contract(contract)

            # 查询购销计划涨跌停价格
            requestBody = {"goodsCode": d["goodsCode"]}
            signature = self.create_signature(requestBody)
            request_time = datetime.strftime(datetime.now(),"%Y-%m-%d %H:%M:%S.%f")

            request_data = {
                "requestHeader":{
                    "token": self.token,
                    "sign": signature,
                    "yqMemberId": self.open_account,
                    "merRequestNo": request_time,
                    "merRequestTime": request_time[:-7]
                },
                "requestBody": requestBody
            }

            url = self.url_base + "/ct/selectGoodsInfo.magpie"

            response = requests.post(url=url, json=request_data, headers={'Content-Type':'application/json'})
            response_data = response.json()
            
            try:
                self.up_dn_limit.update({
                    d["goodsCode"]:{
                        "limit_up": response_data["data"]["responseBody"]["uplimitedPrice"] / 100,
                        "limit_down": response_data["data"]["responseBody"]["downlimitedPrice"] / 100,
                    }
                })
            except:
                pass

        self.gateway.write_log("合约信息查询成功")

    def on_query_account(self, data: dict, request: Request) -> None:
        """"""
        if self.check_error(data, "查询账户"):
            return
        
        for d in data["data"]["responseBody"]:
            account = AccountData(
                accountid=self.open_account,
                balance=d["currentBalance"] / 100,
                pre_balance=d["beginBalance"] / 100,
                available=d["canUseBalance"] / 100,
                frozen=d["forbidBalance"] / 100,
                commission=d["payFee"] / 100,
                margin=(d["lxExDepositBanlance"] + d["lxPreDepositBanlance"]) / 100,
                date=str(datetime.now().date()),
                time=str(datetime.now().time()),
                gateway_name=self.gateway_name,
            )

            try:
                account.percent = round(1 - account.available / account.balance,3) * 100      #资金使用率
            except ZeroDivisionError:
                account.percent = 0

            self.gateway.on_account(account)

    def on_query_position(self, data: dict, request: Request) -> None:
        """"""
        if self.check_error(data, "查询持仓"):
            return

        for d in data["data"]["responseBody"][::-1]:
            if not d.get("goodsCode", None):
                continue

            if d["buyHoldAmount"]:
                long_position = PositionData(
                    gateway_name=self.gateway_name,
                    symbol=d["goodsCode"],
                    exchange=Exchange.SR,
                    direction=Direction.LONG,
                    volume=d["buyHoldAmount"],
                    frozen=d["buyLockedAmount"],
                    price=d["buyAvgPrice"] / 100,
                    pnl=0,
                )

                self.gateway.on_position(long_position)

            if d["sellHoldAmount"]:
                short_position = PositionData(
                    gateway_name=self.gateway_name,
                    symbol=d["goodsCode"],
                    exchange=Exchange.SR,
                    direction=Direction.SHORT,
                    volume=d["sellHoldAmount"],
                    frozen=d["sellLockedAmount"],
                    price=d["sellAvgPrice"] / 100,
                    pnl=0,
                )

                self.gateway.on_position(short_position)

    def on_query_quotes(self, data: dict, request: Request) -> None:
        """"""
        if self.check_error(data, "查询行情"):
            return

        for d in data["data"]["responseBody"]:
            if d["goodsCode"] not in self.subscribe_symbol:
                continue
            
            dt = CHINA_TZ.localize(datetime.now())

            tick = TickData(
                symbol=d["goodsCode"],
                exchange=Exchange.SR,
                datetime=dt,
                name=d["goodsName"],
                volume=int(d["transactionVolume"]),
                # open_interest=d["currentQuantity"],
                last_price=int(d["newDealPrice"]),
                limit_up=self.up_dn_limit.get(d["goodsCode"], 0)["limit_up"],
                limit_down=self.up_dn_limit.get(d["goodsCode"], 0)["limit_down"],
                open_price=int(d["openingPrice"]),
                high_price=int(d["highestPrice"]),
                low_price=int(d["lowestPrice"]),
                pre_close=int(d["closePrice"]),
                bid_price_1=int(d["buyPrice1"]),
                ask_price_1=int(d["sellPrice1"]),
                bid_volume_1=int(d["buyContractVolume1"]),
                ask_volume_1=int(d["sellContractVolume1"]),
                gateway_name=self.gateway_name
            )

            self.gateway.on_tick(tick)

    def on_query_order(self, data: dict, request: Request) -> None:
        """"""
        if self.check_error(data, "查询委托"):
            return

        responseBody = data["data"]["responseBody"]
        if not responseBody.get("items", None):
            return

        for d in responseBody["items"][::-1]:
            timestamp = f'{d["tradingDate"]} {d["entrustTime"]}'
            dt = CHINA_TZ.localize(datetime.strptime(timestamp, "%Y%m%d %H%M%S"))

            entrustNo = str(d["entrustNo"])
            orderid = self.gateway.orders.get(entrustNo, None)

            if not orderid:
                orderid = self.new_orderid()
                            
                order = OrderData(
                    gateway_name=self.gateway_name,
                    symbol=d["goodsCode"],
                    exchange=Exchange.SR,
                    orderid=orderid,
                    direction=DIRECTION_SUGAR2VT.get(str(d["entrustBs"]), None),
                    offset=OFFSET_SUGAR2VT.get(str(d["entrustTs"]), None),
                    price=int(d["entrustPrice"] / 100),
                    volume=int(d["entrustAmount"]),
                    traded=int(d["businessAmount"]),
                    status=STATUS_SUGAR2VT.get(d["entrustStatusStr"], None),
                    datetime=dt
                )
                order.entrustNo = entrustNo
                self.gateway.orders[entrustNo] = orderid

                self.gateway.on_order(order)

            else:
                order: OrderData = self.gateway.orders.get(orderid, None)
                if order.status == Status.SUBMITTING:
                    if d["entrustStatusStr"] == "已报":
                        order.status = Status.NOTTRADED
                        self.gateway.on_order(order)

                    elif d["entrustStatusStr"] == "已成":
                        order.status = Status.ALLTRADED
                        order.traded = int(d["businessAmount"])
                        self.gateway.on_order(order)
                    
                    elif d["entrustStatusStr"] == "部成":
                        order.status = Status.PARTTRADED
                        order.traded = int(d["businessAmount"])
                        self.gateway.on_order(order)

                    elif d["entrustStatusStr"] == "废单":
                        order.status = Status.REJECTED
                        self.gateway.on_order(order)

                elif order.status == Status.NOTTRADED and d["entrustStatusStr"] != "已报":
                    if d["entrustStatusStr"] == "已撤" or d["entrustStatusStr"] == "部撤":
                        order.status = Status.CANCELLED
                        self.gateway.on_order(order)

                    elif d["entrustStatusStr"] == "已成":
                        order.status = Status.ALLTRADED
                        order.traded = int(d["businessAmount"])
                        self.gateway.on_order(order)
                    
                    elif d["entrustStatusStr"] == "部成":
                        order.status = Status.PARTTRADED
                        order.traded = int(d["businessAmount"])
                        self.gateway.on_order(order)

                    elif d["entrustStatusStr"] == "废单":
                        order.status = Status.REJECTED
                        self.gateway.on_order(order)

                elif order.status == Status.PARTTRADED:
                    if d["entrustStatusStr"] == "已成":
                        order.status = Status.ALLTRADED
                        order.traded = int(d["businessAmount"])
                        self.gateway.on_order(order)

                    elif d["entrustStatusStr"] == "部成" and order.traded < int(d["businessAmount"]):
                        order.status = Status.PARTTRADED
                        order.traded = int(d["businessAmount"])
                        self.gateway.on_order(order)
                        
                    elif d["entrustStatusStr"] == "部撤":
                        order.status = Status.CANCELLED
                        self.gateway.on_order(order)

        self.callback_dt = datetime.now()

    def on_query_trade(self, data: dict, request: Request) -> None:
        """"""
        if self.check_error(data, "查询成交"):
            return

        responseBody = data["data"]["responseBody"]
        if not responseBody.get("items", None):
            return

        for d in responseBody["items"][::-1]:
            orderid = self.gateway.orders.get(str(d["entrustNo"]), None)
            if not orderid:
                continue

            businessNo = d["businessNo"]
            
            if businessNo not in self.trade_businessNo:
                timestamp = f'{d["tradingDate"]} {d["businessTime"]}'
                dt = CHINA_TZ.localize(datetime.strptime(timestamp, "%Y%m%d %H%M%S"))
                order: OrderData = self.gateway.orders.get(orderid, None)                

                trade = TradeData(
                    symbol=order.symbol,
                    exchange=Exchange.SR,
                    orderid=order.orderid,
                    tradeid=businessNo,
                    direction=order.direction,
                    offset=order.offset,
                    price=int(d["businessPrice"]),
                    volume=int(d["businessAmount"]),
                    datetime=dt,
                    gateway_name=self.gateway_name,
                )

                self.trade_businessNo.add(businessNo)
                self.gateway.on_trade(trade)
            
    def on_send_order(self, data: dict, request: Request) -> None:
        """"""
        order: OrderData = request.extra

        if self.check_error(data, "委托"):
            order.status = Status.REJECTED
            self.gateway.on_order(order)
            return

        entrustNo = str(data["data"]["responseBody"]["entrustNo"])
        order.entrustNo = entrustNo
        self.gateway.orders[entrustNo] = order.orderid

        self.gateway.on_order(order)

    def on_send_order_failed(self, status_code: str, request: Request) -> None:
        """
        Callback when sending order failed on server.
        """
        order = request.extra
        order.status = Status.REJECTED
        self.gateway.on_order(order)

        msg = f"委托失败，状态码：{status_code}，信息：{request.response.text}"
        self.gateway.write_log(msg)

    def on_send_order_error(
        self,
        exception_type: type,
        exception_value: Exception,
        tb,
        request: Request
    ) -> None:
        """
        Callback when sending order caused exception.
        """
        order = request.extra
        order.status = Status.REJECTED
        self.gateway.on_order(order)

        # Record exception if not ConnectionError
        if not issubclass(exception_type, ConnectionError):
            self.on_error(exception_type, exception_value, tb, request)

    def on_cancel_order(self, data: dict, request: Request) -> None:
        """"""
        cancel_request = request.extra
        order = self.gateway.get_order(cancel_request.orderid)
        if not order:
            return

        if self.check_error(data, "撤单"):
            order.status = Status.REJECTED
        else:
            order.status = Status.CANCELLED
            self.gateway.write_log(f"委托撤单成功：{order.orderid}")

        self.gateway.on_order(order)

    def on_error(
        self,
        exception_type: type,
        exception_value: Exception,
        tb,
        request: Request
    ) -> None:
        """
        Callback to handler request exception.
        """
        msg = f"触发异常，状态码：{exception_type}，信息：{exception_value}"
        self.gateway.write_log(msg)

        sys.stderr.write(
            self.exception_detail(exception_type, exception_value, tb, request)
        )

    def check_error(self, data: dict, func: str = "") -> bool:
        """"""
        if data["succeed"]:
            return False

        error_code = data["errorCode"]
        error_msg = data["errorMsg"]

        self.gateway.write_log(f"{func}请求出错，代码：{error_code}，信息：{error_msg}")
        return True
    
    def new_orderid(self):
        """"""
        prefix = datetime.now().strftime("%Y%m%d-%H%M%S-")

        self.order_count += 1
        suffix = str(self.order_count).rjust(8, "0")

        orderid = prefix + suffix
        return orderid

    def sign(self, request: Request) -> Request:
        """
        Generate SUGAR signature.
        """
        signature = self.create_signature(request.data)
        request_time = datetime.strftime(datetime.now(),"%Y-%m-%d %H:%M:%S.%f")

        request_data = {
            "requestHeader":{
                "token": self.token,
                "sign": signature,
                "yqMemberId": self.open_account,
                "merRequestNo": request_time,
                "merRequestTime": request_time[:-7]
            },
            "requestBody": request.data
        }

        request.headers = {"Content-Type": "application/json"}
        request.data = json.dumps(request_data)

        return request

    def create_signature(self, requestBody: dict) -> str:
        body_data={}
        for key, value in requestBody.items():
            if value != "":
                body_data[key] = value
        
        body_str = ""
        for key in sorted(body_data.keys()):
            body_str += key + "=" + str(body_data[key]) + "&"
        
        body_str = (body_str[:-1] + self.key).lower().replace(" ", "")
        sign_str = get_sha1_secret_str(body_str)

        return sign_str
        
    def get_ip_address(self):
        """获取计算机公网IP地址"""
        f = requests.get("http://myip.dnsomatic.com")
        ip_address = f.text
        return ip_address
        
    def get_mac_address(self):
        """获取计算机MAC物理地址(CMD运行"getmac"获取物理地址)"""
        c = wmi.WMI()

        mac_address = ""
        for interface in c.Win32_NetworkAdapterConfiguration(IPEnabled=1):
            mac_address = interface.MACAddress

        return mac_address


@lru_cache(maxsize=999, typed=True)
def get_sha1_secret_str(body_str:str):
    """
    使用sha1加密算法，返回str加密后的字符串
    """
    sha = hashlib.sha1(body_str.encode('utf-8'))
    encrypts = sha.hexdigest()
    return encrypts
