# 恒投交易客户端 文件接口
# 华富资产 李来佳 28888502

import os
import sys
import copy
import csv
import traceback
from typing import Any, Dict, List
from datetime import datetime, timedelta
from time import sleep
from functools import lru_cache
from collections import OrderedDict
from multiprocessing.dummy import Pool

from vnpy.event import EventEngine
from vnpy.trader.event import EVENT_TIMER
from vnpy.trader.constant import (
    Exchange,
    Product,
    Direction,
    OrderType,
    Status,
    Offset
)
from vnpy.trader.gateway import BaseGateway, LocalOrderManager
from vnpy.trader.object import (
    CancelRequest,
    OrderRequest,
    SubscribeRequest,
    TickData,
    ContractData,
    OrderData,
    TradeData,
    PositionData,
    AccountData
)
from vnpy.trader.utility import get_folder_path, print_dict, extract_vt_symbol, get_stock_exchange, append_data

# 代码 <=> 中文名称
symbol_name_map: Dict[str, str] = {}
# 代码 <=> 交易所
symbol_exchange_map: Dict[str, Exchange] = {}

# 功能<->文件对应
PB_FILE_NAMES = {
    'send_order': 'XHPT_WT',  # 通用接口_委托
    'cancel_order': 'XHPT_CD',  # 通用接口_撤单
    'update_orders': 'XHPT_WTCX',  # 通用接口_委托查询
    'update_trades': 'XHPT_CJCX',  # 通用接口_成交查询
    'positions': 'CC_STOCK_',  # 持仓明细
    'orders': 'WT_STOCK_',  # 当日委托明细
    'trades': 'CJ_STOCK_',  # 当日成交明细
    'accounts': 'ZJ_STOCK_'  # 资金
}

SEND_ORDER_FIELDS = OrderedDict({
    "CPBH": "C32",  # 产品代码/基金代码 <-- 输入参数 -->
    "ZCDYBH": "C16",  # 单元编号/组合编号
    "ZHBH": "C16",  # 组合编号
    "GDDM": "C20",  # 股东代码
    "JYSC": "C3",  # 交易市场
    "ZQDM": "C16",  # 证券代码
    "WTFX": "C4",  # 委托方向
    "WTJGLX": "C1",  # 委托价格类型
    "WTJG": "N11.4",  # 委托价格
    "WTSL": "N12",  # 委托数量
    "WBZDYXH": "N9",  # 第三方系统自定义号
    "WTXH": "N8",  # 委托序号  <-- 输出参数  -->
    "WTSBDM": "N8",  # 委托失败代码
    "SBYY": "C254",  # 失败原因
    "CLBZ": "C1",  # 处理标志  <-- 内部自用字段 -->
    "BYZD": "C2",  # 备用字段
    "WTJE": "N16.2",  # 委托金额  <-- 扩充参数 -->
    "TSBS": "C64",  # 特殊标识
    "YWBS": "C2",  # 业务标识
})

# 撤单csv字段格式定义
CANCEL_ORDER_FIELDS = OrderedDict({
    "WTXH": "N8",  # 委托序号
    "JYSC": "C3",  # 交易市场
    "ZQDM": "C16",  # 证券代码
    "CDCGBZ": "C1",  # 撤单成功标志
    "SBYY": "C254",  # 失败原因
    "CLBZ": "C1",  # 处理标志
    "BYZD": "C2",  # 备用字段
    "BYZD2": "C16",  # 备用字段2
})
# 交易所id <=> Exchange
EXCHANGE_PB2VT: Dict[str, Exchange] = {
    "1": Exchange.SSE,
    "2": Exchange.SZSE,
    "3": Exchange.SHFE,
    "4": Exchange.CZCE,
    "7": Exchange.CFFEX,
    "9": Exchange.DCE,
    "k": Exchange.INE
}
EXCHANGE_VT2PB: Dict[Exchange, str] = {v: k for k, v in EXCHANGE_PB2VT.items()}
EXCHANGE_NAME2VT: Dict[str, Exchange] = {
    "上交所A": Exchange.SSE,
    "深交所A": Exchange.SZSE
}

# 方向  <=> Direction, Offset
DIRECTION_STOCK_PB2VT: Dict[str, Any] = {
    "1": (Direction.LONG, Offset.NONE),  # 买
    "2": (Direction.SHORT, Offset.NONE),  # 卖
    "V": (Direction.LONG, Offset.OPEN),  # 多，开
    "X": (Direction.SHORT, Offset.OPEN),  # 空，开
    "Y": (Direction.LONG, Offset.CLOSE),  # 多，平
    "W": (Direction.SHORT, Offset.CLOSE)  # 空， 平
}
DIRECTION_STOCK_VT2PB: Dict[Any, str] = {v: k for k, v in DIRECTION_STOCK_PB2VT.items()}
DIRECTION_STOCK_NAME2VT: Dict[str, Any] = {
    "卖出": Direction.SHORT,
    "买入": Direction.LONG
}
# 持仓方向 <=> Direction
POSITION_DIRECTION_PB2VT = {
    "1": Direction.LONG,
    "2": Direction.SHORT,
}

# 委托单类型
ORDERTYPE_PB2VT: Dict[str, OrderType] = {
    "0": OrderType.LIMIT,  # 限价单
    "a": OrderType.MARKET,  # 五档即成剩撤（上交所市价）
    "b": OrderType.MARKET,  # 五档即成剩转（上交所市价）
    "A": OrderType.MARKET,  # 五档即成剩撤（深交所市价）
    "C": OrderType.MARKET,  # 即成剩撤（深交所市价）
    "D": OrderType.MARKET,  # 对手方最优（深交所市价，上交所科创板市价）
    "E": OrderType.MARKET,  # 本方最优（深交所市价，上交所科创板市价）
}


def format_dict(d, dict_define):
    """根据dict格式定义进行value转换"""

    for k in dict_define.keys():
        # 原值
        v = d.get(k, '')
        # 目标转换格式
        v_format = dict_define.get(k, None)
        if v_format is None:
            continue
        if 'C' in v_format:
            str_len = int(v_format.replace('C', ''))
            new_v = '{}{}'.format(' ' * (str_len - len(v)), v)
            d.update({k: new_v})
            continue
        elif "N" in v_format:
            v_format = v_format.replace('N', '')
            if '.' in v_format:
                int_len, float_len = v_format.split('.')
                int_len = int(int_len)
                float_len = int(float_len)
                str_v = str(v)
                new_v = '{}{}'.format(' ' * (int_len - len(str_v)), str_v)
            else:
                int_len = int(v_format)
                str_v = str(v)
                new_v = '{}{}'.format(' ' * (int_len - len(str_v)), str_v)
            d.update({k: new_v})

    return d


def get_pb_order_type(exchange, order_type):
    """获取pb的委托类型"""
    # 限价单
    if order_type == OrderType.LIMIT:
        return "0"
    # 市价单
    if exchange == Exchange.SSE:
        return "a"

    if exchange == Exchange.SZSE:
        return "C"

    return "0"


ORDERTYPE_NAME2VT: Dict[str, OrderType] = {
    "五档即成剩撤": OrderType.MARKET,
    "五档即成剩转": OrderType.MARKET,
    "即成剩撤": OrderType.MARKET,
    "对手方最优": OrderType.MARKET,
    "本方最优": OrderType.MARKET,
    "限价单": OrderType.LIMIT,
}

STATUS_NAME2VT: Dict[str, Status] = {
    "未报": Status.SUBMITTING,
    "待报": Status.SUBMITTING,
    "正报": Status.SUBMITTING,
    "已报": Status.NOTTRADED,
    "废单": Status.REJECTED,
    "部成": Status.PARTTRADED,
    "已成": Status.ALLTRADED,
    "部撤": Status.CANCELLED,
    "已撤": Status.CANCELLED,
    "待撤": Status.CANCELLING,
    "未审批": Status.UNKNOWN,
    "审批拒绝": Status.UNKNOWN,
    "未审批即撤销": Status.UNKNOWN,
}

STOCK_CONFIG_FILE = 'tdx_stock_config.pkb2'
from pytdx.hq import TdxHq_API
# 通达信股票行情
from vnpy.data.tdx.tdx_common import get_cache_config, get_tdx_market_code
from pytdx.config.hosts import hq_hosts
from pytdx.params import TDXParams


class PbGateway(BaseGateway):
    default_setting: Dict[str, Any] = {
        "资金账号": "",
        "数据目录": "",
        "产品编号": "",
        "单元编号": "",
        "股东代码_沪": "",
        "股东代码_深": ""
    }

    # 接口支持得交易所清单
    exchanges: List[Exchange] = list(EXCHANGE_VT2PB.keys())

    def __init__(self, event_engine: EventEngine, gateway_name='PB'):
        """"""
        super().__init__(event_engine, gateway_name=gateway_name)
        self.connect_time = datetime.now().strftime("%H%M")
        self.order_manager = LocalOrderManager(self, self.connect_time, 4)

        self.md_api = PbMdApi(self)
        self.td_api = PbTdApi(self)

        self.tdx_connected = False  # 通达信行情API得连接状态

    def connect(self, setting: dict) -> None:
        """"""
        userid = setting["资金账号"]
        csv_folder = setting["数据目录"]
        product_id = setting["产品编号"]
        unit_id = setting["单元编号"]
        holder_ids = {
            Exchange.SSE: setting["股东代码_沪"],
            Exchange.SZSE: setting["股东代码_深"]
        }

        export_folder = os.path.abspath(os.path.join(csv_folder, "数据导出"))
        self.md_api.connect()
        self.td_api.connect(user_id=userid,
                            order_folder=csv_folder,
                            account_folder=export_folder,
                            product_id=product_id,
                            unit_id=unit_id,
                            holder_ids=holder_ids)
        self.init_query()

    def close(self) -> None:
        """"""
        self.md_api.close()
        self.td_api.close()

    def subscribe(self, req: SubscribeRequest) -> None:
        """"""
        self.md_api.subscribe(req)

    def send_order(self, req: OrderRequest) -> str:
        """"""
        return self.td_api.send_order(req)

    def cancel_order(self, req: CancelRequest) -> None:
        """"""
        self.td_api.cancel_order(req)

    def query_account(self) -> None:
        """"""
        self.td_api.query_account()

    def query_position(self) -> None:
        """"""
        self.td_api.query_position()

    def query_orders(self) -> None:
        self.td_api.query_orders()

    def query_trades(self) -> None:
        self.td_api.query_trades()

    def process_timer_event(self, event) -> None:
        """"""
        self.count += 1
        if self.count < 2:
            return
        self.count = 0

        func = self.query_functions.pop(0)
        func()
        self.query_functions.append(func)

    def init_query(self) -> None:
        """"""
        self.count = 0
        self.query_functions = [self.query_account, self.query_position, self.query_orders, self.query_trades]
        self.event_engine.register(EVENT_TIMER, self.process_timer_event)


class PbMdApi(object):

    def __init__(self, gateway: BaseGateway):
        """"""
        super().__init__()

        self.gateway: BaseGateway = gateway
        self.gateway_name: str = gateway.gateway_name

        self.connect_status: bool = False
        self.login_status: bool = False

        self.req_interval = 0.5  # 操作请求间隔500毫秒
        self.req_id = 0  # 操作请求编号
        self.connection_status = False  # 连接状态

        self.symbol_exchange_dict = {}  # tdx合约与vn交易所的字典
        self.symbol_market_dict = {}  # tdx合约与tdx市场的字典
        self.symbol_vn_dict = {}  # tdx合约与vtSymbol的对应
        self.symbol_tick_dict = {}  # tdx合约与最后一个Tick得字典
        self.registed_symbol_set = set()

        self.config = get_cache_config(STOCK_CONFIG_FILE)
        self.symbol_dict = self.config.get('symbol_dict', {})
        self.cache_time = self.config.get('cache_time', datetime.now() - timedelta(days=7))
        self.commission_dict = {}
        self.contract_dict = {}

        if len(self.symbol_dict) == 0:  # or self.cache_time < datetime.now() - timedelta(days=1):
            # self.cache_config()
            self.gateway.write_error(f'本地没有股票信息的缓存配置文件')

        # self.queue = Queue()            # 请求队列
        self.pool = None  # 线程池
        # self.req_thread = None          # 定时器线程

        # copy.copy(hq_hosts)

        self.ip_list = [{'ip': "180.153.18.170", 'port': 7709},
                        {'ip': "180.153.18.171", 'port': 7709},
                        {'ip': "180.153.18.172", 'port': 80},
                        {'ip': "202.108.253.130", 'port': 7709},
                        {'ip': "202.108.253.131", 'port': 7709},
                        {'ip': "202.108.253.139", 'port': 80},
                        {'ip': "60.191.117.167", 'port': 7709},
                        {'ip': "115.238.56.198", 'port': 7709},
                        {'ip': "218.75.126.9", 'port': 7709},
                        {'ip': "115.238.90.165", 'port': 7709},
                        {'ip': "124.160.88.183", 'port': 7709},
                        {'ip': "60.12.136.250", 'port': 7709},
                        {'ip': "218.108.98.244", 'port': 7709},
                        {'ip': "218.108.47.69", 'port': 7709},
                        {'ip': "114.80.63.12", 'port': 7709},
                        {'ip': "114.80.63.35", 'port': 7709},
                        {'ip': "180.153.39.51", 'port': 7709},
                        {'ip': '14.215.128.18', 'port': 7709},
                        {'ip': '59.173.18.140', 'port': 7709}]

        self.best_ip = {'ip': None, 'port': None}
        self.api_dict = {}  # API 的连接会话对象字典
        self.last_tick_dt = {}  # 记录该会话对象的最后一个tick时间

        self.security_count = 50000

        # 股票code name列表
        self.stock_codelist = None

    def ping(self, ip, port=7709):
        """
        ping行情服务器
        :param ip:
        :param port:
        :param type_:
        :return:
        """
        apix = TdxHq_API()
        __time1 = datetime.now()
        try:
            with apix.connect(ip, port):
                if apix.get_security_count(TDXParams.MARKET_SZ) > 9000:  # 0：深市 股票数量 = 9260
                    _timestamp = datetime.now() - __time1
                    self.gateway.write_log('服务器{}:{},耗时:{}'.format(ip, port, _timestamp))
                    return _timestamp
                else:
                    self.gateway.write_log(u'该服务器IP {}无响应'.format(ip))
                    return timedelta(9, 9, 0)
        except:
            self.gateway.write_error(u'tdx ping服务器，异常的响应{}'.format(ip))
            return timedelta(9, 9, 0)

    def select_best_ip(self):
        """
        选择行情服务器
        :return:
        """
        self.gateway.write_log(u'选择通达信股票行情服务器')
        data_future = [self.ping(x.get('ip'), x.get('port')) for x in self.ip_list]

        best_future_ip = self.ip_list[data_future.index(min(data_future))]

        self.gateway.write_log(u'选取 {}:{}'.format(
            best_future_ip['ip'], best_future_ip['port']))
        return best_future_ip

    def connect(self, n=3):
        """
        连接通达讯行情服务器
        :param n:
        :return:
        """
        if self.connection_status:
            for api in self.api_dict:
                if api is not None or getattr(api, "client", None) is not None:
                    self.gateway.write_log(u'当前已经连接,不需要重新连接')
                    return

        self.gateway.write_log(u'开始通达信行情服务器')

        # 选取最佳服务器
        if self.best_ip['ip'] is None and self.best_ip['port'] is None:
            self.best_ip = self.select_best_ip()

        # 创建n个api连接对象实例
        for i in range(n):
            try:
                api = TdxHq_API(heartbeat=True, auto_retry=True, raise_exception=True)
                api.connect(self.best_ip['ip'], self.best_ip['port'])
                # 尝试获取市场合约统计
                c = api.get_security_count(TDXParams.MARKET_SZ)
                if c is None or c < 10:
                    err_msg = u'该服务器IP {}/{}无响应'.format(self.best_ip['ip'], self.best_ip['port'])
                    self.gateway.write_error(err_msg)
                else:
                    self.gateway.write_log(u'创建第{}个tdx连接'.format(i + 1))
                    self.api_dict[i] = api
                    self.last_tick_dt[i] = datetime.now()
                    self.connection_status = True
                    self.security_count = c

            except Exception as ex:
                self.gateway.write_error(u'连接服务器tdx[{}]异常:{},{}'.format(i, str(ex), traceback.format_exc()))
                return

        # 创建连接池，每个连接都调用run方法
        self.pool = Pool(n)
        self.pool.map_async(self.run, range(n))

        # 设置上层的连接状态
        self.gateway.tdxConnected = True

    def reconnect(self, i):
        """
        重连
        :param i:
        :return:
        """
        try:
            self.best_ip = self.select_best_ip()
            api = TdxHq_API(heartbeat=True, auto_retry=True)
            api.connect(self.best_ip['ip'], self.best_ip['port'])
            # 尝试获取市场合约统计
            c = api.get_security_count(TDXParams.MARKET_SZ)
            if c is None or c < 10:
                err_msg = u'该服务器IP {}/{}无响应'.format(self.best_ip['ip'], self.best_ip['port'])
                self.gateway.write_error(err_msg)
            else:
                self.gateway.write_log(u'重新创建第{}个tdx连接'.format(i + 1))
                self.api_dict[i] = api

            sleep(1)
        except Exception as ex:
            self.gateway.write_error(u'重新连接服务器tdx[{}]异常:{},{}'.format(i, str(ex), traceback.format_exc()))
            return

    def close(self):
        """退出API"""
        self.connection_status = False

        # 设置上层的连接状态
        self.gateway.tdxConnected = False

        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    def subscribe(self, req):
        """订阅合约"""
        # 这里的设计是，如果尚未登录就调用了订阅方法
        # 则先保存订阅请求，登录完成后会自动订阅
        vn_symbol = str(req.symbol)
        if '.' in vn_symbol:
            vn_symbol = vn_symbol.split('.')[0]

        self.gateway.write_log(u'通达信行情订阅 {}'.format(str(vn_symbol)))

        tdx_symbol = vn_symbol  # [0:-2] + 'L9'
        tdx_symbol = tdx_symbol.upper()
        self.gateway.write_log(u'{}=>{}'.format(vn_symbol, tdx_symbol))
        self.symbol_vn_dict[tdx_symbol] = vn_symbol

        if tdx_symbol not in self.registed_symbol_set:
            self.registed_symbol_set.add(tdx_symbol)

        # 查询股票信息
        self.qry_instrument(vn_symbol)

        self.check_status()

    def check_status(self):
        # self.gateway.write_log(u'检查tdx接口状态')
        if len(self.registed_symbol_set) == 0:
            return True

        # 若还没有启动连接，就启动连接
        over_time = [((datetime.now() - dt).total_seconds() > 60) for dt in self.last_tick_dt.values()]
        if not self.connection_status or len(self.api_dict) == 0 or any(over_time):
            self.gateway.write_log(u'tdx还没有启动连接，就启动连接')
            self.close()
            self.pool = None
            self.api_dict = {}
            pool_cout = getattr(self.gateway, 'tdx_pool_count', 3)
            self.connect(pool_cout)

        # self.gateway.write_log(u'tdx接口状态正常')

    def qry_instrument(self, symbol):
        """
        查询/更新股票信息
        :return:
        """
        if not self.connection_status:
            return

        api = self.api_dict.get(0)
        if api is None:
            self.gateway.write_log(u'取不到api连接，更新合约信息失败')
            return

        # TODO： 取得股票的中文名
        market_code = get_tdx_market_code(symbol)
        api.to_df(api.get_finance_info(market_code, symbol))

        # 如果有预定的订阅合约，提前订阅
        # if len(all_contacts) > 0:
        #     cur_folder =  os.path.dirname(__file__)
        #     export_file = os.path.join(cur_folder,'contracts.csv')
        #     if not os.path.exists(export_file):
        #         df = pd.DataFrame(all_contacts)
        #         df.to_csv(export_file)

    def run(self, i):
        """
        版本1：Pool内得线程，持续运行,每个线程从queue中获取一个请求并处理
        版本2：Pool内线程，从订阅合约集合中，取出符合自己下标 mode n = 0的合约，并发送请求
        :param i:
        :return:
        """
        # 版本2：
        try:
            api_count = len(self.api_dict)
            last_dt = datetime.now()
            self.gateway.write_log(u'开始运行tdx[{}],{}'.format(i, last_dt))
            while self.connection_status:
                symbols = set()
                for idx, tdx_symbol in enumerate(list(self.registed_symbol_set)):
                    # self.gateway.write_log(u'tdx[{}], api_count:{}, idx:{}, tdx_symbol:{}'.format(i, api_count, idx, tdx_symbol))
                    if idx % api_count == i:
                        try:
                            symbols.add(tdx_symbol)
                            self.processReq(tdx_symbol, i)
                        except BrokenPipeError as bex:
                            self.gateway.write_error(u'BrokenPipeError{},重试重连tdx[{}]'.format(str(bex), i))
                            self.reconnect(i)
                            sleep(5)
                            break
                        except Exception as ex:
                            self.gateway.write_error(
                                u'tdx[{}] exception:{},{}'.format(i, str(ex), traceback.format_exc()))

                            # api = self.api_dict.get(i,None)
                            # if api is None or getattr(api,'client') is None:
                            self.gateway.write_error(u'重试重连tdx[{}]'.format(i))
                            print(u'重试重连tdx[{}]'.format(i), file=sys.stderr)
                            self.reconnect(i)

                # self.gateway.write_log(u'tdx[{}] sleep'.format(i))
                sleep(self.req_interval)
                dt = datetime.now()
                if last_dt.minute != dt.minute:
                    self.gateway.write_log('tdx[{}] check point. {}, process symbols:{}'.format(i, dt, symbols))
                    last_dt = dt
        except Exception as ex:
            self.gateway.write_error(u'tdx[{}] pool.run exception:{},{}'.format(i, str(ex), traceback.format_exc()))

        self.gateway.write_error(u'tdx[{}] {}退出'.format(i, datetime.now()))

    def processReq(self, req, i):
        """
        处理行情信息ticker请求
        :param req:
        :param i:
        :return:
        """
        symbol = req
        if '.' in symbol:
            symbol, exchange = symbol.split('.')
            if exchange == 'SZSE':
                market_code = 0
            else:
                market_code = 1
        else:
            market_code = get_tdx_market_code(symbol)
            exchange = get_stock_exchange(symbol)

        exchange = Exchange(exchange)

        api = self.api_dict.get(i, None)
        if api is None:
            self.gateway.write_log(u'tdx[{}] Api is None'.format(i))
            raise Exception(u'tdx[{}] Api is None'.format(i))

        symbol_config = self.symbol_dict.get('{}_{}'.format(symbol, market_code), {})
        decimal_point = symbol_config.get('decimal_point', 2)

        # self.gateway.write_log(u'tdx[{}] get_instrument_quote:({},{})'.format(i,self.symbol_market_dict.get(symbol),symbol))
        rt_list = api.get_security_quotes([(market_code, symbol)])
        if rt_list is None or len(rt_list) == 0:
            self.gateway.write_log(u'tdx[{}]: rt_list为空'.format(i))
            return
        # else:
        #    self.gateway.write_log(u'tdx[{}]: rt_list数据:{}'.format(i, rt_list))
        if i in self.last_tick_dt:
            self.last_tick_dt[i] = datetime.now()

        # <class 'list'>: [OrderedDict([
        # ('market', 0),
        # ('code', '000001'),
        # ('active1', 1385),
        # ('price', 13.79),
        # ('last_close', 13.69),
        # ('open', 13.65), ('high', 13.81), ('low', 13.56),
        # ('reversed_bytes0', 10449822), ('reversed_bytes1', -1379),
        # ('vol', 193996), ('cur_vol', 96),
        # ('amount', 264540864.0),
        # ('s_vol', 101450),
        # ('b_vol', 92546),
        # ('reversed_bytes2', 0), ('reversed_bytes3', 17185),
        # ('bid1', 13.79), ('ask1', 13.8), ('bid_vol1', 877), ('ask_vol1', 196),
        # ('bid2', 13.78), ('ask2', 13.81), ('bid_vol2', 2586), ('ask_vol2', 1115),
        # ('bid3', 13.77), ('ask3', 13.82), ('bid_vol3', 1562), ('ask_vol3', 807),
        # ('bid4', 13.76), ('ask4', 13.83), ('bid_vol4', 211), ('ask_vol4', 711),
        # ('bid5', 13.75), ('ask5', 13.84), ('bid_vol5', 1931), ('ask_vol5', 1084),
        # ('reversed_bytes4', (385,)), ('reversed_bytes5', 1), ('reversed_bytes6', -41), ('reversed_bytes7', -29), ('reversed_bytes8', 1), ('reversed_bytes9', 0.88),
        # ('active2', 1385)])]
        dt = datetime.now()
        for d in list(rt_list):
            # 忽略成交量为0的无效单合约tick数据
            if d.get('cur_vol', 0) <= 0:
                # self.gateway.write_log(u'忽略成交量为0的无效单合约tick数据:')
                continue

            code = d.get('code', None)
            if symbol != code and code is not None:
                self.gateway.write_log(u'忽略合约{} {} 不一致的tick数据:{}'.format(symbol, d.get('code'), rt_list))
                continue

            tick = TickData(
                gateway_name=self.gateway_name,
                symbol=symbol,
                exchange=exchange,
                datetime=dt,
                date=dt.strftime('%Y-%m-%d'),
                time=dt.strftime('%H:%M:%S')
            )

            if decimal_point > 2:
                tick.pre_close = round(d.get('last_close') / (10 ** (decimal_point - 2)), decimal_point)
                tick.high_price = round(d.get('high') / (10 ** (decimal_point - 2)), decimal_point)
                tick.open_price = round(d.get('open') / (10 ** (decimal_point - 2)), decimal_point)
                tick.low_price = round(d.get('low') / (10 ** (decimal_point - 2)), decimal_point)
                tick.last_price = round(d.get('price') / (10 ** (decimal_point - 2)), decimal_point)

                tick.bid_price_1 = round(d.get('bid1') / (10 ** (decimal_point - 2)), decimal_point)
                tick.bid_volume_1 = d.get('bid_vol1')
                tick.ask_price_1 = round(d.get('ask1') / (10 ** (decimal_point - 2)), decimal_point)
                tick.ask_volume_1 = d.get('ask_vol1')

                if d.get('bid5'):
                    tick.bid_price_2 = round(d.get('bid2') / (10 ** (decimal_point - 2)), decimal_point)
                    tick.bid_volume_2 = d.get('bid_vol2')
                    tick.askPrice2 = round(d.get('ask2') / (10 ** (decimal_point - 2)), decimal_point)
                    tick.ask_volume_2 = d.get('ask_vol2')

                    tick.bid_price_3 = round(d.get('bid3') / (10 ** (decimal_point - 2)), decimal_point)
                    tick.bid_volume_3 = d.get('bid_vol3')
                    tick.ask_price_3 = round(d.get('ask3') / (10 ** (decimal_point - 2)), decimal_point)
                    tick.ask_volume_3 = d.get('ask_vol3')

                    tick.bid_price_4 = round(d.get('bid4') / (10 ** (decimal_point - 2)), decimal_point)
                    tick.bid_volume_4 = d.get('bid_vol4')
                    tick.ask_price_4 = round(d.get('ask4') / (10 ** (decimal_point - 2)), decimal_point)
                    tick.ask_volume_4 = d.get('ask_vol4')

                    tick.bid_price_5 = round(d.get('bid5') / (10 ** (decimal_point - 2)), decimal_point)
                    tick.bid_volume_5 = d.get('bid_vol5')
                    tick.ask_price_5 = round(d.get('ask5') / (10 ** (decimal_point - 2)), decimal_point)
                    tick.ask_volume_5 = d.get('ask_vol5')

            else:
                tick.pre_close = d.get('last_close')
                tick.high_price = d.get('high')
                tick.open_price = d.get('open')
                tick.low_price = d.get('low')
                tick.last_price = d.get('price')

                tick.bid_price_1 = d.get('bid1')
                tick.bid_volume_1 = d.get('bid_vol1')
                tick.ask_price_1 = d.get('ask1')
                tick.ask_volume_1 = d.get('ask_vol1')

                if d.get('bid5'):
                    tick.bid_price_2 = d.get('bid2')
                    tick.bid_volume_2 = d.get('bid_vol2')
                    tick.ask_price_2 = d.get('ask2')
                    tick.ask_volume_2 = d.get('ask_vol2')

                    tick.bid_price_3 = d.get('bid3')
                    tick.bid_volume_3 = d.get('bid_vol3')
                    tick.ask_price_3 = d.get('ask3')
                    tick.ask_volume_3 = d.get('ask_vol3')

                    tick.bid_price_4 = d.get('bid4')
                    tick.bid_volume_4 = d.get('bid_vol4')
                    tick.ask_price_4 = d.get('ask4')
                    tick.ask_volume_4 = d.get('ask_vol4')

                    tick.bid_price_5 = d.get('bid5')
                    tick.bid_volume_5 = d.get('bid_vol5')
                    tick.ask_price_5 = d.get('ask5')
                    tick.ask_volume_5 = d.get('ask_vol5')

            tick.volume = d.get('vol', 0)
            tick.open_interest = d.get('amount', 0)

            # 修正毫秒
            last_tick = self.symbol_tick_dict.get(symbol, None)
            if (last_tick is not None) and tick.datetime.replace(microsecond=0) == last_tick.datetime:
                # 与上一个tick的时间（去除毫秒后）相同,修改为500毫秒
                tick.datetime = tick.datetime.replace(microsecond=500)
                tick.time = tick.datetime.strftime('%H:%M:%S.%f')[0:12]
            else:
                tick.datetime = tick.datetime.replace(microsecond=0)
                tick.time = tick.datetime.strftime('%H:%M:%S.%f')[0:12]

            tick.date = tick.datetime.strftime('%Y-%m-%d')
            tick.trading_day = tick.datetime.strftime('%Y-%m-%d')

            # 指数没有涨停和跌停，就用昨日收盘价正负10%
            tick.limit_up = tick.pre_close * 1.1
            tick.limit_down = tick.pre_close * 0.9

            # 排除非交易时间得tick
            if tick.datetime.hour not in [9, 10, 11, 13, 14, 15]:
                return
            elif tick.datetime.hour == 9 and tick.datetime.minute <= 25:
                return
            elif tick.datetime.hour == 15 and tick.datetime.minute >= 0:
                return

            self.symbol_tick_dict[symbol] = tick

            self.gateway.on_tick(tick)


class PbTdApi(object):

    def __init__(self, gateway: BaseGateway):
        """"""
        super().__init__()
        self._active = False
        self.gateway: BaseGateway = gateway
        self.gateway_name: str = gateway.gateway_name

        self.userid: str = ""  # 资金账号
        self.product_id: str = ""  # 产品编号（在pb客户端看到）
        self.unit_id: str = "1"  # 单元编号（在pb客户端设置)，缺省是1
        self.holder_ids = {}

        self.order_folder = ""  # 埋单csv文件所在目录
        self.account_folder = ""  # 账号导出csv所在目录

        # 缓存了当前交易日
        self.trading_day = datetime.now().strftime('%Y-%m-%d')
        self.trading_date = self.trading_day.replace('-', '')

        self.connect_status: bool = False
        self.login_status: bool = False

        # 所有交易
        self.trades = {}  # tradeid: trade

        # 未获取本地更新检查的orderid清单
        self.unchecked_orderids = []

    def close(self):
        pass

    def connect(self, user_id, order_folder, account_folder, product_id, unit_id="1", holder_ids={}):
        """连接"""
        self.userid = user_id
        self.order_folder = order_folder
        self.product_id = product_id
        self.unit_id = unit_id
        self.holder_ids = holder_ids

        if os.path.exists(self.order_folder):
            self.connect_status = True

        self.account_folder = account_folder
        if os.path.exists(self.account_folder):
            self.login_status = True

    def get_data(self, file_path, field_names=None):
        """获取文件内容"""
        if not os.path.exists(file_path):
            return None

        results = []
        with open(file=file_path, mode='r', encoding='gbk', ) as f:
            reader = csv.DictReader(f=f, fieldnames=field_names, delimiter=",")
            for row in reader:
                results.append(row)

        return results

    def query_account(self):
        """获取资金账号信息"""

        # 账号的文件
        accounts_csv = os.path.abspath(os.path.join(self.account_folder,
                                                    self.trading_date,
                                                    '{}{}.csv'.format(
                                                        PB_FILE_NAMES.get('accounts'),
                                                        self.trading_date)))
        # csv => 所有账号资金清单
        account_list = self.get_data(accounts_csv)
        if not account_list:
            return

        for data in account_list:
            if data["资金账户"] != self.userid:
                continue
            account = AccountData(
                gateway_name=self.gateway_name,
                accountid=self.userid,
                balance=float(data["产品净值"]),
                frozen=float(data["产品净值"]) - float(data["可用余额"]),
                currency="人民币",
                trading_day=self.trading_day
            )
            self.gateway.on_account(account)

    def query_position(self):
        """获取持仓信息"""

        # 持仓的文件
        positions_csv = os.path.abspath(os.path.join(self.account_folder,
                                                     self.trading_date,
                                                     '{}{}.csv'.format(
                                                         PB_FILE_NAMES.get('positions'),
                                                         self.trading_date)))
        # csv => 所有持仓清单
        position_list = self.get_data(positions_csv)
        if not position_list:
            return

        for data in position_list:
            if data["资金账户"] != self.userid:
                continue
            symbol = data["证券代码"]

            # symbol => Exchange
            exchange = symbol_exchange_map.get(symbol, None)
            if not exchange:
                exchange_str = get_stock_exchange(code=symbol)
                if len(exchange_str) > 0:
                    exchange = Exchange(exchange_str)
                    symbol_exchange_map.update({symbol: exchange})

            name = symbol_name_map.get(symbol, None)
            if not name:
                name = data["证券名称"]
                symbol_name_map.update({symbol: name})

            position = PositionData(
                gateway_name=self.gateway_name,
                accountid=self.userid,
                symbol=data["证券代码"],
                exchange=exchange,
                direction=Direction.NET,
                name=name,
                volume=int(data["持仓数量"]),
                yd_volume=int(data["可用数量"]),
                price=float(data["成本价"]),
                cur_price=float(data["最新价"]),
                pnl=float(data["浮动盈亏"]),
                holder_id=data["股东"]
            )
            self.gateway.on_position(position)

    def query_orders(self):
        """获取所有委托"""
        # 所有委托的文件
        orders_csv = os.path.abspath(os.path.join(self.account_folder,
                                                  self.trading_date,
                                                  '{}{}.csv'.format(
                                                      PB_FILE_NAMES.get('orders'),
                                                      self.trading_date)))

        # csv => 所有委托记录
        order_list = self.get_data(orders_csv)
        if not order_list:
            return

        for data in order_list:
            if data["资金账户"] != self.userid:
                continue

            sys_orderid = str(data["委托序号"])

            # 检查是否存在本地order_manager缓存中
            order = self.gateway.order_manager.get_order_with_sys_orderid(sys_orderid)
            order_date = data["委托日期"]
            order_time = data["委托时间"]
            order_status = STATUS_NAME2VT.get(data["委托状态"])
            if order is None:
                local_orderid = self.gateway.order_manager.get_local_orderid(sys_orderid)
                order_dt = datetime.strptime(f'{order_date} {order_time}', "%Y%m%d %H%M%S")
                order = OrderData(
                    gateway_name=self.gateway_name,
                    symbol=data["证券代码"],
                    exchange=EXCHANGE_NAME2VT.get(data["交易市场"]),
                    orderid=local_orderid,
                    sys_orderid=sys_orderid,
                    accountid=self.userid,
                    type=ORDERTYPE_NAME2VT.get(data["价格类型"], OrderType.LIMIT),
                    direction=DIRECTION_STOCK_NAME2VT.get(data["委托方向"]),
                    offset=Offset.NONE,
                    price=float(data["委托价格"]),
                    volume=float(data["委托数量"]),
                    traded=float(data["成交数量"]),
                    status=order_status,
                    datetime=order_dt,
                    time=order_dt.strftime('%H:%M:%S')
                )
                self.gateway.order_manager.on_order(order)
                continue
            else:
                if order.status != order_status or order.traded != float(data["成交数量"]):
                    order.traded = float(data["成交数量"])
                    order.status = order_status
                    self.gateway.order_manager.on_order(order)
                    continue

    def query_trades(self):
        """获取所有成交"""
        # 所有成交的文件
        trades_csv = os.path.abspath(os.path.join(self.account_folder,
                                                  self.trading_date,
                                                  '{}{}.csv'.format(
                                                      PB_FILE_NAMES.get('trades'),
                                                      self.trading_date)))

        # csv => 所有成交记录
        trade_list = self.get_data(trades_csv)
        if not trade_list:
            return

        for data in trade_list:
            if data["资金账户"] != self.userid:
                continue

            sys_orderid = str(data["委托序号"])
            sys_tradeid = str(data["成交序号"])

            # 检查是否存在本地trades缓存中
            trade = self.trades.get(sys_tradeid, None)

            if trade is None:
                local_orderid = self.gateway.order_manager.get_local_orderid(sys_orderid)
                trade_date = data["成交日期"]
                trade_time = data["成交时间"]
                trade_dt = datetime.strptime(f'{trade_date} {trade_time}', "%Y%m%d %H%M%S")
                trade = TradeData(
                    gateway_name=self.gateway_name,
                    symbol=data["证券代码"],
                    exchange=EXCHANGE_NAME2VT.get(data["交易市场"]),
                    orderid=local_orderid,
                    tradeid=sys_tradeid,
                    sys_orderid=sys_orderid,
                    accountid=self.userid,
                    direction=DIRECTION_STOCK_NAME2VT.get(data["委托方向"]),
                    offset=Offset.NONE,
                    price=float(data["成交价格"]),
                    volume=float(data["成交数量"]),
                    datetime=trade_dt,
                    time=trade_dt.strftime('%H:%M:%S'),
                    trade_amount=float(data["成交金额"]),
                    commission=float(data["总费用"])
                )
                self.trades[sys_tradeid] = trade
                self.gateway.on_trade(copy.copy(trade))
                continue

    def check_send_order(self):
        """检查更新委托文件"""
        # 当日send_order的文件
        send_order_csv = os.path.abspath(os.path.join(self.order_folder,
                                                      '{}{}.csv'.format(
                                                          PB_FILE_NAMES.get('send_order'),
                                                          self.trading_date)))
        # csv => 所有send_order记录
        order_list = self.get_data(send_order_csv, field_names=SEND_ORDER_FIELDS.keys())

        # 逐一处理
        for data in order_list:
            local_orderid = data.get('WBZDYXH', "").lstrip(' ')
            if local_orderid is "":
                continue

            if local_orderid not in self.unchecked_orderids:
                continue

            # 从本地order_manager中获取order
            order = self.gateway.order_manager.get_order_with_local_orderid(local_orderid)
            # 判断order取不到，或者order状态不是SUBMITTING
            if order is None or order.status != Status.SUBMITTING:
                continue

            # 检查是否具有系统委托编号
            if order.sys_orderid == "":
                sys_orderid = data.get('WTXH', '').lstrip(' ')
                if len(sys_orderid) == 0:
                    continue

                # 委托失败标志
                if sys_orderid == "0":
                    err_msg = data.get('SBYY', '').lstrip(' ')
                    err_id = data.get('WTSBDM', '').lstrip(' ')
                    order.status = Status.REJECTED
                    self.gateway.order_manager.on_order(order)
                    self.gateway.write_error(msg=err_msg, error={"ErrorID": err_id, "ErrorMsg": "委托失败"})
                else:
                    self.gateway.order_manager.update_orderid_map(local_orderid=local_orderid, sys_orderid=sys_orderid)
                    order.sys_orderid = sys_orderid
                    order.status = Status.NOTTRADED
                    self.gateway.order_manager.on_order(order)
                    self.gateway.write_log(f'委托成功')

                # 移除检查的id
                self.gateway.write_log(f'本地委托单更新检查完毕，移除{local_orderid}')
                self.unchecked_orderids.remove(local_orderid)

    def send_order(self, req: OrderRequest):
        """委托"""
        # 创建本地orderid
        local_orderid = self.gateway.order_manager.new_local_orderid()

        # req => order
        order = req.create_order_data(orderid=local_orderid, gateway_name=self.gateway_name)

        csv_file = os.path.abspath(os.path.join(self.order_folder,
                                                '{}{}.csv'.format(PB_FILE_NAMES.get('send_order'), self.trading_date)))
        # 股票买卖，强制offset = Offset.NONE
        order.offset = Offset.NONE

        data = {
            "CPBH": self.product_id,  # 产品代码/基金代码 <-- 输入参数 -->
            "ZCDYBH": self.unit_id,  # 单元编号/组合编号
            "ZHBH": self.unit_id,  # 组合编号
            "GDDM": self.holder_ids.get(order.exchange),  # 股东代码
            "JYSC": EXCHANGE_VT2PB.get(order.exchange),  # 交易市场
            "ZQDM": order.symbol,  # 证券代码
            "WTFX": DIRECTION_STOCK_VT2PB.get((order.direction, order.offset)),  # 委托方向
            "WTJGLX": get_pb_order_type(order.exchange, order.type),  # 委托价格类型
            "WTJG": round(order.price, 4),  # 委托价格
            "WTSL": int(order.volume),  # 委托数量
            "WBZDYXH": local_orderid  # 第三方系统自定义号
        }

        # 更新所有字段得长度
        order_data = format_dict(data, SEND_ORDER_FIELDS)

        append_data(file_name=csv_file,
                    dict_data=order_data,
                    field_names=SEND_ORDER_FIELDS.keys(),
                    auto_header=False,
                    encoding='gbk')

        # 设置状态为提交中
        order.status = Status.SUBMITTING
        # 添加待检查列表
        self.unchecked_orderids.append(local_orderid)
        # 登记并发送on_order事件
        self.gateway.order_manager.on_order(order)

        # 添加定时检查任务
        if self.check_send_order not in self.gateway.query_functions:
            self.gateway.write_log(f'添加定时检查到任务队列中')
            self.gateway.query_functions.append(self.check_send_order)

    def cancel_order(self, req: CancelRequest):
        """撤单"""

        # 获取订单
        order = self.gateway.order_manager.get_order_with_local_orderid(local_orderid=req.orderid)

        # 订单不存在，或者已经全部成交，已经被拒单，已经撤单
        if order is None or order.status in [Status.ALLTRADED, Status.REJECTED, Status.CANCELLING, Status.CANCELLED]:
            self.gateway.write_log(f'订单{req.orderid}不存在, 撤单失败')
            return False

        sys_orderid = self.gateway.order_manager.get_sys_orderid(req.orderid)

        if len(sys_orderid) == 0:
            self.gateway.write_log(f'订单{req.orderid}=》系统委托id不存在，撤单失败')
            return False

        data = {
            "WTXH": sys_orderid,  # 委托序号
        }
        # 更新所有字段得长度
        cancel_data = format_dict(data, CANCEL_ORDER_FIELDS)

        csv_file = os.path.abspath(os.path.join(self.order_folder,
                                                '{}{}.csv'.format(PB_FILE_NAMES.get('cancel_order'), self.trading_date)))
        append_data(file_name=csv_file,
                    dict_data=cancel_data,
                    field_names=CANCEL_ORDER_FIELDS.keys(),
                    auto_header=False,
                    encoding='gbk')
        return True
