import multiprocessing
import sys
from time import sleep
from datetime import datetime, time
from logging import DEBUG

import chinese_calendar as cnc

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine
from vnpy.trader.utility import load_json

from vnpy.gateway.sugar import SugarGateway
from vnpy.gateway.ctp import CtpGateway
from vnpy.app.spread_trading import SpreadTradingApp
from vnpy.app.data_recorder import DataRecorderApp
from vnpy.app.data_recorder.engine import EVENT_RECORDER_LOG


SETTINGS["log.active"] = True       # 控制是否要启动LogEngine，默认为True。如果修改为False则后续几项设置都会失效，VN Trader在运行时无日志输出或是日志文件生成（可以降低部分系统延时）。
SETTINGS["log.level"] = DEBUG       # 控制日志输出的级别，日志输出从频繁到精简可以分成DEBUG、INFO、WARNING、ERROR、CRITICAL五个级别，分别对应10、20、30、40、50的整数值。
SETTINGS["log.console"] = True      # 控制终端是否输出日志，当设置为True时，通过终端运行脚本来启动VN Trader，日志信息会输出在终端中；如果通过VN Station来直接启动VN Trader，则无console输出。


sugar_setting = load_json("connect_sugar.json")
ctp_setting = load_json("connect_ctp.json")


def check_trading_period():
    """"""
    now = datetime.now()
    current_time = now.time()

    trading = False

    if (now.weekday() < 5 and not cnc.is_in_lieu(now) and 
        (time(8, 59) <= current_time <= time(10, 15)
        or time(10, 29) <= current_time <= time(11, 30)
        or time(13, 29) <= current_time <= time(15, 0)
        or time(20, 59) <= current_time <= time(23, 0))
    ):
        trading = True

    return trading

def run_child():
    """
    Running in the child process.
    """
    SETTINGS["log.file"] = True         # 控制是否要将日志输出到文件中，建议设置为True，否则无法记录生成的日志。

    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    main_engine.add_gateway(SugarGateway)
    main_engine.add_gateway(CtpGateway)

    record_engine = main_engine.add_app(DataRecorderApp)
    spread_engine = main_engine.add_app(SpreadTradingApp)
    main_engine.write_log("行情记录及价差交易APP创建成功")

    log_engine = main_engine.get_engine("log")
    event_engine.register(EVENT_RECORDER_LOG, log_engine.process_log_event)
    main_engine.write_log("注册日志事件监听")

    main_engine.connect(sugar_setting, "SUGAR")
    main_engine.write_log("连接SUGAR接口")

    sleep(5)

    main_engine.connect(ctp_setting, "CTP")
    main_engine.write_log("连接CTP接口")

    sleep(15)

    spread_engine.start()
    main_engine.write_log("价差交易引擎启动成功")

    main_engine.write_log("开始记录行情数据……")

    while True:
        sleep(5)

        trading = check_trading_period()
        if not trading:
            print(f"{datetime.now()}\t关闭子进程")
            main_engine.close()
            sys.exit(0)


def run_parent():
    """
    Running in the parent process.
    """
    print(f"{datetime.now()}\t启动CTA策略守护父进程")

    child_process = None

    while True:
        trading = check_trading_period()

        # Start child process in trading period
        if trading and child_process is None:
            print(f"{datetime.now()}\t启动子进程")
            child_process = multiprocessing.Process(target=run_child)
            child_process.start()
            print(f"{datetime.now()}\t子进程启动成功")

        # 非记录时间则退出子进程
        if not trading and child_process is not None:
            if not child_process.is_alive():
                child_process = None
                print(f"{datetime.now()}\t子进程关闭成功")

        sleep(5)


if __name__ == "__main__":
    run_parent()

