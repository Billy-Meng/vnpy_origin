import multiprocessing
import sys
from time import sleep
from datetime import date, datetime, time
from logging import INFO

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine
from vnpy.trader.utility import load_json, send_dingding, send_weixin

from vnpy.gateway.ctp import CtpGateway
from vnpy.app.cta_strategy import CtaStrategyApp
from vnpy.app.cta_strategy.base import EVENT_CTA_LOG


SETTINGS["log.active"] = True       # 控制是否要启动LogEngine，默认为True。如果修改为False则后续几项设置都会失效，VN Trader在运行时无日志输出或是日志文件生成（可以降低部分系统延时）。
SETTINGS["log.level"] = INFO        # 控制日志输出的级别，日志输出从频繁到精简可以分成DEBUG、INFO、WARNING、ERROR、CRITICAL五个级别，分别对应10、20、30、40、50的整数值。
SETTINGS["log.console"] = True      # 控制终端是否输出日志，当设置为True时，通过终端运行脚本来启动VN Trader，日志信息会输出在终端中；如果通过VN Station来直接启动VN Trader，则无console输出。

ctp_setting = load_json("connect_ctp.json")


# Chinese futures market trading period (day/night)
DAY_START = time(8, 45)
DAY_END = time(15, 30)

NIGHT_START = time(20, 45)
NIGHT_END = time(2, 45)


def check_trading_period():
    """"""
    current_time = datetime.now().time()

    trading = False
    if (
        (current_time >= DAY_START and current_time <= DAY_END)
        or (current_time >= NIGHT_START)
        or (current_time <= NIGHT_END)
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
    main_engine.add_gateway(CtpGateway)
    cta_engine = main_engine.add_app(CtaStrategyApp)
    main_engine.write_log("主引擎创建成功")

    log_engine = main_engine.get_engine("log")
    event_engine.register(EVENT_CTA_LOG, log_engine.process_log_event)
    main_engine.write_log("注册日志事件监听")

    main_engine.connect(ctp_setting, "CTP")
    main_engine.write_log("连接CTP接口")

    sleep(30)

    cta_engine.init_engine()
    main_engine.write_log("CTA策略初始化完成")

    cta_engine.init_all_strategies()

    sleep(60)   # Leave enough time to complete strategy initialization

    main_engine.write_log("CTA策略全部初始化")

    cta_engine.start_all_strategies()
    main_engine.write_log("CTA策略全部启动")

    send_weixin("CTA策略全部启动")
    # send_dingding("CTA策略全部启动")

    while True:

        # main_engine.save_commission_margin_ratio()          # 每5分钟保存一次手续费率、保证金率数据

        trading = check_trading_period()
        if not trading:
            print("关闭子进程")
            main_engine.close()
            sys.exit(0)


        sleep(300)


def run_parent():
    """
    Running in the parent process.
    """
    print("启动CTA策略守护父进程")

    child_process = None

    while True:
        trading = check_trading_period()

        # Start child process in trading period
        if trading and child_process is None:
            print("启动子进程")
            child_process = multiprocessing.Process(target=run_child)
            child_process.start()
            print("子进程启动成功")

        # 非记录时间则退出子进程
        if not trading and child_process is not None:
            if not child_process.is_alive():
                child_process = None
                print("子进程关闭成功")

        sleep(5)


if __name__ == "__main__":
    run_parent()
