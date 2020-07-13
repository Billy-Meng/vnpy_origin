import multiprocessing
from time import sleep
from datetime import datetime, time
from logging import DEBUG

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine
from vnpy.trader.utility import load_json, save_json

from vnpy.gateway.ctp import CtpGateway
from vnpy.app.data_recorder import DataRecorderApp
from vnpy.app.data_recorder.engine import EVENT_RECORDER_LOG


SETTINGS["log.active"] = True       # 控制是否要启动LogEngine，默认为True。如果修改为False则后续几项设置都会失效，VN Trader在运行时无日志输出或是日志文件生成（可以降低部分系统延时）。
SETTINGS["log.level"] = DEBUG       # 控制日志输出的级别，日志输出从频繁到精简可以分成DEBUG、INFO、WARNING、ERROR、CRITICAL五个级别，分别对应10、20、30、40、50的整数值。
SETTINGS["log.console"] = True      # 控制终端是否输出日志，当设置为True时，通过终端运行脚本来启动VN Trader，日志信息会输出在终端中；如果通过VN Station来直接启动VN Trader，则无console输出。


ctp_setting = load_json("connect_ctp.json")


def run_child():
    """
    Running in the child process.
    """
    SETTINGS["log.file"] = True         # 控制是否要将日志输出到文件中，建议设置为True，否则无法记录生成的日志。

    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    main_engine.add_gateway(CtpGateway)
    main_engine.add_app(DataRecorderApp)
    main_engine.write_log("行情记录引擎创建成功")

    log_engine = main_engine.get_engine("log")
    event_engine.register(EVENT_RECORDER_LOG, log_engine.process_log_event)
    main_engine.write_log("注册日志事件监听")

    main_engine.connect(ctp_setting, "CTP")
    main_engine.write_log("连接CTP接口")

    sleep(15)

    main_engine.write_log("开始记录行情数据……")

    while True:
        sleep(1)


def run_parent():
    """
    Running in the parent process.
    """
    print("启动行情记录守护父进程")

    # Chinese futures market trading period (day/night)
    DAY_START = time(8, 45)
    DAY_END = time(15, 30)

    NIGHT_START = time(20, 45)
    NIGHT_END = time(2, 45)

    child_process = None

    while True:
        current_time = datetime.now().time()
        trading = False

        # Check whether in trading period
        if (
            (current_time >= DAY_START and current_time <= DAY_END)
            or (current_time >= NIGHT_START)
            or (current_time <= NIGHT_END)
        ):
            trading = True

        # Start child process in trading period
        if trading and child_process is None:
            print("启动子进程")
            child_process = multiprocessing.Process(target=run_child)
            child_process.start()
            print("子进程启动成功")

        # 非记录时间则退出子进程
        if not trading and child_process is not None:
            print("关闭子进程")
            child_process.terminate()
            child_process.join()
            child_process = None
            print("子进程关闭成功")

        # 数据清洗


        sleep(5)


def add_tick_recording(vt_symbol: str):
    if vt_symbol in tick_recordings:
        print(f"已在Tick记录列表中：{vt_symbol}")
        return

    else:
        symbol, exchange = vt_symbol.split(".")
        tick_recordings[vt_symbol] = {
            "symbol": symbol,
            "exchange": exchange,
            "gateway_name": gateway_name
        }

        save_json("data_recorder_setting.json", {"tick":tick_recordings, "bar":bar_recordings})
        print(f"添加Tick记录成功：{vt_symbol}")

def add_bar_recording(vt_symbol: str):
    if vt_symbol in bar_recordings:
        print(f"已在K线记录列表中：{vt_symbol}")
        return

    else:
        symbol, exchange = vt_symbol.split(".")
        bar_recordings[vt_symbol] = {
            "symbol": symbol,
            "exchange": exchange,
            "gateway_name": gateway_name
        }

        save_json("data_recorder_setting.json", {"tick":tick_recordings, "bar":bar_recordings})
        print(f"添加K线记录成功：{vt_symbol}")

def remove_tick_recording(vt_symbol: str):
    if vt_symbol not in tick_recordings:
        print(f"不在K线记录列表中：{vt_symbol}")
        return

    tick_recordings.pop(vt_symbol)
    save_json("data_recorder_setting.json", {"tick":tick_recordings, "bar":bar_recordings})
    print(f"移除Tick记录成功：{vt_symbol}")

def remove_bar_recording(vt_symbol: str):
    if vt_symbol not in bar_recordings:
        print(f"不在K线记录列表中：{vt_symbol}")
        return

    bar_recordings.pop(vt_symbol)
    save_json("data_recorder_setting.json", {"tick":tick_recordings, "bar":bar_recordings})
    print(f"移除K线记录成功：{vt_symbol}")

def remove_all():
    tick_recordings = {}
    bar_recordings = {}
    save_json("data_recorder_setting.json", {"tick":tick_recordings, "bar":bar_recordings})
    print(f"清除Tick和K线行情记录！")

if __name__ == "__main__":
    # 启动行情记录进程
    run_parent()

    # 查看当前Tick和Bar记录列表
    # data_recorder_setting = load_json("data_recorder_setting.json")
    # tick_recordings = data_recorder_setting.get("tick", {})
    # bar_recordings = data_recorder_setting.get("bar", {})
    # print(f"Tick记录列表：{[i for i in tick_recordings.keys()]}")
    # print(f"Bar记录列表：{[i for i in bar_recordings.keys()]}")

    # 添加新的Tick和Bar记录需求，格式为 vt_symbol
    # New_List = ["IF2006.CFFEX", "IH2006.CFFEX", "IC2006.CFFEX", 
    # "cu2007.SHFE", "al2007.SHFE", "zn2007.SHFE", "au2012.SHFE", "ag2012.SHFE", "rb2010.SHFE", "ru2009.SHFE", "ni2008.SHFE", 
    # "fu2009.SHFE", "sc2007.INE", "p2009.DCE", "c2009.DCE", "i2009.DCE", "a2009.DCE", "b2009.DCE", "j2009.DCE", "jm2009.DCE", "pp2009.DCE", "v2009.DCE", 
    # "AP010.CZCE", "SR009.CZCE", "TA010.CZCE"]
    # gateway_name = "CTP"
    # for vt_symbol in New_List:
    #     add_tick_recording(vt_symbol)
    #     add_bar_recording(vt_symbol)


    # 删除Tick和Bar记录需求，格式为 vt_symbol
    # Remove_List = ["ag2012.SHFE", "rb2010.SHFE", "IF2006.CFFEX"]
    # for vt_symbol in Remove_List:
    #     remove_tick_recording(vt_symbol)
    #     remove_tick_recording(vt_symbol)

    # 清除Tick和K线行情记录
    # remove_all()

