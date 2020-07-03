from vnpy.trader.utility import load_json, save_json

# 更新使用的数据库名称
database_name = "JQDATA.db"
setting = load_json("vt_setting.json")
setting.update({"database.database": database_name})
save_json("vt_setting.json", setting)


from vnpy.trader.engine import MainEngine, EventEngine
from vnpy.app.data_manager.engine import ManagerEngine
from vnpy.trader.constant import Interval, Exchange

manager = ManagerEngine(MainEngine, EventEngine)


# 获取数据库中所有品种的Bar数据统计信息，包括字段：symbol，exchange，interval，count，start，end
bar_data_statistics = manager.get_bar_data_available()

# 从数据接口下载更新数据库中所有品种的Bar数据至最新
for d in bar_data_statistics:
    num = manager.download_bar_data_from_datasource(d["symbol"], Exchange(d["exchange"]), Interval(d["interval"]), d["end"])
    print(f'{d["symbol"]} \t新下载更新Bar数量：{num}')