from vnpy.trader.utility import load_json, save_json

# 更新使用的数据库名称
database_name = "JQDATA.db"
setting = load_json("vt_setting.json")
setting.update({"database.database": database_name})
save_json("vt_setting.json", setting)

# 加载回测品种信息
product_info = load_json("回测品种信息.json")

import sys
sys.path.append(r"C:\Users\Administrator\Desktop\VNPY\My_Strategies")

# import warnings
# warnings.filterwarnings("ignore")

from datetime import date
from vnpy.trader.constant import Interval
from vnpy.app.cta_strategy.base import BacktestingMode
from vnpy.app.cta_strategy.backtesting import BacktestingEngine, OptimizationSetting


def run_optimization(strategy, product, setting, ga=False):
    engine = BacktestingEngine()
    engine.set_parameters(vt_symbol=product_info[product]["vt_symbol"], rate_type=product_info[product]["rate_type"],rate=product_info[product]["rate"], 
                          slippage=product_info[product]["slippage"], size=product_info[product]["size"], pricetick=product_info[product]["pricetick"], 
                          start=start, end= end, capital=capital, interval=Interval.MINUTE, mode=BacktestingMode.BAR)
    engine.add_strategy(strategy_class=strategy, setting={})

    if not ga:
        # 多进程穷举参数优化
        result = engine.run_optimization(setting)
    else:
        # 遗传算法参数优化
        result = engine.run_ga_optimization(setting)

    return result

if __name__ == "__main__":
    
    from strategies.AAA import AAA

    # 回测参数设置
    strategy= AAA                       # 参数优化策略
    product = "IF"                      # 参数优化品种
    start   = date(2020, 1,  1)         # 回测起始时间
    end     = date(2020, 6, 30)         # 回测结束时间
    capital = 500000                    # 初始资金

    # 参数优化设置
    setting = OptimizationSetting()
    setting.set_target("total_return")
    setting.add_parameter("long_period", 3, 10, 1)
    setting.add_parameter("rsi_win", 10, 20, 1)
    setting.add_parameter("k_line_win", 6, 20, 1)
    setting.add_parameter("donchian_win", 6, 20, 1)
    setting.add_parameter("trailing_long", 0.5, 1, 0.1)
    setting.add_parameter("trailing_short", 0.5, 1, 0.1)

    print("开始运行参数优化")
    result = run_optimization(strategy=strategy, product=product, setting=setting, ga=False)