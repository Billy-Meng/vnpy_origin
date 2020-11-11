# -*- coding:utf-8 -*-
import sys
sys.path.append(r"C:\Users\Administrator\Desktop\VNPY\My_Strategies")

from datetime import datetime
from vnpy.trader.utility import load_json, save_json, remain_alpha
from vnpy.app.cta_strategy.base import BacktestingMode
from vnpy.trader.constant import Interval

from strategies.Strategy01_IF_8M_Live import Strategy01_IF_8M_Live


if __name__ == "__main__":

    # 更新配置文件数据库名称
    setting = load_json("vt_setting.json")
    setting.update({"database.database":"JQDATA.db", "datasource.api": "jqdata"})
    save_json("vt_setting.json", setting)

    # 加载回测品种信息
    product_info = load_json("回测品种信息.json")

    # 回测策略
    cta_strategy = Strategy01_IF_8M_Live

    # 回测品种参数设置
    symbol    = "IF9999"
    interval  = Interval.MINUTE
    start     = datetime(2018, 1, 1, 0, 0, 0)
    end       = datetime(2030, 1, 1, 0, 0, 0)
    capital   = 500000                                                          # 初始资金
    mode      = BacktestingMode.BAR                                             # BacktestingMode.TICK为TICK回测模式，BacktestingMode.BAR为BAR回测模式

    vt_symbol = symbol + "." + product_info[remain_alpha(symbol)]["exchange"]   # 品种代码
    rate_type = product_info[remain_alpha(symbol)]["rate_type"]                 # 固定手续费类型："浮动手续费"、"固定手续费"
    rate      = product_info[remain_alpha(symbol)]["rate"]                      # 【单边】每手固定手续费，或手续费比率   0.25/10000
    slippage  = product_info[remain_alpha(symbol)]["slippage"]                  # 【单边】滑点数
    size      = product_info[remain_alpha(symbol)]["size"]                      # 合约乘数
    pricetick = product_info[remain_alpha(symbol)]["pricetick"]                 # 最小变动价位

    print(f"手续费类型：{rate_type} \t 手续费/率：{rate} \t 合约乘数：{size} \t 最小变动价位：{pricetick}")


    # 执行回测
    from vnpy.app.cta_strategy.backtesting import BacktestingEngine, OptimizationSetting

    engine = BacktestingEngine()
    engine.set_parameters(vt_symbol= vt_symbol, interval=interval, start=start, end= end, rate_type=rate_type,
                          rate=rate, slippage=slippage, size=size, pricetick=pricetick, capital=capital, mode=mode)
    engine.add_strategy(cta_strategy, {})
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    engine.calculate_statistics(save_statistics=True, save_csv=False, save_chart=False)     #  True  False
    # engine.show_chart()
    # engine.save_tab_chart()
