# -*- coding:utf-8 -*-
import sys
sys.path.append("C:\Users\Administrator\Desktop\VNPY\My_Strategies")

from datetime import datetime
from vnpy.trader.utility import load_json, save_json

from strategies.AAA import AAA


if __name__ == "__main__":

    # 更新配置文件数据库名称
    setting = load_json("vt_setting.json")
    setting.update({"database.database":"JQDATA.db", "datasource.api": "jqdata"})
    save_json("vt_setting.json", setting)


    from vnpy.app.cta_strategy.backtesting import BacktestingEngine, OptimizationSetting

    engine = BacktestingEngine()
    engine.add_strategy(AAA, {})

    engine.set_parameters(
        vt_symbol="NI9999.SHFE",
        interval="1m",
        start=datetime(2018, 1, 1),
        end=datetime(2020, 7, 30),
        rate_type="固定手续费",
        rate=0.5,
        slippage=10,
        size=1,
        pricetick=10,
        capital=1000000
    )

    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    engine.calculate_statistics(save_statistics=True, save_csv=False, save_chart=False)
    # engine.show_chart()
