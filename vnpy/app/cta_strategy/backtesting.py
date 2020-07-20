# -*- coding:utf-8 -*-
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Callable, Union
from itertools import product
from functools import lru_cache
from time import perf_counter
from pathlib import Path
import multiprocessing
import random
import traceback
import os
import pickle

import numpy as np
from pandas import DataFrame, merge
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import empyrical
import talib
from deap import creator, base, tools, algorithms

from vnpy.trader.constant import (Direction, Offset, Exchange,
                                  Interval, Status, RateType)
from vnpy.trader.database import database_manager
from vnpy.trader.object import OrderData, TradeData, BarData, TickData
from vnpy.trader.utility import round_to, BarGenerator
from vnpy.chart.my_pyecharts import MyPyecharts, Tab, Line, Bar, Grid, EffectScatter, opts, JsCode

from .base import (
    BacktestingMode,
    EngineType,
    STOPORDER_PREFIX,
    StopOrder,
    StopOrderStatus,
    INTERVAL_DELTA_MAP
)
from .template import CtaTemplate

# 忽略警告信息
import warnings
warnings.filterwarnings("ignore")

# Set deap algo
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


class OptimizationSetting:
    """
    Setting for runnning optimization.
    """

    def __init__(self):
        """"""
        self.params = {}
        self.target_name = ""

    def add_parameter(
        self, name: str, start: float, end: float = None, step: float = None
    ):
        """"""
        if not end and not step:
            self.params[name] = [start]
            return

        if start >= end:
            print("参数优化起始点必须小于终止点")
            return

        if step <= 0:
            print("参数优化步进必须大于0")
            return

        value = start
        value_list = []

        while value <= end:
            value_list.append(value)
            value += step

        self.params[name] = value_list

    def set_target(self, target_name: str):
        """"""
        self.target_name = target_name

    def generate_setting(self):
        """"""
        keys = self.params.keys()
        values = self.params.values()
        products = list(product(*values))

        settings = []
        for p in products:
            setting = dict(zip(keys, p))
            settings.append(setting)

        return settings

    def generate_setting_ga(self):
        """"""
        settings_ga = []
        settings = self.generate_setting()
        for d in settings:
            param = [tuple(i) for i in d.items()]
            settings_ga.append(param)
        return settings_ga


class BacktestingEngine:
    """"""

    engine_type = EngineType.BACKTESTING
    gateway_name = "BACKTESTING"

    def __init__(self):
        """"""
        self.vt_symbol = ""
        self.symbol = ""
        self.exchange = None
        self.start = None
        self.end = None
        self.rate_type = RateType.FIXED
        self.rate = 0
        self.slippage = 0
        self.size = 1
        self.pricetick = 0
        self.capital = 1_000_000
        self.mode = BacktestingMode.BAR
        self.inverse = False

        self.strategy_class = None
        self.strategy = None
        self.tick: TickData
        self.bar: BarData
        self.datetime = None

        self.interval = None
        self.days = 0
        self.frequency = None
        self.callback = None
        self.history_data = []

        self.stop_order_count = 0
        self.stop_orders = {}
        self.active_stop_orders = {}

        self.limit_order_count = 0
        self.limit_orders = {}
        self.active_limit_orders = {}

        self.trade_count = 0
        self.trades = {}

        self.logs = []

        self.daily_results = {}

        self.daily_df = None
        self.bar_data_df = None
        self.trade_data_df = None
        self.trade_result_df = None
        self.trade_daily_df = None
        self.statistics = None
        self.optimization_df = None

    def clear_data(self):
        """
        Clear all data of last backtesting.
        """
        self.strategy = None
        self.tick = None
        self.bar = None
        self.datetime = None

        self.stop_order_count = 0
        self.stop_orders.clear()
        self.active_stop_orders.clear()

        self.limit_order_count = 0
        self.limit_orders.clear()
        self.active_limit_orders.clear()

        self.trade_count = 0
        self.trades.clear()

        self.logs.clear()
        self.daily_results.clear()

        self.daily_df = None
        self.bar_data_df = None
        self.trade_data_df = None
        self.trade_result_df = None
        self.trade_daily_df = None
        self.statistics = None
        self.optimization_df = None

    def set_parameters(
        self,
        vt_symbol: str,
        interval: Interval,
        start: datetime,
        rate_type: RateType,
        rate: float,
        slippage: float,
        size: float,
        pricetick: float,
        capital: int = 0,
        end: datetime = None,
        mode: BacktestingMode = BacktestingMode.BAR,
        inverse: bool = False
    ):
        """"""
        self.mode = mode
        self.vt_symbol = vt_symbol
        self.interval = Interval(interval)
        self.rate_type = RateType(rate_type)
        self.rate = rate
        self.slippage = slippage
        self.size = size
        self.pricetick = pricetick
        self.start = start

        self.symbol, exchange_str = self.vt_symbol.split(".")
        self.exchange = Exchange(exchange_str)

        self.capital = capital
        self.end = end
        self.mode = mode
        self.inverse = inverse

    def add_strategy(self, strategy_class: type, setting: dict):
        """"""
        self.strategy_class = strategy_class
        self.strategy = strategy_class(
            self, strategy_class.__name__, self.vt_symbol, setting
        )

    def load_data(self):
        """"""
        self.output("开始加载历史数据")

        if not self.end:
            self.end = datetime.now()

        if self.start >= self.end:
            self.output("起始日期必须小于结束日期")
            return

        self.history_data.clear()       # Clear previously loaded history data

        # # Load 30 days of data each time and allow for progress update
        # progress_delta = timedelta(days=30)
        # total_delta = self.end - self.start
        # interval_delta = INTERVAL_DELTA_MAP[self.interval]

        # start = self.start
        # end = self.start + progress_delta
        # progress = 0

        # while start < self.end:
        #     end = min(end, self.end)  # Make sure end time stays within set range

        #     if self.mode == BacktestingMode.BAR:
        #         data = load_bar_data(
        #             self.symbol,
        #             self.exchange,
        #             self.interval,
        #             start,
        #             end
        #         )
        #     else:
        #         data = load_tick_data(
        #             self.symbol,
        #             self.exchange,
        #             start,
        #             end
        #         )

        #     self.history_data.extend(data)

        #     progress += progress_delta / total_delta
        #     progress = min(progress, 1)
        #     progress_bar = "#" * int(progress * 10)
        #     self.output(f"加载进度：{progress_bar} [{progress:.0%}]")

        #     start = end + interval_delta
        #     end += (progress_delta + interval_delta)


        # 简化历史数据加载流程
        if self.mode == BacktestingMode.BAR:
            data = load_bar_data(self.symbol, self.exchange, self.interval, self.start, self.end)
        else:
            data = load_tick_data(self.symbol, self.exchange, self.start, self.end)

        self.history_data = data


        self.output(f"历史数据加载完成，数据量：{len(self.history_data)}")        

        self.get_bar_data_df()          # 获取加载的Bar历史数据，生成 DataFrame，并赋值给 self.bar_data_df

    def run_backtesting(self):
        """"""
        if self.mode == BacktestingMode.BAR:
            func = self.new_bar
        else:
            func = self.new_tick

        self.strategy.on_init()

        # Use the first [days] of history data for initializing strategy
        day_count = 1
        ix = 0

        for ix, data in enumerate(self.history_data):
            if self.datetime and data.datetime.day != self.datetime.day:
                day_count += 1
                if day_count >= self.days:
                    break

            self.datetime = data.datetime

            try:
                self.callback(data)
            except Exception:
                self.output("触发异常，回测终止")
                self.output(traceback.format_exc())
                return

        self.strategy.inited = True
        self.output("策略初始化完成")

        self.strategy.on_start()
        self.strategy.trading = True
        self.output("开始回放历史数据")

        # Use the rest of history data for running backtesting
        for data in self.history_data[ix:]:
            try:
                func(data)
            except Exception:
                self.output("触发异常，回测终止")
                self.output(traceback.format_exc())
                return

        self.output("历史数据回放结束")

    def calculate_result(self):
        """"""
        self.output("开始计算逐日盯市盈亏")

        if not self.trades:
            self.output("成交记录为空，无法计算")
            return

        # Add trade data into daily reuslt.
        for trade in self.trades.values():
            d = trade.datetime.date()
            daily_result = self.daily_results[d]
            daily_result.add_trade(trade)

        # Calculate daily result by iteration.
        pre_close = 0
        start_pos = 0

        for daily_result in self.daily_results.values():
            daily_result.calculate_pnl(
                pre_close,
                start_pos,
                self.size,
                self.rate_type,
                self.rate,
                self.slippage,
                self.inverse
            )

            pre_close = daily_result.close_price
            start_pos = daily_result.end_pos

        # Generate dataframe
        results = defaultdict(list)

        for daily_result in self.daily_results.values():
            for key, value in daily_result.__dict__.items():
                results[key].append(value)

        self.daily_df = DataFrame.from_dict(results).set_index("date")

        self.get_trade_data_df()             # 提取成交记录，生成 DataFrame，并赋值给 self.trade_data_df
        self.calculate_trade_result()        # 计算每笔交易盈亏，生成 DataFrame，并赋值给 self.trade_result_df

        self.output("逐日盯市盈亏计算完成")
        return self.daily_df

    def show_chart(self, df: DataFrame = None):
        """"""
        # Check DataFrame input exterior
        if df is None:
            df = self.daily_df

        # Check for init DataFrame
        if df is None:
            return

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["净值曲线", "净值回撤率", "每日净盈亏", "净盈亏分布"],
            vertical_spacing=0.06
        )

        balance_line = go.Scatter(
            x=df.index,
            y=df["balance"],
            mode="lines",
            name="净值曲线"
        )
        ddpercent_scatter = go.Scatter(
            x=df.index,
            y=df["ddpercent"],
            fillcolor="red",
            fill='tozeroy',
            mode="lines",
            name="净值回撤率"
        )
        pnl_bar = go.Bar(y=df["net_pnl"], name="每日净盈亏")
        pnl_histogram = go.Histogram(x=df["net_pnl"], nbinsx=100, name="净盈亏分布")

        fig.add_trace(balance_line, row=1, col=1)
        fig.add_trace(ddpercent_scatter, row=2, col=1)
        fig.add_trace(pnl_bar, row=3, col=1)
        fig.add_trace(pnl_histogram, row=4, col=1)

        fig.update_layout(height=1000, width=1000)
        fig.show()

    def run_optimization(self, optimization_setting: OptimizationSetting, save_csv=True, output=False):
        """"""
        # Get optimization setting and target
        settings = optimization_setting.generate_setting()
        target_name = optimization_setting.target_name

        if not settings:
            self.output("优化参数组合为空，请检查")
            return

        if not target_name:
            self.output("优化目标未设置，请检查")
            return

        self.output(f"参数优化空间：{len(settings)}")

        # Use multiprocessing pool for running backtesting with different setting
        # Force to use spawn method to create new process (instead of fork on Linux)
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(multiprocessing.cpu_count())

        # 生成桌面文件路径
        home_path = Path.home()
        temp_path = home_path.joinpath("Desktop")
        filename  = f"多进程穷举参数优化结果(目标：{target_name}) {self.symbol} {self.strategy_class.__name__} [{self.start.strftime('%Y%m%d')}~{self.end.strftime('%Y%m%d')}].txt"
        filepath  = temp_path.joinpath(filename)
        
        results = []
        count = 0
        result_dict = defaultdict(list)

        start = perf_counter()

        for setting in settings:
            count += 1

            print(f'{"=" * 200}')
            print(f'{datetime.now()}\t正在进行第【 {count} 】组参数回测\t优化参数：{setting}')

            result = (pool.apply_async(optimize, (
                target_name,
                self.strategy_class,
                setting,
                self.vt_symbol,
                self.interval,
                self.start,
                self.rate_type,
                self.rate,
                self.slippage,
                self.size,
                self.pricetick,
                self.capital,
                self.end,
                self.mode,
                self.inverse
            )))
            results.append(result)

            value = result.get()

            # 在字典中添加参数列表
            value_dict = eval(value[0])
            for k, v in value_dict.items():
                result_dict[k].append(v)

            # 在字典中添加结果列表
            target_name_list = [
                "total_return", "annual_return", "cagr", "annual_volatility", "sharpe_ratio", "omega_ratio", "calmar_ratio", "sortino_ratio", "R_squared", "max_drawdown", "max_ddpercent", "average_drawdown", 
                "lw_drawdown", "max_drawdown_duration", "return_drawdown_ratio", "return_std", "rate_of_win", "profit_loss_ratio", "daily_net_pnl", "average_net_pnl", "daily_trade_count", "trade_duration"
            ]

            for target_name in target_name_list:
                result_dict[target_name].append(value[2][target_name])

            # 在字典中保存所有统计指标结果
            result_dict["statistics"].append(value[2])

            info_1 = (f'''\t总收益率：{value[2]["total_return"]:<10}\t年化收益率：{value[2]["annual_return"]:<10}\t年复合增长率：{value[2]["cagr"]:<10}\t夏普比率：{value[2]["sharpe_ratio"]:<16} \tCalmar比率：{value[2]["calmar_ratio"]:<10} \tOmega比率：{value[2]["omega_ratio"]:<10} \t索提诺比率：{value[2]["sortino_ratio"]}''')
            info_2 = (f'''\t%最大回撤：{value[2]["max_ddpercent"]:<10}\t%平均回撤：{value[2]["average_drawdown"]:<10}\t%线性加权回撤：{value[2]["lw_drawdown"]:<8}\t最大回撤金额：{value[2]["max_drawdown"]:<12}\t收益回撤比：{value[2]["return_drawdown_ratio"]:<10}\t最大回撤天数：{value[2]["max_drawdown_duration"]:<10}\t最大回撤区间：{value[2]["max_drawdown_range"]}''')
            info_3 = (f'''\t日均净盈亏：{value[2]["daily_net_pnl"]:<10}\t笔均净盈亏：{value[2]["average_net_pnl"]:<10}\t日均交易笔数：{value[2]["daily_trade_count"]:<10}\t单日最多交易笔数：{value[2]["daily_trade_max"]:<12}\t平均持仓小时：{value[2]["trade_duration"]:<10}\t%胜率：{value[2]["rate_of_win"]:<10} \t盈亏比：{value[2]["profit_loss_ratio"]}''')

            print(f"{info_1}\n{info_2}\n{info_3}")

            # 将参数优化每一次运行的临时结果提前保存，以防系统崩溃
            with open(filepath, "a+") as f:
                f.write(f"{count}\t{value[0]}\t{info_1}\t{info_2}\t{info_3}\n".replace(" ",""))

        pool.close()
        pool.join()

        # 生成多进程优化结果 DataFrame
        self.optimization_df = DataFrame(result_dict).sort_values("total_return", ascending=False).reset_index(drop=True)

        # 保存参数优化结果至桌面CSV
        if save_csv:
            csv_filepath = temp_path.joinpath(filename.replace("txt", "csv"))
            self.optimization_df.to_csv(csv_filepath)

        # Sort results and output
        result_values = [result.get() for result in results]
        result_values.sort(reverse=True, key=lambda result: result[1])

        end = perf_counter()
        cost = round((end - start)/3600, 2)

        self.output(f"多进程穷举算法参数优化完成，耗时 {cost} 小时")

        # 使用GUI界面时输出回测结果给控件
        if not output:
            return result_values

    def run_ga_optimization(self, optimization_setting: OptimizationSetting, population_size=100, ngen_size=30, output=True):
        """"""
        # Get optimization setting and target
        settings = optimization_setting.generate_setting_ga()
        target_name = optimization_setting.target_name

        if not settings:
            self.output("优化参数组合为空，请检查")
            return

        if not target_name:
            self.output("优化目标未设置，请检查")
            return

        # Define parameter generation function
        def generate_parameter():
            """"""
            return random.choice(settings)

        def mutate_individual(individual, indpb):
            """"""
            size = len(individual)
            paramlist = generate_parameter()
            for i in range(size):
                if random.random() < indpb:
                    individual[i] = paramlist[i]
            return individual,

        # Create ga object function
        global ga_target_name
        global ga_strategy_class
        global ga_setting
        global ga_vt_symbol
        global ga_interval
        global ga_start
        global ga_rate_type
        global ga_rate
        global ga_slippage
        global ga_size
        global ga_pricetick
        global ga_capital
        global ga_end
        global ga_mode
        global ga_inverse

        ga_target_name = target_name
        ga_strategy_class = self.strategy_class
        ga_setting = settings[0]
        ga_vt_symbol = self.vt_symbol
        ga_interval = self.interval
        ga_start = self.start
        ga_rate_type = self.rate_type
        ga_rate = self.rate
        ga_slippage = self.slippage
        ga_size = self.size
        ga_pricetick = self.pricetick
        ga_capital = self.capital
        ga_end = self.end
        ga_mode = self.mode
        ga_inverse = self.inverse

        # Set up genetic algorithem
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, generate_parameter)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutate_individual, indpb=1)
        toolbox.register("evaluate", ga_optimize)
        toolbox.register("select", tools.selNSGA2)

        total_size = len(settings)
        pop_size = population_size                      # number of individuals in each generation
        lambda_ = pop_size                              # number of children to produce at each generation
        mu = int(pop_size * 0.8)                        # number of individuals to select for the next generation

        cxpb = 0.95         # probability that an offspring is produced by crossover
        mutpb = 1 - cxpb    # probability that an offspring is produced by mutation
        ngen = ngen_size    # number of generation

        pop = toolbox.population(pop_size)
        hof = tools.ParetoFront()               # end result of pareto front

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        np.set_printoptions(suppress=True)
        stats.register("mean", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        # Multiprocessing is not supported yet.
        # pool = multiprocessing.Pool(multiprocessing.cpu_count())
        # toolbox.register("map", pool.map)

        # Run ga optimization
        self.output(f"参数优化空间：{total_size}")
        self.output(f"每代族群总数：{pop_size}")
        self.output(f"优良筛选个数：{mu}")
        self.output(f"迭代次数：{ngen}")
        self.output(f"交叉概率：{cxpb:.0%}")
        self.output(f"突变概率：{mutpb:.0%}")

        start = perf_counter()

        algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            mu,
            lambda_,
            cxpb,
            mutpb,
            ngen,
            stats,
            halloffame=hof
        )

        end = perf_counter()
        cost = round((end - start)/3600, 2)

        self.output(f"遗传算法参数优化完成，耗时 {cost} 小时")

        # Return result list
        results = []

        for parameter_values in hof:
            setting = dict(parameter_values)
            target_value = ga_optimize(parameter_values)[0]
            results.append((setting, target_value, {}))

        # 生成桌面文件路径
        home_path = Path.home()
        temp_path = home_path.joinpath("Desktop")
        filename  = f"遗传算法参数优化结果(目标：{target_name}) {self.symbol} {self.strategy_class.__name__} [{self.start.strftime('%Y%m%d')}~{self.end.strftime('%Y%m%d')}].txt"
        filepath  = temp_path.joinpath(filename)

        # 保存参数优化结果至桌面文件，并输出日志
        for value in results:
            info_1 = (f'''\t总收益率：{value[2]["total_return"]:<10}\t年化收益率：{value[2]["annual_return"]:<10}\t年复合增长率：{value[2]["cagr"]:<10}\t夏普比率：{value[2]["sharpe_ratio"]:<16} \tCalmar比率：{value[2]["calmar_ratio"]:<10} \tOmega比率：{value[2]["omega_ratio"]:<10} \t索提诺比率：{value[2]["sortino_ratio"]}''')
            info_2 = (f'''\t%最大回撤：{value[2]["max_ddpercent"]:<10}\t%平均回撤：{value[2]["average_drawdown"]:<10}\t%线性加权回撤：{value[2]["lw_drawdown"]:<8}\t最大回撤金额：{value[2]["max_drawdown"]:<12}\t收益回撤比：{value[2]["return_drawdown_ratio"]:<10}\t最大回撤天数：{value[2]["max_drawdown_duration"]:<10}\t最大回撤区间：{value[2]["max_drawdown_range"]}''')
            info_3 = (f'''\t日均净盈亏：{value[2]["daily_net_pnl"]:<10}\t笔均净盈亏：{value[2]["average_net_pnl"]:<10}\t日均交易笔数：{value[2]["daily_trade_count"]:<10}\t单日最多交易笔数：{value[2]["daily_trade_max"]:<12}\t平均持仓小时：{value[2]["trade_duration"]:<10}\t%胜率：{value[2]["rate_of_win"]:<10} \t盈亏比：{value[2]["profit_loss_ratio"]}''')

            print(f"{info_1}\n{info_2}\n{info_3}")

            with open(filepath, "a+") as f:
                f.write(f"{value[0]}\t{info_1}\t{info_2}\t{info_3}\n".replace(" ",""))

        return results

    def update_daily_close(self, price: float):
        """"""
        d = self.datetime.date()

        daily_result = self.daily_results.get(d, None)
        if daily_result:
            daily_result.close_price = price
        else:
            self.daily_results[d] = DailyResult(d, price)

    def new_bar(self, bar: BarData):
        """"""
        self.bar = bar
        self.datetime = bar.datetime

        self.cross_limit_order()
        self.cross_stop_order()

        try:
            if self.strategy.x_second < 60:
                self.strategy.on_second_bar(bar)
            elif self.strategy.x_second == 60:
                self.strategy.on_bar(bar)
        except AttributeError:
            self.strategy.on_bar(bar)

        self.update_daily_close(bar.close_price)

    def new_tick(self, tick: TickData):
        """"""
        self.tick = tick
        self.datetime = tick.datetime

        self.cross_limit_order()
        self.cross_stop_order()
        self.strategy.on_tick(tick)

        self.update_daily_close(tick.last_price)

    def cross_limit_order(self):
        """
        Cross limit order with last bar/tick data.
        """
        if self.mode == BacktestingMode.BAR:
            long_cross_price = self.bar.low_price
            short_cross_price = self.bar.high_price
            long_best_price = self.bar.open_price
            short_best_price = self.bar.open_price
        else:
            long_cross_price = self.tick.ask_price_1
            short_cross_price = self.tick.bid_price_1
            long_best_price = long_cross_price
            short_best_price = short_cross_price

        for order in list(self.active_limit_orders.values()):
            # Push order update with status "not traded" (pending).
            if order.status == Status.SUBMITTING:
                order.status = Status.NOTTRADED
                self.strategy.on_order(order)

            # Check whether limit orders can be filled.
            long_cross = (
                order.direction == Direction.LONG
                and order.price >= long_cross_price
                and long_cross_price > 0
            )

            short_cross = (
                order.direction == Direction.SHORT
                and order.price <= short_cross_price
                and short_cross_price > 0
            )

            if not long_cross and not short_cross:
                continue

            if order.offset != Offset.OPEN and self.strategy.trade_net_volume  < order.volume:
                self.active_limit_orders.pop(order.vt_orderid)
                print("！" * 100)
                print(f"cross_limit_order报错！平仓委托交易数量超过可平仓净头寸！当前净持仓量：{self.strategy.trade_net_volume}，问题委托信息：{order}")
                continue

            # Push order udpate with status "all traded" (filled).
            order.traded = order.volume
            order.status = Status.ALLTRADED
            self.strategy.on_order(order)

            self.active_limit_orders.pop(order.vt_orderid)

            # Push trade update
            self.trade_count += 1

            if long_cross:
                trade_price = min(order.price, long_best_price)
                pos_change = order.volume
            else:
                trade_price = max(order.price, short_best_price)
                pos_change = -order.volume

            trade = TradeData(
                symbol=order.symbol,
                exchange=order.exchange,
                orderid=order.orderid,
                tradeid=str(self.trade_count),
                direction=order.direction,
                offset=order.offset,
                price=trade_price,
                volume=order.volume,
                datetime=self.datetime,
                gateway_name=self.gateway_name,
            )

            self.trades[trade.vt_tradeid] = trade
                        
            self.strategy.pos += pos_change
            self.strategy.on_trade(trade)


    def cross_stop_order(self):
        """
        Cross stop order with last bar/tick data.
        """
        if self.mode == BacktestingMode.BAR:
            long_cross_price = self.bar.high_price
            short_cross_price = self.bar.low_price
            long_best_price = self.bar.open_price
            short_best_price = self.bar.open_price
        else:
            long_cross_price = self.tick.last_price
            short_cross_price = self.tick.last_price
            long_best_price = long_cross_price
            short_best_price = short_cross_price

        for stop_order in list(self.active_stop_orders.values()):
            # Check whether stop order can be triggered.
            long_cross = (
                stop_order.direction == Direction.LONG
                and stop_order.price <= long_cross_price
            )

            short_cross = (
                stop_order.direction == Direction.SHORT
                and stop_order.price >= short_cross_price
            )

            if not long_cross and not short_cross:
                continue

            # Create order data.
            self.limit_order_count += 1

            order = OrderData(
                symbol=self.symbol,
                exchange=self.exchange,
                orderid=str(self.limit_order_count),
                direction=stop_order.direction,
                offset=stop_order.offset,
                price=stop_order.price,
                volume=stop_order.volume,
                traded=0,
                status=Status.NOTTRADED,
                gateway_name=self.gateway_name,
                datetime=self.datetime
            )

            # 将状态更新前的委托推送到策略
            self.strategy.on_stop_order(stop_order)
            self.strategy.on_order(order)

            if order.offset != Offset.OPEN and self.strategy.trade_net_volume  < order.volume:                
                self.limit_orders[order.vt_orderid] = order
                self.active_stop_orders.pop(stop_order.stop_orderid)
                print("！" * 60)
                print(f"cross_stop_order报错！平仓委托交易数量超过可平仓净头寸！当前净持仓量：{self.strategy.trade_net_volume}，问题委托信息：{order}")
                continue

            # Push order udpate with status "all traded" (filled).
            order.traded = stop_order.volume
            order.status = Status.ALLTRADED
            self.limit_orders[order.vt_orderid] = order

            # Update stop order.
            stop_order.vt_orderids.append(order.vt_orderid)
            stop_order.status = StopOrderStatus.TRIGGERED

            if stop_order.stop_orderid in self.active_stop_orders:
                self.active_stop_orders.pop(stop_order.stop_orderid)

            # Push update to strategy.
            self.strategy.on_stop_order(stop_order)
            self.strategy.on_order(order)

            # Create trade data.
            if long_cross:
                trade_price = max(stop_order.price, long_best_price)
                pos_change = order.volume
            else:
                trade_price = min(stop_order.price, short_best_price)
                pos_change = -order.volume

            self.trade_count += 1

            trade = TradeData(
                symbol=order.symbol,
                exchange=order.exchange,
                orderid=order.orderid,
                tradeid=str(self.trade_count),
                direction=order.direction,
                offset=order.offset,
                price=trade_price,
                volume=order.volume,
                datetime=self.datetime,
                gateway_name=self.gateway_name,
            )

            self.trades[trade.vt_tradeid] = trade

            self.strategy.pos += pos_change
            self.strategy.on_trade(trade)

    def load_bar(
        self,
        vt_symbol: str,
        days: int,
        interval: Interval,
        frequency: int,
        callback: Callable,
        use_database: bool
    ):
        """"""
        self.days = days
        self.frequency = frequency
        self.callback = callback

    def load_tick(self, vt_symbol: str, days: int, callback: Callable):
        """"""
        self.days = days
        self.callback = callback

    def send_order(
        self,
        strategy: CtaTemplate,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        stop: bool,
        lock: bool,
        market: bool
    ):
        """"""
        price = round_to(price, self.pricetick)
        if stop:
            vt_orderid = self.send_stop_order(direction, offset, price, volume)
        else:
            vt_orderid = self.send_limit_order(direction, offset, price, volume)
        return [vt_orderid]

    def send_stop_order(
        self,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float
    ):
        """"""
        self.stop_order_count += 1

        stop_order = StopOrder(
            vt_symbol=self.vt_symbol,
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            stop_orderid=f"{STOPORDER_PREFIX}.{self.stop_order_count}",
            strategy_name=self.strategy.strategy_name,
        )

        self.active_stop_orders[stop_order.stop_orderid] = stop_order
        self.stop_orders[stop_order.stop_orderid] = stop_order

        return stop_order.stop_orderid

    def send_limit_order(
        self,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float
    ):
        """"""
        self.limit_order_count += 1

        order = OrderData(
            symbol=self.symbol,
            exchange=self.exchange,
            orderid=str(self.limit_order_count),
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=Status.SUBMITTING,
            gateway_name=self.gateway_name,
            datetime=self.datetime
        )

        self.active_limit_orders[order.vt_orderid] = order
        self.limit_orders[order.vt_orderid] = order

        return order.vt_orderid

    def cancel_order(self, strategy: CtaTemplate, vt_orderid: str):
        """
        Cancel order by vt_orderid.
        """
        if vt_orderid.startswith(STOPORDER_PREFIX):
            self.cancel_stop_order(strategy, vt_orderid)
        else:
            self.cancel_limit_order(strategy, vt_orderid)

    def cancel_stop_order(self, strategy: CtaTemplate, vt_orderid: str):
        """"""
        if vt_orderid not in self.active_stop_orders:
            return
        stop_order = self.active_stop_orders.pop(vt_orderid)

        stop_order.status = StopOrderStatus.CANCELLED
        self.strategy.on_stop_order(stop_order)

    def cancel_limit_order(self, strategy: CtaTemplate, vt_orderid: str):
        """"""
        if vt_orderid not in self.active_limit_orders:
            return
        order = self.active_limit_orders.pop(vt_orderid)

        order.status = Status.CANCELLED
        self.strategy.on_order(order)

    def cancel_all(self, strategy: CtaTemplate):
        """
        Cancel all orders, both limit and stop.
        """
        vt_orderids = list(self.active_limit_orders.keys())
        for vt_orderid in vt_orderids:
            self.cancel_limit_order(strategy, vt_orderid)

        stop_orderids = list(self.active_stop_orders.keys())
        for vt_orderid in stop_orderids:
            self.cancel_stop_order(strategy, vt_orderid)

    def write_log(self, msg: str, strategy: CtaTemplate = None):
        """
        Write log message.
        """
        msg = f"{self.datetime}\t{msg}"
        self.logs.append(msg)

    def send_email(self, msg: str, strategy: CtaTemplate = None):
        """
        Send email to default receiver.
        """
        pass

    def sync_strategy_data(self, strategy: CtaTemplate):
        """
        Sync strategy data into json file.
        """
        pass

    def get_engine_type(self):
        """
        Return engine type.
        """
        return self.engine_type

    def get_pricetick(self, strategy: CtaTemplate):
        """
        Return contract pricetick data.
        """
        return self.pricetick

    def put_strategy_event(self, strategy: CtaTemplate):
        """
        Put an event to update strategy status.
        """
        pass

    def output(self, msg):
        """
        Output message of backtesting engine.
        """
        print(f"{datetime.now()}\t{msg}")

    def get_all_trades(self):
        """
        Return all trade data of current backtesting result.
        """
        return list(self.trades.values())

    def get_all_orders(self):
        """
        Return all limit order data of current backtesting result.
        """
        return list(self.limit_orders.values())

    def get_all_daily_results(self):
        """
        Return all daily result data.
        """
        return list(self.daily_results.values())

# =================================================================================================================================================================================================================

    # 新增回测统计指标
    def get_bar_data_df(self):
        """获取加载的Bar历史数据，并生成 DataFrame"""
        bar_data = [bar.__dict__ for bar in self.history_data]
        bar_data_df = DataFrame(bar_data)
        bar_data_df = bar_data_df.set_index("datetime")
        bar_data_df = bar_data_df[["symbol", "open_price", "high_price", "low_price", "close_price", "volume", "open_interest"]]

        self.bar_data_df = bar_data_df

    def get_trade_data_df(self):
        """提取成交记录，并生成 DataFrame"""
        trades = self.get_all_trades()

        if trades:
            trade_list = [trade.__dict__ for trade in trades]
            trade_df = DataFrame(trade_list)
            trade_df = trade_df.set_index("datetime")
            trade_df = trade_df.sort_index()
            # trade_df包括字段："datetime", "gateway_name", "symbol", "exchange", "orderid", "tradeid", "direction", "offset", "price", "volume", "vt_symbol", "vt_orderid", "vt_tradeid"
            trade_df = trade_df.rename(columns={"price": "trade_price", "volume": "trade_volume"})
            trade_df["exchange"] = trade_df.exchange.apply(lambda x : x.value)
            trade_df["direction"] = trade_df.direction.apply(lambda x : x.value)
            trade_df["offset"] = trade_df.offset.apply(lambda x : x.value)
            trade_df["net_volume"] = self.strategy.trade_net_volume_list
            trade_df["signal"] = self.strategy.signal_list
            trade_df["strategy"] = self.strategy_class.__name__

            self.trade_data_df = trade_df

    def calculate_trade_result(self):
        """计算每笔交易盈亏"""
        if not self.trades:
            return

        trade_number_list = self.strategy.trade_number_list
        trade_type_list = self.strategy.trade_type_list
        trade_pnl_list = self.strategy.trade_pnl_list
        commission_list = self.strategy.commission_list
        slippage_list = self.strategy.slippage_list
        net_pnl_list = self.strategy.net_pnl_list
        balance_list = self.strategy.balance_list
        pnl_point_list = self.strategy.pnl_point_list
        trade_volume_list = self.strategy.trade_volume_list
        trade_duration_list = self.strategy.trade_duration_list
        trade_date_list = self.strategy.trade_date_list
        trade_start_time_list = self.strategy.trade_start_time_list
        trade_end_time_list = self.strategy.trade_end_time_list

        # 处理回测最后交易为未结清头寸的情况
        last_trade_net_volume = self.strategy.trade_net_volume_list[-1]

        if last_trade_net_volume > 0:
            last_close_price = self.history_data[-1].close_price          # 获取回测期加载历史数据最后一根Bar的收盘价
            last_datetime = self.history_data[-1].datetime                # 获取回测期加载历史数据最后一根Bar的时间

            trades = self.get_all_trades()
            # 最后一笔成交为买开或卖平时，平多仓
            if (trades[-1].direction == Direction.LONG and trades[-1].offset == Offset.OPEN) \
                or (trades[-1].direction == Direction.SHORT and (trades[-1].offset == Offset.CLOSE
                    or trades[-1].offset == Offset.CLOSETODAY or trades[-1].offset == Offset.CLOSEYESTERDAY)):
                trade_pnl = (last_close_price - self.strategy.long_cost) * last_trade_net_volume * self.size
                pnl_point = last_close_price - self.strategy.long_cost
                trade_type = "多头"
            
            # 最后一笔成交为卖开或买平时，平空仓
            else:
                trade_pnl = (self.strategy.short_cost - last_close_price) * last_trade_net_volume * self.size
                pnl_point = self.strategy.short_cost - last_close_price
                trade_type = "空头"

            if self.rate_type == RateType.FIXED:        # 固定手续费模式
                trade_commission = last_trade_net_volume * self.rate
            else:                                       # 浮动手续费模式
                trade_commission = last_close_price * last_trade_net_volume * self.size * self.rate
            trade_slippage = last_trade_net_volume * self.size * self.slippage

            net_pnl = trade_pnl - trade_commission - trade_slippage
            balance = round(net_pnl, 2) + self.strategy.balance

            trade_volume = last_trade_net_volume + self.strategy.trade_volume
            trade_pnl = trade_pnl + self.strategy.trade_pnl
            trade_commission = trade_commission + self.strategy.trade_commission
            trade_slippage = trade_slippage + self.strategy.trade_slippage
            net_pnl = net_pnl + self.strategy.net_pnl

            # 完成最后一次完整交易，将结果添加至列表尾
            trade_number = self.strategy.trade_number + 1
            trade_number_list.append(trade_number)
            trade_type_list.append(trade_type)
            trade_pnl_list.append(trade_pnl)
            commission_list.append(trade_commission)
            slippage_list.append(trade_slippage)
            net_pnl_list.append(net_pnl)
            balance_list.append(balance)
            pnl_point_list.append(pnl_point)
            trade_volume_list.append(trade_volume)
            trade_duration_list.append(self.strategy.entry_minute)
            trade_date_list.append(last_datetime.date())
            trade_end_time_list.append(last_datetime)

            
        trade_result_df = DataFrame({
            "trade_type": trade_type_list,
            "trade_pnl": trade_pnl_list,
            "trade_commission": commission_list,
            "trade_slippage": slippage_list,
            "net_pnl": net_pnl_list,
            "balance": balance_list,
            "pnl_point": pnl_point_list,
            "trade_volume": trade_volume_list,
            "duration": trade_duration_list,
            "trade_date": trade_date_list,
            "start_time": trade_start_time_list,
            "end_time": trade_end_time_list,
            "symbol": self.symbol,
            "strategy": self.strategy_class.__name__,
        }, index=trade_number_list)

        trade_result_df.index.name = "trade_number"

        self.trade_result_df = trade_result_df

    def calculate_statistics(self, trade_df: DataFrame = None, daily_df: DataFrame = None, capital = None, save_statistics = False, save_csv = False, save_chart = False, output=True):
        """计算统计指标"""
        if self.symbol:
            self.output(f"开始计算 {self.symbol} {self.strategy_class.__name__} 统计指标")
        else:
            self.output("开始计算投资组合统计指标")

        # 内部调用DataFrame和capital
        if trade_df is None and daily_df is None:
            trade_df = self.trade_result_df
            daily_df = self.daily_df
            capital  = self.capital
        else:
            # 外部传入时，必须传入trade_df，daily_df，capital
            if capital is None:
                return

            # 根据传入的capital重新计算 trade_df 的 balance，以便提取结束资金end_balance
            trade_df["balance"] = trade_df["net_pnl"].cumsum() + capital

        # Check for init DataFrame
        if trade_df is None and not self.trades:
            # 没有成交记录则设置所有统计指标为0
            start_date = datetime.now().date()
            end_date = datetime.now().date()

            total_days = 0
            profit_days = 0
            loss_days = 0

            end_balance = 0

            total_return = 0
            annual_return = 0
            cagr = 0
            annual_volatility = 0
            max_drawdown = 0
            max_ddpercent = 0
            average_drawdown = 0
            lw_drawdown = 0
            average_square_drawdown = 0
            max_drawdown_duration = 0
            max_drawdown_range = ""

            daily_return = 0
            return_std = 0
            return_drawdown_ratio = 0
            sharpe_ratio = 0
            omega_ratio = 0
            calmar_ratio = 0
            downside_risk = 0
            sortino_ratio = 0
            R_squared = 0
            tail_ratio = 0

            total_pnl = 0
            total_net_pnl = 0
            total_commission = 0
            total_slippage = 0
            total_pnl_point = 0
            total_trade_count = 0
            total_trade = 0

            daily_net_pnl = 0
            daily_commission = 0
            daily_slippage = 0
            daily_pnl_point = 0
            daily_trade_count = 0
            daily_trade_max = 0
            trade_volume_max = 0

            max_profit = 0
            max_loss = 0
            profit_times = 0
            loss_times = 0
            rate_of_win = 0
            profit_loss_ratio = 0

            average_net_pnl = 0
            average_commission = 0
            average_slippage = 0
            average_pnl_point = 0
            trade_duration = 0
            trade_duration_max = 0

            total_profit = 0
            profit_mean = 0
            profit_duration = 0
            profit_duration_max = 0

            total_loss = 0
            loss_mean = 0
            loss_duration = 0
            loss_duration_max = 0

            long_total_trade = 0
            long_profit_times = 0
            long_loss_times = 0
            long_rate_of_win = 0
            long_total_profit = 0
            long_total_loss = 0
            long_profit_loss_ratio = 0
            long_total_net_pnl = 0
            long_average_net_pnl = 0
            long_trade_duration = 0
            long_trade_duration_max = 0

            short_total_trade = 0
            short_profit_times = 0
            short_loss_times = 0
            short_rate_of_win = 0
            short_total_profit = 0
            short_total_loss = 0
            short_profit_loss_ratio = 0
            short_total_net_pnl = 0
            short_average_net_pnl = 0
            short_trade_duration = 0
            short_trade_duration_max = 0

        else:
            # 计算基于trade_df的统计指标
            total_trade = len(trade_df)                                                                # 总交易笔数
            max_profit = trade_df["net_pnl"].max()                                                     # 单笔最大盈利
            max_loss = trade_df["net_pnl"].min()                                                       # 单笔最大亏损
            profit_times = len(trade_df[trade_df["net_pnl"] >= 0])                                     # 交易盈利笔数
            loss_times = len(trade_df[trade_df["net_pnl"] < 0])                                        # 交易亏损笔数
            rate_of_win = profit_times / (profit_times + loss_times) * 100                             # 胜率 = 盈利的所有次数 / 总交易场次 x 100%

            total_profit = trade_df[trade_df["net_pnl"] >= 0].net_pnl.sum()                            # 盈利总金额
            profit_mean = total_profit / profit_times                                                  # 盈利交易均值
            profit_duration = trade_df[trade_df["net_pnl"] >= 0].duration.mean()                       # 盈利持仓小时
            profit_duration_max = trade_df[trade_df["net_pnl"] >= 0].duration.max()                    # 盈利最长持仓

            total_loss = trade_df[trade_df["net_pnl"] < 0].net_pnl.sum()                               # 亏损总金额
            loss_mean = total_loss / loss_times                                                        # 亏损交易均值
            loss_duration = trade_df[trade_df["net_pnl"] < 0].duration.mean()                          # 亏损持仓小时
            loss_duration_max = trade_df[trade_df["net_pnl"] < 0].duration.max()                       # 亏损最长持仓

            profit_loss_ratio = (total_profit/profit_times) / abs(total_loss / loss_times)             # 盈亏比 = 盈利的平均金额 / 亏损的平均金额

            total_net_pnl = trade_df["net_pnl"].sum()                                                  # 交易净盈亏
            average_net_pnl = total_net_pnl / total_trade                                                   # 平均每笔净盈亏
            trade_duration = trade_df["duration"].mean()                                               # 平均持仓小时
            trade_duration_max = trade_df["duration"].max()                                            # 最长持仓小时

            total_pnl = trade_df["trade_pnl"].sum()                                                    # 交易总盈亏
            total_commission = trade_df["trade_commission"].sum()                                      # 总交易手续费
            total_slippage = trade_df["trade_slippage"].sum()                                          # 总交易滑点费
            total_pnl_point = trade_df["pnl_point"].sum()                                              # 总交易净盈亏点数
            average_commission = total_commission / total_trade                                        # 平均每笔手续费
            average_slippage = total_slippage / total_trade                                            # 平均每笔滑点费
            average_pnl_point = total_pnl_point / total_trade                                          # 平均每笔净盈亏点数

            long_total_trade = len(trade_df[trade_df["trade_type"] == "多头"])                                                     # 多头交易笔数
            long_profit_times = len(trade_df[(trade_df["trade_type"] == "多头") & (trade_df["net_pnl"] >= 0)])                     # 多头盈利笔数
            long_loss_times = len(trade_df[(trade_df["trade_type"] == "多头") & (trade_df["net_pnl"] < 0)])                        # 多头亏损笔数

            try:
                long_rate_of_win = long_profit_times / (long_profit_times + long_loss_times) * 100                                 # 多头胜率
            except ZeroDivisionError:
                long_rate_of_win = 0

            long_total_profit = trade_df[(trade_df["trade_type"] == "多头") & (trade_df["net_pnl"] >= 0)].net_pnl.sum()            # 多头盈利总金额
            long_total_loss = trade_df[(trade_df["trade_type"] == "多头") & (trade_df["net_pnl"] < 0)].net_pnl.sum()               # 多头亏损总金额
            long_profit_loss_ratio = (long_total_profit/long_profit_times) / abs(long_total_loss / long_loss_times)                # 多头盈亏比
            long_total_net_pnl = trade_df[trade_df["trade_type"] == "多头"].net_pnl.sum()                                          # 多头交易净盈亏
            long_average_net_pnl = long_total_net_pnl / long_total_trade                                                                # 多头平均每笔净盈亏
            long_trade_duration = trade_df[trade_df["trade_type"] == "多头"].duration.mean()                                       # 多头平均持仓小时
            long_trade_duration_max = trade_df[trade_df["trade_type"] == "多头"].duration.max()                                    # 多头最长持仓小时

            short_total_trade = len(trade_df[trade_df["trade_type"] == "空头"])                                                     # 空头交易笔数
            short_profit_times = len(trade_df[(trade_df["trade_type"] == "空头") & (trade_df["net_pnl"] >= 0)])                     # 空头盈利笔数
            short_loss_times = len(trade_df[(trade_df["trade_type"] == "空头") & (trade_df["net_pnl"] < 0)])                        # 空头亏损笔数

            try:
                short_rate_of_win = short_profit_times / (short_profit_times + short_loss_times) * 100                              # 空头胜率
            except ZeroDivisionError:
                short_rate_of_win = 0

            short_total_profit = trade_df[(trade_df["trade_type"] == "空头") & (trade_df["net_pnl"] >= 0)].net_pnl.sum()            # 空头盈利总金额
            short_total_loss = trade_df[(trade_df["trade_type"] == "空头") & (trade_df["net_pnl"] < 0)].net_pnl.sum()               # 空头亏损总金额
            short_profit_loss_ratio = (short_total_profit/short_profit_times) / abs(short_total_loss / short_loss_times)            # 空头盈亏比
            short_total_net_pnl = trade_df[trade_df["trade_type"] == "空头"].net_pnl.sum()                                          # 空头交易净盈亏
            short_average_net_pnl = short_total_net_pnl / short_total_trade                                                              # 空头平均每笔净盈亏
            short_trade_duration = trade_df[trade_df["trade_type"] == "空头"].duration.mean()                                       # 空头平均持仓小时
            short_trade_duration_max = trade_df[trade_df["trade_type"] == "空头"].duration.max()                                    # 空头最长持仓小时


            # 计算基于daily_df的统计指标
            daily_df["balance"] = daily_df["net_pnl"].cumsum() + capital

            if len(daily_df[daily_df["balance"] <= 0]) > 0:
                self.output("*" * 30)
                self.output(f'账号爆仓！爆仓日期为:{daily_df[daily_df["balance"] <= 0].index[0]}')
                self.output("*" * 30)
            
            daily_df["return"] = np.log(daily_df["balance"] / daily_df["balance"].shift(1)).fillna(0)
            daily_df["highlevel"] = daily_df["balance"].rolling(min_periods=1, window=len(daily_df), center=False).max()
            daily_df["drawdown"] = daily_df["balance"] - daily_df["highlevel"]
            daily_df["ddpercent"] = daily_df["drawdown"] / daily_df["highlevel"] * 100

            start_date = daily_df.index[0]
            end_date = daily_df.index[-1]

            total_days = len(daily_df)
            profit_days = len(daily_df[daily_df["net_pnl"] > 0])
            loss_days = len(daily_df[daily_df["net_pnl"] < 0])

            end_balance = trade_df["balance"].iloc[-1]                 # 此处调整为基于trade_df的结束资金，算上未平仓交易的手续费和滑点费

            max_drawdown = daily_df["drawdown"].min()
            max_ddpercent = daily_df["ddpercent"].min()                               # 百分比最大回撤：empyrical.max_drawdown(returns)
            average_drawdown = daily_df["ddpercent"].mean()                           # 百分比平均回撤：就是每日百分比回撤的算数平均，即采用简单的算术平均计算策略整体每日亏损的风险。
            lw_drawdown = talib.LINEARREG(daily_df["ddpercent"], total_days)[-1]      # 百分比线性加权回撤：就是使用最小二乘法（OLS）对回撤曲线进行线性回归，从而得到一条回撤的趋势线。最小化误差平方和的方法有利于避免极端情况的影响，让我们把关注点更多集中在策略的整体风险水平上。一个好的策略，其回撤的回归直线的斜率应该尽可能的小。
            daily_df["ddpercent^2"] = daily_df["ddpercent"] * daily_df["ddpercent"]                             
            average_square_drawdown = - daily_df["ddpercent^2"].mean()                # 百分比均方回撤：每日百分比回撤的平方的期望值，采用这种计量方法的原因在于认为策略样本内回测属于小样本评估，属于有偏估算。
            max_ddpercent_end = daily_df["ddpercent"].idxmin()

            if isinstance(max_ddpercent_end, date):
                max_ddpercent_start = daily_df["balance"][:max_ddpercent_end].idxmax()
                max_drawdown_duration = (max_ddpercent_end - max_ddpercent_start).days    # %最大回撤天数
                max_drawdown_range = f"{max_ddpercent_start}~{max_ddpercent_end}"         # %最大回撤区间
            else:
                max_drawdown_duration = 0

            daily_net_pnl = total_net_pnl / total_days                # 日均净盈亏
            daily_commission = total_commission / total_days          # 日均手续费
            daily_slippage = total_slippage / total_days              # 日均滑点费
            daily_pnl_point = total_pnl_point / total_days            # 日均净盈亏点数
            daily_trade_count = total_trade / total_days              # 日均交易笔数
            daily_trade_max = int(trade_df[["trade_date", "symbol"]].groupby("trade_date").count().max())    # 单日最多交易笔数

            try:
                trade_volume_max = self.trade_data_df.trade_volume.max()        # 单次最大成交手数
            except AttributeError:
                trade_volume_max = 0

            total_trade_count = daily_df["trade_count"].sum()         # 总共成交的委托数量

            total_return = (end_balance / capital - 1) * 100          # 总收益率
            annual_return = total_return / total_days * 240           # 年化收益率：empyrical.annual_return(returns, period='daily', annualization=None)
            daily_return = daily_df["return"].mean() * 100            # 日均收益率
            return_std = daily_df["return"].std() * 100               # 收益标准差
            return_drawdown_ratio = -annual_return / max_ddpercent    # 收益回撤比(年化收益率/最大回撤率)

            if return_std:
                sharpe_ratio = daily_return / return_std * np.sqrt(240)     # 夏普比率：empyrical.sharpe_ratio(returns, risk_free=0, period='daily', annualization=None)
            else:
                sharpe_ratio = 0

            cagr = empyrical.cagr(daily_df["return"], period='daily', annualization=240) * 100                                          # 年复合增长率
            annual_volatility = empyrical.annual_volatility(daily_df["return"], alpha=2.0, annualization=240) * 100                     # 年化波动率：用来测量策略的风险性，波动越大代表策略风险越高。
            omega_ratio = empyrical.omega_ratio(daily_df["return"], risk_free=0.0, required_return=0.0, annualization=240)              # Omega比率：上涨概率比下跌概率，衡量的是收益-损失比率，一般越大越好。
            calmar_ratio = empyrical.calmar_ratio(daily_df["return"], annualization=240)                                                # Calmar比率：描述的是收益和最大回撤之间的关系。计算方式为年化复合收益率与历史最大回撤之间的比率。Calmar比率数值越大，基金的业绩表现越好。反之，基金的业绩表现越差。
            downside_risk = empyrical.downside_risk(daily_df["return"], required_return=0, annualization=240)                           # 下限风险：在基金投资中，下限风险是指基金净值增长率的下方标准差，测量的是基金的净值增长率与目标回报率或者期望回报率反向的偏差程度。
            sortino_ratio = empyrical.sortino_ratio(daily_df["return"], required_return=0, annualization=240, _downside_risk=None)      # 索提诺比率：一种衡量投资组合相对表现的方法。与夏普比率(Sharpe Ratio)有相似之处，但索提诺比率运用下偏标准差而不是总标准差，以区别不利和有利的波动。和夏普比率类似，这一比率越高，表明基金承担相同单位下行风险能获得更高的超额回报率。索提诺比率可以看做是夏普比率在衡量对冲基金/私募基金时的一种修正方式。
            R_squared = empyrical.stability_of_timeseries(daily_df["return"])                                                           # R平方：是反映业绩基准的变动对基金表现的影响，影响程度以 0～100 计。如果R平方值等于100，表示基金回报的变动完全由业绩基准的变动所致；若R平方值等于35，即35%的基金回报可归因于业绩基准的变动。简言之，R 平方值越低，由业绩基准变动导致的基金业绩的变动便越少。此外，R平方也可用来确定β系数或α系数的准确性。一般而言，基金的R平方值越高，其两个系数的准确性便越高。
            tail_ratio = empyrical.tail_ratio(daily_df["return"])                                                                       # 尾部比率：

            # 用于生成每日平仓头寸后的净盈亏散点图
            trade_daily_df = trade_df[["trade_date", "net_pnl"]].groupby("trade_date").sum()
            self.trade_daily_df = trade_daily_df.sort_index()


        # 统计指标
        statistics = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": capital,
            "end_balance": round(end_balance,2),
            "total_return": round(total_return,2),
            "annual_return": round(annual_return,2),
            "cagr": round(cagr,2),
            "annual_volatility": round(annual_volatility,2),

            "max_drawdown": round(max_drawdown,2),
            "max_ddpercent": round(max_ddpercent,2),
            "average_drawdown": round(average_drawdown,2),
            "lw_drawdown": round(lw_drawdown,2),
            "average_square_drawdown": round(average_square_drawdown,2),
            "max_drawdown_duration": max_drawdown_duration,
            "max_drawdown_range": max_drawdown_range,

            "sharpe_ratio": round(sharpe_ratio,2),
            "sortino_ratio": round(sortino_ratio,2),
            "omega_ratio": round(omega_ratio,2),
            "calmar_ratio": round(calmar_ratio,2),
            "return_drawdown_ratio": round(return_drawdown_ratio,2),
            "R_squared": round(R_squared,2),
            "tail_ratio": round(tail_ratio,2),
            "downside_risk": round(downside_risk,2),
            "daily_return": round(daily_return,2),
            "return_std": round(return_std,2),

            "total_pnl": round(total_pnl,2),
            "total_net_pnl": round(total_net_pnl,2),
            "total_commission": round(total_commission,2),
            "total_slippage": round(total_slippage,2),
            "total_pnl_point": round(total_pnl_point,2),
            "total_trade_count": total_trade_count,
            "total_trade": total_trade,

            "daily_net_pnl": round(daily_net_pnl,2),
            "daily_commission": round(daily_commission,2),
            "daily_slippage": round(daily_slippage,2),
            "daily_pnl_point": round(daily_pnl_point,2),
            "daily_trade_count": round(daily_trade_count,2),

            "daily_trade_max": daily_trade_max,
            "trade_volume_max": trade_volume_max,
            "max_profit": round(max_profit,2),
            "max_loss": round(max_loss,2),

            "profit_times": profit_times,
            "loss_times": loss_times,
            "rate_of_win": round(rate_of_win,2),
            "profit_loss_ratio": round(profit_loss_ratio,2),
            "average_net_pnl": round(average_net_pnl,2),
            "average_commission": round(average_commission,2),
            "average_slippage": round(average_slippage,2),
            "average_pnl_point": round(average_pnl_point,2),
            "trade_duration": round(trade_duration,2),
            "trade_duration_max": round(trade_duration_max,2),

            "total_profit": round(total_profit,2),
            "total_loss": round(total_loss,2),
            "profit_mean": round(profit_mean,2),
            "loss_mean": round(loss_mean,2),
            "profit_duration": round(profit_duration,2),
            "loss_duration": round(loss_duration,2),
            "profit_duration_max": round(profit_duration_max,2),
            "loss_duration_max": round(loss_duration_max,2),

            "long_total_trade": long_total_trade,
            "long_profit_times": long_profit_times,
            "long_loss_times": long_loss_times,
            "long_rate_of_win": round(long_rate_of_win,2),
            "long_total_profit": round(long_total_profit,2),
            "long_total_loss": round(long_total_loss,2),
            "long_profit_loss_ratio": round(long_profit_loss_ratio,2),
            "long_total_net_pnl": round(long_total_net_pnl,2),
            "long_average_net_pnl": round(long_average_net_pnl,2),
            "long_trade_duration": round(long_trade_duration,2),
            "long_trade_duration_max": round(long_trade_duration_max,2),

            "short_total_trade": short_total_trade,
            "short_profit_times": short_profit_times,
            "short_loss_times": short_loss_times,
            "short_rate_of_win": round(short_rate_of_win,2),
            "short_total_profit": round(short_total_profit,2),
            "short_total_loss": round(short_total_loss,2),
            "short_profit_loss_ratio": round(short_profit_loss_ratio,2),
            "short_total_net_pnl": round(short_total_net_pnl,2),
            "short_average_net_pnl": round(short_average_net_pnl,2),
            "short_trade_duration": round(short_trade_duration,2),
            "short_trade_duration_max": round(short_trade_duration_max,2),
        }

        # 转换成中文
        statistics_cn = {
            "首个交易日": start_date,
            "最后交易日": end_date,
            "总交易日": total_days,
            "盈利交易日": profit_days,
            "亏损交易日": loss_days,
            "起始资金": f"{capital:.2f}",
            "结束资金": f"{end_balance:.2f}",
            "总收益率": f"{total_return:.2f}%",
            "年化收益率": f"{annual_return:.2f}%",
            "年复合增长率": f"{cagr:.2f}%",
            "年化波动率": f"{annual_volatility:.2f}%",

            "最大回撤金额": f"{max_drawdown:.2f}",
            "%最大回撤": f"{max_ddpercent:.2f}%",
            "%平均回撤": f"{average_drawdown:.2f}%",
            "%线性加权回撤": f"{lw_drawdown:.2f}%",
            "%均方回撤": f"{average_square_drawdown:.2f}%",
            "最大回撤天数": max_drawdown_duration,
            "最大回撤区间": max_drawdown_range,

            "夏普比率": f"{sharpe_ratio:.2f}",
            "索提诺比率": f"{sortino_ratio:.2f}",
            "Omega比率": f"{omega_ratio:.2f}",
            "Calmar比率": f"{calmar_ratio:.2f}",
            "收益回撤比": f"{return_drawdown_ratio:.2f}",
            "R平方": f"{R_squared:.2f}",
            "尾部比率": f"{tail_ratio:.2f}",
            "下限风险": f"{downside_risk:.2f}",
            "日均收益率": f"{daily_return:.2f}%",
            "收益标准差": f"{return_std:.2f}%",

            "交易总盈亏": f"{total_pnl:.2f}",
            "交易净盈亏": f"{total_net_pnl:.2f}",
            "交易手续费": f"{total_commission:.2f}",
            "交易滑点费": f"{total_slippage:.2f}",
            "总净盈亏点数": f"{total_pnl_point:.2f}",
            "总成交数量": total_trade_count,
            "总交易笔数": total_trade,

            "日均净盈亏": f"{daily_net_pnl:.2f}",
            "日均手续费": f"{daily_commission:.2f}",
            "日均滑点费": f"{daily_slippage:.2f}",
            "日均净盈亏点数": f"{daily_pnl_point:.2f}",
            "日均交易笔数": f"{daily_trade_count:.2f}",

            "单日最多交易笔数": daily_trade_max,
            "单次最大成交手数": trade_volume_max,
            "单笔最大盈利": f"{max_profit:.2f}",
            "单笔最大亏损": f"{max_loss:.2f}",

            "交易盈利笔数": profit_times,
            "交易亏损笔数": loss_times,
            "胜率": f"{rate_of_win:.2f}%",
            "盈亏比": f"{profit_loss_ratio:.2f}",
            "平均每笔净盈亏": f"{average_net_pnl:.2f}",
            "平均每笔手续费": f"{average_commission:.2f}",
            "平均每笔滑点费": f"{average_slippage:.2f}",
            "平均每笔净盈亏点数": f"{average_pnl_point:.2f}",
            "平均持仓小时": f"{trade_duration:.2f}",
            "最长持仓小时": f"{trade_duration_max:.2f}",

            "盈利总金额": f"{total_profit:.2f}",
            "亏损总金额": f"{total_loss:.2f}",
            "盈利交易均值": f"{profit_mean:.2f}",
            "亏损交易均值": f"{loss_mean:.2f}",
            "盈利持仓小时": f"{profit_duration:.2f}",
            "亏损持仓小时": f"{loss_duration:.2f}",
            "盈利最长持仓": f"{profit_duration_max:.2f}",
            "亏损最长持仓": f"{loss_duration_max:.2f}",

            "多头胜率": f"{long_rate_of_win:.2f}%",
            "空头胜率": f"{short_rate_of_win:.2f}%",
            "多头盈亏比": f"{long_profit_loss_ratio:.2f}",
            "空头盈亏比": f"{short_profit_loss_ratio:.2f}",
            "多头交易笔数": f"{long_total_trade}",
            "空头交易笔数": f"{short_total_trade}",
            "多头盈利笔数": f"{long_profit_times}",
            "空头盈利笔数": f"{short_profit_times}",
            "多头亏损笔数": f"{long_loss_times}",
            "空头亏损笔数": f"{short_loss_times}",
            "多头盈利总金额": f"{long_total_profit:.2f}",
            "空头盈利总金额": f"{short_total_profit:.2f}",
            "多头亏损总金额": f"{long_total_loss:.2f}",
            "空头亏损总金额": f"{short_total_loss:.2f}",
            "多头交易净盈亏": f"{long_total_net_pnl:.2f}",
            "空头交易净盈亏": f"{short_total_net_pnl:.2f}",
            "多头平均每笔净盈亏": f"{long_average_net_pnl:.2f}",
            "空头平均每笔净盈亏": f"{short_average_net_pnl:.2f}",
            "多头平均持仓小时": f"{long_trade_duration:.2f}",
            "空头平均持仓小时": f"{short_trade_duration:.2f}",
            "多头最长持仓小时": f"{long_trade_duration_max:.2f}",
            "空头最长持仓小时": f"{short_trade_duration_max:.2f}",
        }


        # Filter potential error infinite value
        for key, value in statistics.items():
            if value in (np.inf, -np.inf):
                value = 0
            statistics[key] = np.nan_to_num(value)

        self.output("策略统计指标计算完成")


        # Output
        if output:
            self.output("-" * 60)

            for key, value in statistics_cn.items():
                self.output(f'{key:15}\t{value}')

            self.output("-" * 60)

        if self.trades:
            signal_times = self.trade_data_df[["signal", "symbol"]].groupby(["signal"]).count()
            for ix, row in signal_times.iterrows():
                if row.name < 2:
                    self.output(f"开多 {row.name:<5}\t{row.symbol}")
                elif row.name < 3:
                    self.output(f"开空 {row.name:<5}\t{row.symbol}")
                elif row.name < 4:
                    self.output(f"平多 {row.name:<5}\t{row.symbol}")
                elif row.name < 5:
                    self.output(f"平空 {row.name:<5}\t{row.symbol}")

            self.output("-" * 60)

        # 生成桌面路径
        home_path = Path.home()
        temp_path = home_path.joinpath("Desktop")

        # 生成主要参数字符串
        if self.symbol:
            param_dict = self.strategy.get_parameters()
            for key in ["x_second", "long_period", "init_lots", "capital"]:

                try:
                    param_dict.pop(key)
                except KeyError:
                    pass
                    
            self.main_parameters = f"{param_dict}".replace(":","=").replace("'","").replace("{","").replace("}","").replace(" ","")

        # 生成交易日期范围
        self.trade_date_range = f'{start_date.strftime("%Y%m%d")}~{end_date.strftime("%Y%m%d")}'

        # 保存统计数据至桌面
        if save_statistics:
            
            if self.symbol:
                filename  = f"回测统计数据[{self.symbol} {self.strategy_class.__name__}][{self.trade_date_range}][{self.main_parameters}].txt"
                filepath  = temp_path.joinpath(filename)
                
                with open(filepath, "w+") as f:
                    [f.write(f"{key:12}\t{value}\n") for key, value in statistics_cn.items()]

                    f.write("\n\n")

                    for ix, row in signal_times.iterrows():
                        if row.name < 2:
                            f.write(f"开多 {row.name:<8}\t{row.symbol}\n")
                        elif row.name < 3:
                            f.write(f"开空 {row.name:<8}\t{row.symbol}\n")
                        elif row.name < 4:
                            f.write(f"平多 {row.name:<8}\t{row.symbol}\n")
                        elif row.name < 5:
                            f.write(f"平空 {row.name:<8}\t{row.symbol}\n")


            else:
                filename  = f"投资组合统计数据[{self.trade_date_range}].txt"
                filepath  = temp_path.joinpath(filename)
                
                with open(filepath, "w+") as f:
                    [f.write(f"{key:12}\t{value}\n") for key, value in statistics_cn.items()]


        # 保存逐笔交易记录和逐日盯市记录至桌面CSV
        if save_csv:
            if self.symbol:
                trade_df.to_csv(temp_path.joinpath(f"逐笔交易记录 {self.symbol} {self.strategy_class.__name__} [{self.trade_date_range}].csv"))
                daily_df.to_csv(temp_path.joinpath(f"逐日盯市记录 {self.symbol} {self.strategy_class.__name__} [{self.trade_date_range}].csv"))

            else:
                trade_df.to_csv(temp_path.joinpath(f"投资组合逐笔交易记录 [{self.trade_date_range}].csv"))
                daily_df.to_csv(temp_path.joinpath(f"投资组合逐日盯市记录 [{self.trade_date_range}].csv"))


        # 保存分析图表至桌面
        if save_chart:

            statistics_chart = self.statistics_chart(statistics)
            daily_chart = self.daily_grid_chart(daily_df)
            trade_chart = self.trade_grid_chart(trade_df, trade_daily_df, daily_df)
            duration_grid_chart = self.duration_grid_chart(trade_df)

            tab_chart = Tab(page_title="综合分析图表")
            tab_chart.add(statistics_chart, "各类统计指标")
            tab_chart.add(daily_chart, "逐日收益分析图")
            tab_chart.add(trade_chart, "交易盈亏分布图")
            tab_chart.add(duration_grid_chart, "持仓时间分布图")

            # 生成文件保存路径
            if self.symbol:
                filename  = f"综合分析图表[{self.symbol} {self.strategy_class.__name__}][{self.trade_date_range}][{self.main_parameters}].html"
                filepath  = temp_path.joinpath(filename)
            else:
                filename  = f"投资组合分析图表[{self.trade_date_range}].html"
                filepath  = temp_path.joinpath(filename)

            tab_chart.render(filepath)


        self.statistics = statistics

        return statistics

    def generate_bar_data(self, bar_data_list = None, window = 1, interval = Interval.MINUTE, df = True) -> Union[DataFrame, list]:
        """通过1分钟Bar，生成指定周期的Bar数据，返回DataFrame或列表"""
        bg = BarGenerator(window=window, interval=interval)

        if bar_data_list is None:
            if self.history_data:
                for bar in self.history_data:
                    bg.generate_bar(bar)

                if df:
                    return bg.get_bar_data_df()
                else:
                    return bg.bar_data_list

        else:
            for bar in bar_data_list:
                bg.generate_bar(bar)

            if df:
                return bg.get_bar_data_df()
            else:
                return bg.bar_data_list

    # 单图模式，绘制蜡烛图和资金曲线，叠加主图技术指标
    def draw_kline_chart(self):
        trade_result_df = self.trade_result_df.set_index("start_time")
        trade_data = merge(self.trade_data_df, trade_result_df[["net_pnl", "balance"]], how="left", left_index=True, right_index=True)
        
        kline_chart = MyPyecharts(bar_data=self.bar_data_df, trade_data=trade_data, grid=False, grid_quantity=0, chart_id=20)
        kline_chart.kline()
        kline_chart.overlap_trade()
        kline_chart.overlap_net_pnl()
        # kline_chart.overlap_balance(capital=self.capital)
        # kline_chart.overlap_sma([5, 10, 20, 60])
        # kline_chart.overlap_boll(timeperiod=14, nbdevup=2, nbdevdn=2, matype=0)
        chart = kline_chart.grid_graph()

        return chart

    # 层叠多图模式，绘制蜡烛图，叠加主、副图技术指标
    def draw_grid_chart(self):        
        grid_chart = MyPyecharts(bar_data=self.bar_data_df, trade_data=self.trade_data_df, grid=True, grid_quantity=1, chart_id=30)
        grid_chart.kline()
        grid_chart.overlap_trade()
        # grid_chart.overlap_sma([5, 10, 20, 60])
        # grid_chart.overlap_boll(timeperiod=14, nbdevup=2, nbdevdn=2, matype=0)
        chart = grid_chart.grid_graph(grid_graph_1 = grid_chart.grid_macd(fastperiod=12, slowperiod=26, signalperiod=9, grid_index=1))

        return chart

    # 收益曲线，每日净盈亏，资金回撤曲线
    def daily_grid_chart(self, daily_df=None):
        if daily_df is None:
            daily_df = self.daily_df

        balance_line = (
            Line()
            .add_xaxis(xaxis_data = list(daily_df.index))
            .add_yaxis(
                series_name = "账号净值",
                y_axis = daily_df.balance.apply(lambda x: round(x, 2)).values.tolist(),
                label_opts = opts.LabelOpts(is_show=False),
                is_symbol_show = False,
                linestyle_opts = opts.LineStyleOpts(width=3, opacity=1, type_="solid", color="#8A0000"),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="账号净值", pos_top="0%"),

                # 多图组合
                datazoom_opts=[
                    opts.DataZoomOpts(
                        is_show=False,
                        type_="inside",
                        xaxis_index=[0, 0],
                        range_start=0,
                        range_end=100,
                    ),
                    opts.DataZoomOpts(
                        is_show=False,
                        type_="inside",
                        xaxis_index=[0, 1],
                        range_start=0,
                        range_end=100,
                    ),
                    opts.DataZoomOpts(
                        is_show=True,
                        type_="slider",
                        xaxis_index=[0, 2],
                        pos_top="96%",
                        range_start=0,
                        range_end=100,
                    ),
                ],
                    
                xaxis_opts = opts.AxisOpts(
                    is_scale=True,
                    type_="category",
                    boundary_gap=False,
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    split_number=20,
                    min_="dataMin",
                    max_="dataMax",
                ),

                yaxis_opts = opts.AxisOpts(
                    is_scale = True,
                    splitarea_opts = opts.SplitAreaOpts(is_show = True, areastyle_opts = opts.AreaStyleOpts(opacity=0.8)),
                    splitline_opts = opts.SplitLineOpts(is_show=True),
                ),
                
                brush_opts = opts.BrushOpts(
                    tool_box = ["rect", "polygon", "keep","lineX","lineY", "clear"],
                    x_axis_index = "all",
                    brush_link = "all",
                    out_of_brush = {"colorAlpha": 0.1},
                    brush_type = "lineX",
                ),
                
                tooltip_opts = opts.TooltipOpts(
                    is_show = True,
                    trigger = "axis",
                    trigger_on = "mousemove",
                    axis_pointer_type = "cross",
                    background_color = "rgba(245, 245, 245, 0.8)",
                    border_width = 1,
                    border_color = "#ccc",
                    textstyle_opts = opts.TextStyleOpts(color = "#000", font_size = 12, font_family = "Arial", font_weight = "lighter", ),
                ),

                toolbox_opts = opts.ToolboxOpts(orient = "horizontal", pos_left = "right", ),
                
                legend_opts = opts.LegendOpts(is_show = True, type_ = "scroll", selected_mode = "multiple", 
                                            pos_left = "40%", pos_top = "0%", legend_icon = "roundRect",),

                # 多图的 axis 连在一块
                axispointer_opts = opts.AxisPointerOpts(
                    is_show = True,
                    link=[{"xAxisIndex": "all"}],
                    label=opts.LabelOpts(background_color="#777"),
                ),
            )
        )

        net_pnl_bar = (
            Bar()
            .add_xaxis(xaxis_data = list(daily_df.index))
            .add_yaxis(
                series_name = "每日净盈亏",
                y_axis = daily_df.net_pnl.apply(lambda x: round(x, 2)).values.tolist(),
                label_opts = opts.LabelOpts(is_show=False),
                itemstyle_opts = opts.ItemStyleOpts(
                    color=JsCode(
                        """
                        function(params) {
                            var colorList;
                            if (params.data >= 0) {
                                colorList = '#ef232a';
                            } else {
                                colorList = '#14b143';
                            }
                            return colorList;
                        }
                        """
                    ),
                    opacity=1,
                ),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="每日净盈亏", pos_top="34.5%"),
                legend_opts = opts.LegendOpts(pos_left = "45%", pos_top = "0%"),
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                yaxis_opts=opts.AxisOpts(
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                    splitline_opts=opts.SplitLineOpts(is_show=True),
                    axislabel_opts=opts.LabelOpts(is_show=True),
                ),
            )
        )

        ddpercent_line = (
            Line()
            .add_xaxis(xaxis_data = list(daily_df.index))
            .add_yaxis(
                series_name = "净值回撤率",
                y_axis = daily_df.ddpercent.apply(lambda x: round(x, 2)).values.tolist(),
                label_opts = opts.LabelOpts(is_show=False),
                is_symbol_show = False,
                linestyle_opts = opts.LineStyleOpts(width=2, opacity=1, type_="solid", color="#8A0000"),
                areastyle_opts=opts.AreaStyleOpts(opacity=0.5, color="#8A0000"),        # 区域填充样式配置项
            ) 
            .set_global_opts(
                title_opts=opts.TitleOpts(title="净值回撤率", pos_top="66%"),
                legend_opts = opts.LegendOpts(pos_left = "50%", pos_top = "0%"),
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                yaxis_opts=opts.AxisOpts(
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                    splitline_opts=opts.SplitLineOpts(is_show=True),
                    axislabel_opts=opts.LabelOpts(is_show=True),
                ),
            )
        )

        grid_chart = (
            Grid(init_opts=opts.InitOpts(width="1900px", height="900px", chart_id="收益回撤图"))
            .add(balance_line, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="5%", height="27%"))
            .add(net_pnl_bar, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="38%", height="27%"))
            .add(ddpercent_line, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="69%", height="27%"))
        )

        return grid_chart

    # 每笔交易盈亏分布，每日交易盈亏分布
    def trade_grid_chart(self, trade_df = None, trade_daily_df = None, daily_df = None):
        if trade_df is None and daily_df is None:
            trade_df = self.trade_result_df
            trade_daily_df = self.trade_daily_df
            daily_df = self.daily_df

        trade_scatter = (
            EffectScatter()
            .add_xaxis(xaxis_data = trade_df.end_time.apply(lambda x:x.strftime("%Y-%m-%d %H:%M")).values.tolist())
            .add_yaxis(
                series_name = "每笔净盈亏",
                y_axis = trade_df.net_pnl.apply(lambda x: round(x, 2)).values.tolist(),
                symbol_size = 8,
                label_opts = opts.LabelOpts(is_show=False),
            ) 
            .set_global_opts(
                title_opts=opts.TitleOpts(title="每笔交易净盈亏分布", pos_top="0%"),
                yaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
                datazoom_opts=[
                    opts.DataZoomOpts(
                        is_show=False,
                        type_="inside",
                        xaxis_index=[0, 0],
                        range_start=0,
                        range_end=100,
                    ),
                    opts.DataZoomOpts(
                        is_show=False,
                        type_="inside",
                        xaxis_index=[0, 1],
                        range_start=0,
                        range_end=100,
                    ),
                    opts.DataZoomOpts(
                        is_show=False,
                        type_="inside",
                        xaxis_index=[0, 2],
                        range_start=0,
                        range_end=100,
                    ),
                ],
                
                brush_opts = opts.BrushOpts(
                    tool_box = ["rect", "polygon", "keep","lineX","lineY", "clear"],
                    x_axis_index = "all",
                    brush_link = "all",
                    out_of_brush = {"colorAlpha": 0.1},
                    brush_type = "lineX",
                ),
                
                tooltip_opts = opts.TooltipOpts(
                    is_show = True,
                    trigger = "item",
                    axis_pointer_type = "shadow",
                    background_color = "rgba(245, 245, 245, 0.8)",
                    border_width = 1,
                    border_color = "#ccc",
                    textstyle_opts = opts.TextStyleOpts(color = "#000", font_size = 12, font_family = "Arial", font_weight = "lighter", ),
                ),

                toolbox_opts = opts.ToolboxOpts(orient = "horizontal", pos_left = "right", ),
                
                legend_opts = opts.LegendOpts(is_show = True, type_ = "scroll", selected_mode = "multiple", 
                                            pos_left = "40%", pos_top = "0%", legend_icon = "roundRect",),
            )
        )

        trade_daily_scatter = (
            EffectScatter()
            .add_xaxis(xaxis_data = list(trade_daily_df.index))
            .add_yaxis(
                series_name = "每日平仓净盈亏",
                y_axis = trade_daily_df.net_pnl.apply(lambda x: round(x, 2)).values.tolist(),
                symbol_size = 8,
                label_opts = opts.LabelOpts(is_show=False),
            ) 
            .set_global_opts(
                title_opts=opts.TitleOpts(title="每日平仓净盈亏分布", pos_top="33.5%"),
                yaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
                legend_opts = opts.LegendOpts(is_show = True, type_ = "scroll", selected_mode = "multiple", 
                                                pos_left = "48%", pos_top = "0%", legend_icon = "roundRect",),
            )
        )

        daily_scatter = (
            EffectScatter()
            .add_xaxis(xaxis_data = list(daily_df.index))
            .add_yaxis(
                series_name = "逐日盯市净盈亏",
                y_axis = daily_df.net_pnl.apply(lambda x: round(x, 2)).values.tolist(),
                symbol_size = 8,
                label_opts = opts.LabelOpts(is_show=False),
            ) 
            .set_global_opts(
                title_opts=opts.TitleOpts(title="逐日盯市净盈亏分布", pos_top="66%"),
                yaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
                legend_opts = opts.LegendOpts(is_show = True, type_ = "scroll", selected_mode = "multiple", 
                                                pos_left = "56%", pos_top = "0%", legend_icon = "roundRect",),
            )
        )

        grid_chart = (
            Grid(init_opts=opts.InitOpts(width="1900px", height="900px", chart_id="交易净盈亏分布"))
            .add(trade_scatter, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="4%", height="27%"))
            .add(trade_daily_scatter, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="37%", height="27%"))
            .add(daily_scatter, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="70%", height="27%"))
        )

        return grid_chart

    def statistics_chart(self, statistics = None):
        if statistics is None:
            s = self.statistics
        else:
            s = statistics
            
        capita_bar = (
            Bar()
            .add_xaxis(["起始资金", "结束资金", "交易总盈亏", "交易净盈亏", "交易手续费", "交易滑点费"])
            .add_yaxis(
                series_name="盈亏情况",
                y_axis=[int(s["capital"]), round(s["end_balance"],2), round(s["total_pnl"],2), round(s["total_net_pnl"],2), round(s["total_commission"],2), round(s["total_slippage"],2)],
                label_opts=opts.LabelOpts(font_size=15, font_weight="bolder"),
            )
            .set_global_opts(
                legend_opts = opts.LegendOpts(is_show = False),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=15, font_weight="bolder")),
                yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=12, font_weight="bolder"), splitline_opts=opts.SplitLineOpts(is_show=True),),
            )
        )

        return_bar = (
            Bar()
            .add_xaxis(["总收益率%", "年化收益率%", "年复合增长率%", "年化波动率%", "日均收益率%", "胜率%", "盈亏比"])
            .add_yaxis(
                series_name="收益率%",
                y_axis=[round(s["total_return"], 2), round(s["annual_return"], 2), round(s["cagr"], 2), round(s["annual_volatility"], 2), round(s["daily_return"], 2), round(s["rate_of_win"], 2), round(s["profit_loss_ratio"], 2)],
                label_opts=opts.LabelOpts(font_size=15, font_weight="bolder"),
            )
            .set_global_opts(
                legend_opts = opts.LegendOpts(is_show = False),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=12, font_weight="bolder", rotate=0)),
                yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=12, font_weight="bolder"), splitline_opts=opts.SplitLineOpts(is_show=True),),
            )
        )

        rate_bar = (
            Bar()
            .add_xaxis(["夏普比率", "Omega比率", "Calmar比率", "下限风险", "索提诺比率", "R平方", "尾部比率", "收益回撤比"])
            .add_yaxis(
                series_name="比率指标",
                y_axis=[round(s["sharpe_ratio"], 2), round(s["omega_ratio"], 2), round(s["calmar_ratio"], 2), round(s["downside_risk"], 2), round(s["sortino_ratio"], 2), round(s["R_squared"], 2), round(s["tail_ratio"], 2), round(s["return_drawdown_ratio"], 2)],
                label_opts=opts.LabelOpts(font_size=15, font_weight="bolder"),
            )
            .set_global_opts(
                legend_opts = opts.LegendOpts(is_show = False),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=12, font_weight="bolder", rotate=0)),
                yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=12, font_weight="bolder"), splitline_opts=opts.SplitLineOpts(is_show=True),),
            )
        )

        drawdown_bar = (
            Bar()
            .add_xaxis(["最大回撤金额(万)", "%最大回撤", "%平均回撤", "%线性加权回撤", "%均方回撤", "最长回撤天数"])
            .add_yaxis(
                series_name="回撤情况",
                y_axis=[round(s["max_drawdown"]/10000, 4), round(s["max_ddpercent"], 2), round(s["average_drawdown"], 2), round(s["lw_drawdown"], 2), round(s["average_square_drawdown"], 2), int(s["max_drawdown_duration"])],
                label_opts=opts.LabelOpts(font_size=15, font_weight="bolder"),
            )
            .set_global_opts(
                legend_opts = opts.LegendOpts(is_show = False),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=12, font_weight="bolder", rotate=-10, horizontal_align="center", margin=16)),
                yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=12, font_weight="bolder"), splitline_opts=opts.SplitLineOpts(is_show=True),),
            )
        )

        trade_bar = (
            Bar()
            .add_xaxis(["总交易笔数", "盈利交易笔数", "亏损交易笔数", "日均交易笔数", "平均持仓小时", "盈利持仓小时", "亏损持仓小时"])
            .add_yaxis(
                series_name="交易笔数",
                y_axis=[int(s["total_trade"]), int(s["profit_times"]), int(s["loss_times"]), round(s["daily_trade_count"],2), round(s["trade_duration"],2), round(s["profit_duration"],2), round(s["loss_duration"],2)],
                label_opts=opts.LabelOpts(font_size=15, font_weight="bolder"),
            )
            .set_global_opts(
                legend_opts = opts.LegendOpts(is_show = False),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=12, font_weight="bolder", rotate=-10, horizontal_align="center", margin=16)),
                yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=12, font_weight="bolder"), splitline_opts=opts.SplitLineOpts(is_show=True),),
            )
        )

        grid_chart = (
            Grid(init_opts=opts.InitOpts(width="1900px", height="900px", chart_id="统计指标"))
            .add(capita_bar, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="1%", height="25%"))
            .add(return_bar, grid_opts=opts.GridOpts(pos_left="5%", pos_right="50%", pos_top="35%", height="25%"))
            .add(rate_bar, grid_opts=opts.GridOpts(pos_left="55%", pos_right="5%", pos_top="35%", height="25%"))
            .add(drawdown_bar, grid_opts=opts.GridOpts(pos_left="5%", pos_right="50%", pos_top="70%", height="25%"))
            .add(trade_bar, grid_opts=opts.GridOpts(pos_left="55%", pos_right="5%", pos_top="70%", height="25%"))
        )

        return grid_chart

    # 每笔交易持仓时间分布
    def duration_grid_chart(self, trade_df = None):
        if trade_df is None:
            trade_df = self.trade_result_df
        
        trade_scatter = (
            EffectScatter()
            .add_xaxis(xaxis_data = trade_df.end_time.apply(lambda x:x.strftime("%Y-%m-%d %H:%M")).values.tolist())
            .add_yaxis(
                series_name = "总体持仓时间",
                y_axis = trade_df.duration.values.tolist(),
                symbol_size = 8,
                label_opts = opts.LabelOpts(is_show=False),
            ) 
            .set_global_opts(
                title_opts=opts.TitleOpts(title="每笔交易持仓时间分布", pos_top="0%"),
                yaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
                datazoom_opts=[
                    opts.DataZoomOpts(
                        is_show=False,
                        type_="inside",
                        xaxis_index=[0, 0],
                        range_start=0,
                        range_end=100,
                    ),
                    opts.DataZoomOpts(
                        is_show=False,
                        type_="inside",
                        xaxis_index=[0, 1],
                        range_start=0,
                        range_end=100,
                    ),
                    opts.DataZoomOpts(
                        is_show=False,
                        type_="inside",
                        xaxis_index=[0, 2],
                        range_start=0,
                        range_end=100,
                    ),
                ],
                
                brush_opts = opts.BrushOpts(
                    tool_box = ["rect", "polygon", "keep","lineX","lineY", "clear"],
                    x_axis_index = "all",
                    brush_link = "all",
                    out_of_brush = {"colorAlpha": 0.1},
                    brush_type = "lineX",
                ),
                
                tooltip_opts = opts.TooltipOpts(
                    is_show = True,
                    trigger = "item",
                    axis_pointer_type = "shadow",
                    background_color = "rgba(245, 245, 245, 0.8)",
                    border_width = 1,
                    border_color = "#ccc",
                    textstyle_opts = opts.TextStyleOpts(color = "#000", font_size = 12, font_family = "Arial", font_weight = "lighter", ),
                ),

                toolbox_opts = opts.ToolboxOpts(orient = "horizontal", pos_left = "right", ),
                
                legend_opts = opts.LegendOpts(is_show = True, type_ = "scroll", selected_mode = "multiple", 
                                            pos_left = "40%", pos_top = "0%", legend_icon = "roundRect",),
            )
        )

        profit_scatter = (
            EffectScatter()
            .add_xaxis(xaxis_data = list(trade_df[trade_df["net_pnl"] >= 0].end_time.apply(lambda x:x.strftime("%Y-%m-%d %H:%M")).values.tolist()))
            .add_yaxis(
                series_name = "盈利持仓小时",
                y_axis = trade_df[trade_df["net_pnl"] >= 0].duration.values.tolist(),
                symbol_size = 8,
                label_opts = opts.LabelOpts(is_show=False),
            ) 
            .set_global_opts(
                title_opts=opts.TitleOpts(title="盈利交易持仓时间分布", pos_top="33.5%"),
                yaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
                legend_opts = opts.LegendOpts(is_show = True, type_ = "scroll", selected_mode = "multiple", 
                                                pos_left = "48%", pos_top = "0%", legend_icon = "roundRect",),
            )
        )

        loss_scatter = (
            EffectScatter()
            .add_xaxis(xaxis_data = list(trade_df[trade_df["net_pnl"] < 0].end_time.apply(lambda x:x.strftime("%Y-%m-%d %H:%M")).values.tolist()))
            .add_yaxis(
                series_name = "亏损持仓小时",
                y_axis = trade_df[trade_df["net_pnl"] < 0].duration.values.tolist(),
                symbol_size = 8,
                label_opts = opts.LabelOpts(is_show=False),
            ) 
            .set_global_opts(
                title_opts=opts.TitleOpts(title="亏损交易持仓时间分布", pos_top="66%"),
                yaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
                legend_opts = opts.LegendOpts(is_show = True, type_ = "scroll", selected_mode = "multiple", 
                                                pos_left = "56%", pos_top = "0%", legend_icon = "roundRect",),
            )
        )

        grid_chart = (
            Grid(init_opts=opts.InitOpts(width="1900px", height="900px", chart_id="交易持仓时间分布"))
            .add(trade_scatter, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="4%", height="27%"))
            .add(profit_scatter, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="37%", height="27%"))
            .add(loss_scatter, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="70%", height="27%"))
        )

        return grid_chart

    def save_tab_chart(self):
        page_title = f"{self.symbol} {self.strategy_class.__name__}"
        tab_chart = Tab(page_title=page_title)
        tab_chart.add(self.statistics_chart(), "各类统计指标")
        tab_chart.add(self.daily_grid_chart(), "日收益分析图")
        tab_chart.add(self.trade_grid_chart(), "交易盈亏分布图")
        tab_chart.add(self.duration_grid_chart(), "持仓时间分布图")
        tab_chart.add(self.draw_kline_chart(), "K线 + 成交记录 + 净值曲线")
        # tab_chart.add(self.draw_grid_chart(), "K线 + 成交记录 + 技术指标")

        # 生成文件保存路径
        home_path = Path.home()
        temp_name = "Desktop"
        temp_path = home_path.joinpath(temp_name)
        filename  = f"{self.symbol}{self.strategy_class.__name__}[{self.trade_date_range}][{self.main_parameters}].html"
        filepath  = temp_path.joinpath(filename)

        tab_chart.render(filepath)

# =================================================================================================================================================================================================================


class DailyResult:
    """"""

    def __init__(self, date: date, close_price: float):
        """"""
        self.date = date
        self.close_price = close_price
        self.pre_close = 0

        self.trades = []
        self.trade_count = 0

        self.start_pos = 0
        self.end_pos = 0

        self.turnover = 0
        self.commission = 0
        self.slippage = 0

        self.trading_pnl = 0
        self.holding_pnl = 0
        self.total_pnl = 0
        self.net_pnl = 0

    def add_trade(self, trade: TradeData):
        """"""
        self.trades.append(trade)

    def calculate_pnl(
        self,
        pre_close: float,
        start_pos: float,
        size: int,
        rate_type: RateType,
        rate: float,
        slippage: float,
        inverse: bool
    ):
        """"""
        # If no pre_close provided on the first day,
        # use value 1 to avoid zero division error
        if pre_close:
            self.pre_close = pre_close
        else:
            self.pre_close = 1

        # Holding pnl is the pnl from holding position at day start
        self.start_pos = start_pos
        self.end_pos = start_pos

        if not inverse:     # For normal contract
            self.holding_pnl = self.start_pos * \
                (self.close_price - self.pre_close) * size
        else:               # For crypto currency inverse contract
            self.holding_pnl = self.start_pos * \
                (1 / self.pre_close - 1 / self.close_price) * size

        # Trading pnl is the pnl from new trade during the day
        self.trade_count = len(self.trades)

        for trade in self.trades:
            if trade.direction == Direction.LONG:
                pos_change = trade.volume
            else:
                pos_change = -trade.volume

            self.end_pos += pos_change

            # For normal contract
            if not inverse:
                turnover = trade.volume * size * trade.price
                self.trading_pnl += pos_change * (self.close_price - trade.price) * size
                if rate_type == RateType.FIXED:
                    self.commission += trade.volume * rate
                self.slippage += trade.volume * size * slippage
            # For crypto currency inverse contract
            else:
                turnover = trade.volume * size / trade.price
                self.trading_pnl += pos_change * (1 / trade.price - 1 / self.close_price) * size
                if rate_type == RateType.FIXED:
                    self.commission += trade.volume * rate
                self.slippage += trade.volume * size * slippage / (trade.price ** 2)

            self.turnover += turnover
            if rate_type == RateType.FLOAT:
                self.commission += turnover * rate

        # Net pnl takes account of commission and slippage cost
        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission - self.slippage


def optimize(
    target_name: str,
    strategy_class: CtaTemplate,
    setting: dict,
    vt_symbol: str,
    interval: Interval,
    start: datetime,
    rate_type: RateType,
    rate: float,
    slippage: float,
    size: float,
    pricetick: float,
    capital: int,
    end: datetime,
    mode: BacktestingMode,
    inverse: bool
):
    """
    Function for running in multiprocessing.pool
    """
    engine = BacktestingEngine()

    engine.set_parameters(
        vt_symbol=vt_symbol,
        interval=interval,
        start=start,
        rate_type=rate_type,
        rate=rate,
        slippage=slippage,
        size=size,
        pricetick=pricetick,
        capital=capital,
        end=end,
        mode=mode,
        inverse=inverse
    )

    engine.add_strategy(strategy_class, setting)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    statistics = engine.calculate_statistics(output=False)

    target_value = statistics[target_name]
    return (str(setting), target_value, statistics)


@lru_cache(maxsize=1000000)
def _ga_optimize(parameter_values: tuple):
    """"""
    setting = dict(parameter_values)

    result = optimize(
        ga_target_name,
        ga_strategy_class,
        setting,
        ga_vt_symbol,
        ga_interval,
        ga_start,
        ga_rate_type,
        ga_rate,
        ga_slippage,
        ga_size,
        ga_pricetick,
        ga_capital,
        ga_end,
        ga_mode,
        ga_inverse
    )
    return (result[1],)


def ga_optimize(parameter_values: list):
    """"""
    return _ga_optimize(tuple(parameter_values))


@lru_cache(maxsize=999)
def load_bar_data(
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    start: datetime,
    end: datetime
):
    """bar数据缓存为pkl格式到本地硬盘"""
    dir_path = f"C:\\Users\\Administrator\\Desktop\\Pickle_Data\\"
    file_name = f'{symbol}_{exchange.value}_{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}_bar'
    pickle_path = dir_path + file_name + ".pkl"
    data_size = 0

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if not os.path.exists(pickle_path):   
        bar_data = database_manager.load_bar_data(symbol, exchange, interval, start, end)
        pickle_file = open(pickle_path, 'wb')    
        pickle.dump(bar_data, pickle_file)
        pickle_file.close()

    else:
        pickle_file = open(pickle_path,'rb')
        bar_data =pickle.load(pickle_file)
        pickle_file.close()

    # Pickle_Data文件夹超过规定GB容量则清空缓存数据
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for file_name in filenames:         #当前目录所有文件名
            data_size += os.path.getsize(dirpath + file_name)

    if data_size / (1024 ** 3) > 20:
        for dirpath, dirnames, filenames in os.walk(dir_path):
            for file_name in filenames:           
                os.remove(dirpath + file_name)

    return bar_data


@lru_cache(maxsize=999)
def load_tick_data(
    symbol: str,
    exchange: Exchange,
    start: datetime,
    end: datetime
):
    """tick数据缓存为pkl格式到本地硬盘"""
    dir_path = f"C:\\Users\\Administrator\\Desktop\\Pickle_Data\\"
    file_name = f'{symbol}_{exchange.value}_{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}_tick'
    pickle_path = dir_path + file_name + ".pkl"
    data_size = 0

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if not os.path.exists(pickle_path):        
        tick_data = database_manager.load_tick_data( symbol, exchange, start, end )
        pickle_file = open(pickle_path,'wb')    
        pickle.dump(tick_data,pickle_file)
        pickle_file.close()

    else:        
        pickle_file = open(pickle_path,'rb')
        tick_data =pickle.load(pickle_file)
        pickle_file.close()

    # Pickle_Data文件夹超过规定GB容量则清空缓存数据
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for file_name in filenames:         #当前目录所有文件名
            data_size += os.path.getsize(dirpath + file_name)

    if data_size / (1024 ** 3) > 20:
        for dirpath, dirnames, filenames in os.walk(dir_path):
            for file_name in filenames:           
                os.remove(dirpath + file_name)    

    return tick_data


# GA related global value
ga_end = None
ga_mode = None
ga_target_name = None
ga_strategy_class = None
ga_setting = None
ga_vt_symbol = None
ga_interval = None
ga_start = None
ga_rate_type = None
ga_rate = None
ga_slippage = None
ga_size = None
ga_pricetick = None
ga_capital = None
