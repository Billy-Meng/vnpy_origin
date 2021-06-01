from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pymongo
import PySimpleGUI as sg
import talib as ta

from pyecharts.commons.utils import JsCode
from pyecharts import options as opts
from pyecharts.charts import Bar, Kline, Line, Grid

# 忽略警告信息
import warnings
warnings.filterwarnings("ignore")


def main():
    # 连接MongoDB客户端
    client = pymongo.MongoClient("mongodb://localhost:27017/")

    coll = client["data"]["db_bar_data"]

    symbol_set = set()
    for d in coll.find({}, {"symbol":1, "_id":0}):
        symbol_set.add(d["symbol"])
    
    symbol_list = sorted(list(symbol_set))

    start = (datetime.now() - timedelta(days=30)).date().strftime("%Y-%m-%d") + " 00:00"
    end = datetime.now().strftime("%Y-%m-%d %H:%M")

    layout = [
        [sg.Text("套利合约"), sg.Combo(symbol_list, default_value=symbol_list[0], size=(30, 1), key="symbol")],
        [sg.Text("起始时间"), sg.Input(start, size=(30, 1), key="start_dt", tooltip="时间格式：2020-12-31 12:30")],
        [sg.Text("结束时间"), sg.Input(end, size=(30, 1), key="end_dt", tooltip="时间格式：2020-12-31 12:30")],
        [sg.Text("时间周期"), sg.Input(1, size=(30, 1), key="period")],
        [sg.Text("周期类型"), sg.Combo(["M", "H", "D"], default_value="D", size=(30, 1), key="interval", tooltip="M:分, H:时, D:日")],
        [sg.Text("布林周期"), sg.Input(30, size=(30, 1), key="boll_win")],
        [sg.Text("布林偏差"), sg.Input(2, size=(30, 1), key="boll_dev")],
        [sg.Button("生成", key="generate")]
    ]  

    window = sg.Window("套利合约K线生成").Layout(layout)

    while True:
        event, values = window.Read()
        if event is None:       # 关闭窗体
            break

        elif event == "generate":
            symbol = values["symbol"]
            start_dt = datetime.strptime(values["start_dt"], "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(values["end_dt"], "%Y-%m-%d %H:%M")
            period = int(values["period"])
            interval = values["interval"]
            boll_win = int(values["boll_win"])
            boll_dev = int(values["boll_dev"])

            find_cond = {"symbol": symbol, "datetime": {"$gte": start_dt, "$lte": end_dt}}

            bar_list = []
            for d in coll.find(find_cond, {"_id":0}):
                bar_list.append(d)

            new_bar_list = generate_bar(bar_list, window=period, interval=interval)

            df = pd.DataFrame(new_bar_list)
            df.set_index("datetime", drop=True, inplace=True)
            df.sort_index(ascending=True, inplace=True)

            # 绘制蜡烛图，叠加主、副图技术指标
            grid_chart = MyPyecharts(bar_data=df)
            grid_chart.kline()
            # grid_chart.overlap_sma([5, 10, 20, 60])
            grid_chart.overlap_boll(timeperiod=boll_win, nbdevup=boll_dev, nbdevdn=boll_dev, matype=0)
            chart = grid_chart.grid_graph(grid_graph = grid_chart.grid_volume(grid_index=1))
            chart.render(f'【{symbol}】【{period}{interval}】{str(start_dt)[:-3].replace(":", "-")} ~ {str(end_dt)[:-3].replace(":", "-")}.html')

    window.Close()

def generate_bar(bar_list, window=10, interval="M"):
    """ 通过一分钟的Bar，生成各级别周期Bar """
    bar_data = []
    last_bar = None
    window_bar = None
    interval_count = 0
    for bar in bar_list:
        if not window_bar:
            window_bar = {
                "symbol": bar["symbol"],
                "datetime": bar["datetime"],
                "open_price": bar["open_price"],
                "high_price": bar["high_price"],
                "low_price": bar["low_price"],
                "close_price": bar["close_price"],
                "volume": bar["volume"],
                "open_interest": bar["open_interest"],
            }

        else:
            window_bar["high_price"] = max(
                window_bar["high_price"], bar["high_price"])
            window_bar["low_price"] = min(
                window_bar["low_price"], bar["low_price"])

        window_bar["close_price"] = bar["close_price"]
        window_bar["volume"] += int(bar["volume"])
        window_bar["open_interest"] = bar["open_interest"]

        # Check if window bar completed
        finished = False

        # X分钟K线合成
        if interval.upper() in ["M", "分钟"]:
            # 整除切分法进行分钟K线合成，合成 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60分钟K线
            if window not in [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]:
                print("整除法合成N分钟K线，时间窗口须为 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60 其中之一！")
                return
            else:
                if not (bar["datetime"].minute + 1) % window:
                    finished = True
            
        # X小时K线合成，计数切分法进行N小时K线合成，可以合成任意小时K线
        elif interval.upper() in ["H", "小时"]:
            if last_bar and bar["datetime"].hour != last_bar["datetime"].hour:
                # 1-hour bar
                if window == 1:
                    finished = True
                # x-hour bar
                else:
                    interval_count += 1

                    if not interval_count % window:
                        finished = True
                        interval_count = 0

        # 日K线合成
        elif interval.upper() in ["D", "日"]:
            if last_bar and bar["datetime"].date() != last_bar["datetime"].date():
                finished = True

        if finished:
            bar_data.append(window_bar)
            window_bar = None

        # Cache last bar object
        last_bar = bar

    return bar_data

class MyPyecharts():

    kline_chart = None
    grid_chart = None
    
    def __init__(self, bar_data):
        self.bar_data = bar_data
        self.bar_data_datetime = list(self.bar_data.index.strftime("%Y-%m-%d %H:%M"))

    def kline(self):
        """"""
        if "_" in self.bar_data.symbol[0]:
            symbol_split = self.bar_data.symbol[0].split("_")
            series_name = "_".join([symbol_split[0], symbol_split[1][-2:], symbol_split[2][-2:]])
        else:
            series_name = self.bar_data.symbol[0]
        
        kline = Kline(init_opts=opts.InitOpts(width="1400px", height="800px"))
        kline.add_xaxis(xaxis_data=self.bar_data_datetime)
        kline.add_yaxis(
            series_name=series_name,
            yaxis_index = 0,
            y_axis=self.bar_data[["open_price", "close_price", "low_price", "high_price"]].values.tolist(),
            itemstyle_opts=opts.ItemStyleOpts(color="#ef232a", color0="#14b143", border_color="#8A0000", border_color0="#008F28", opacity=0.8),
        )

        kline.set_global_opts(
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_show=False,
                    type_="inside",
                    xaxis_index=[0, 0],
                    range_start=0,
                    range_end=100,
                ),
                opts.DataZoomOpts(
                    is_show=True,
                    type_="slider",
                    xaxis_index=[0, 1],
                    pos_top="95%",
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
                axis_pointer_type = "cross",
                background_color = "rgba(245, 245, 245, 0.8)",
                border_width = 1,
                border_color = "#ccc",
                textstyle_opts = opts.TextStyleOpts(color = "#000", font_size = 12, font_family = "Arial", font_weight = "lighter", ),
            ),

            toolbox_opts = opts.ToolboxOpts(orient = "horizontal", pos_left = "right", pos_top = "0%"),
            
            legend_opts = opts.LegendOpts(is_show = True, type_ = "scroll", selected_mode = "multiple", 
                                        pos_left = "left", pos_top = "0%", legend_icon = "roundRect",),

            # 多图的 axis 连在一块
            axispointer_opts = opts.AxisPointerOpts(
                is_show = True,
                link=[{"xAxisIndex": "all"}],
                label=opts.LabelOpts(background_color="#777"),
            ),
        )

        self.kline_chart = kline

    def overlap_sma(self, sma_series:list = [5, 10, 20, 60]):
        # 在K线图上绘制 SMA 均线
        SMA_series = sma_series

        SMA_line = (
            Line()
            .add_xaxis(xaxis_data=self.bar_data_datetime)
            .set_global_opts(xaxis_opts=opts.AxisOpts(type_="category"))
        )

        for i in SMA_series:
            sma_value = ta.SMA(np.array(self.bar_data["close_price"]), i)
            SMA_line.add_yaxis(
                series_name = f"SMA_{i}",
                y_axis = sma_value.tolist(),
                is_symbol_show = False,
                is_smooth = True,
                is_hover_animation = False,
                linestyle_opts = opts.LineStyleOpts(width=2, opacity=0.8),
                label_opts = opts.LabelOpts(is_show=False),
                tooltip_opts = opts.TooltipOpts(is_show=False),
            )

        self.kline_chart.overlap(SMA_line)

    def overlap_boll(self, timeperiod=14, nbdevup=2, nbdevdn=2, matype=0):
        # 在K线图上绘制 布林带

        boll_up, boll_mid, boll_low = ta.BBANDS(np.array(self.bar_data["close_price"]),\
                                                timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)

        BOLL_line = (
            Line()
            .add_xaxis(xaxis_data=self.bar_data_datetime) 
            .add_yaxis(
                series_name = "布林上轨",
                y_axis = boll_up.tolist(),
                is_symbol_show = False,
                is_smooth = True,
                is_hover_animation = False,
                linestyle_opts = opts.LineStyleOpts(width=2, opacity=0.8),
                label_opts = opts.LabelOpts(is_show=False),
                tooltip_opts = opts.TooltipOpts(is_show=False),
            )
            .add_yaxis(
                series_name = "布林中轨",
                y_axis = boll_mid.tolist(),
                is_symbol_show = False,
                is_smooth = True,
                is_hover_animation = False,
                linestyle_opts = opts.LineStyleOpts(width=2, opacity=0.8),
                label_opts = opts.LabelOpts(is_show=False),
                tooltip_opts = opts.TooltipOpts(is_show=False),
            )
            .add_yaxis(
                series_name = "布林下轨",
                y_axis = boll_low.tolist(),
                is_symbol_show = False,
                is_smooth = True,
                is_hover_animation = False,
                linestyle_opts = opts.LineStyleOpts(width=2, opacity=0.8),
                label_opts = opts.LabelOpts(is_show=False),
                tooltip_opts = opts.TooltipOpts(is_show=False),
            )
        )

        self.kline_chart.overlap(BOLL_line)

    def grid_volume(self, grid_index=1):
        """"""
        vol_bar = (
            Bar()
            .add_xaxis(xaxis_data = self.bar_data_datetime)
            .add_yaxis(
                series_name = "成交量",
                y_axis = self.bar_data["volume"].tolist(),
                xaxis_index=grid_index,
                yaxis_index=grid_index,
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(
                    opacity=0.8,
                    color=JsCode(
                        """
                        function(params) {
                            var colorList;
                            if (barData[params.dataIndex][1] >= barData[params.dataIndex][0]) {
                                colorList = '#ef232a';
                            } else {
                                colorList = '#14b143';
                            }
                            return colorList;
                        }
                        """
                    )
                ),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    grid_index=grid_index,
                ),
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )

        return vol_bar

    def grid_graph(self, grid_graph = None):
        # 创建组合图表画布对象
        self.grid_chart = Grid(init_opts=opts.InitOpts(width="1900px", height="900px"))

        bar_data = [[row["open_price"], row["close_price"], row["low_price"], row["high_price"]] for ix, row in self.bar_data.iterrows()]
        self.grid_chart.add_js_funcs("var barData = {}".format(bar_data))
    
        self.grid_chart.add(
            chart = self.kline_chart,
            grid_index = 0,
            grid_opts = opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="10%", height="60%"),
        )
        self.grid_chart.add(
            chart = grid_graph,
            grid_index = 1,
            grid_opts = opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="75%", height="20%"),
        )

        return self.grid_chart

    def render(self, html_name="render.html"):
        return self.grid_chart.render(html_name)

if __name__ == '__main__':
    main()
