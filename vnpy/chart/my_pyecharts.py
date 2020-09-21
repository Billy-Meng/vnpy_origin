# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import talib as ta

from pyecharts.commons.utils import JsCode
from pyecharts import options as opts
from pyecharts.globals import ThemeType, SymbolType
from pyecharts.charts import Bar, Kline, Line, Grid, EffectScatter, Tab

# 忽略警告信息
import warnings
warnings.filterwarnings("ignore")

class MyPyecharts():

    kline_chart = None
    grid_chart = None
    
    def __init__(self, bar_data, trade_data, grid=True, grid_quantity=0, chart_id=20):
        self.bar_data = bar_data
        self.trade_data = trade_data
        self.grid = grid
        self.grid_quantity = grid_quantity
        self.chart_id = chart_id
        self.bar_data_datetime = list(self.bar_data.index.strftime("%Y-%m-%d %H:%M"))

    def kline(self):
        """"""
        kline = Kline(init_opts=opts.InitOpts(width="1400px", height="800px"))
        kline.add_xaxis(xaxis_data=self.bar_data_datetime)
        kline.add_yaxis(
            series_name=f"{self.bar_data.symbol[0]}",
            yaxis_index = 0,
            y_axis=self.bar_data[["open_price", "close_price", "low_price", "high_price"]].values.tolist(),
            itemstyle_opts=opts.ItemStyleOpts(color="#ec0000", color0="#00da3c", border_color="#8A0000", border_color0="#008F28", opacity=0.8),
        )

        if self.grid == True and self.grid_quantity == 1:
            kline.set_global_opts(
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

                # 添加主标题和副标题
                # title_opts = opts.TitleOpts(title = "主标题", subtitle = "  副标题"),  

                # 多图的 axis 连在一块
                axispointer_opts = opts.AxisPointerOpts(
                    is_show = True,
                    link=[{"xAxisIndex": "all"}],
                    label=opts.LabelOpts(background_color="#777"),
                ),
            )

        elif self.grid == True and self.grid_quantity == 2:
            kline.set_global_opts(
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
                        is_show=True,
                        type_="slider",
                        xaxis_index=[0, 1],
                        pos_top="95%",
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

                # 添加主标题和副标题
                # title_opts = opts.TitleOpts(title = "主标题", subtitle = "  副标题"),  

                # 多图的 axis 连在一块
                axispointer_opts = opts.AxisPointerOpts(
                    is_show = True,
                    link=[{"xAxisIndex": "all"}],
                    label=opts.LabelOpts(background_color="#777"),
                ),
            )

        else:
            kline.set_global_opts(
                # 单图形
                datazoom_opts = opts.DataZoomOpts(
                    is_show=False,
                    type_="inside",
                    range_start=0,
                    range_end=100,        
                ),
                    
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

                # 添加主标题和副标题
                # title_opts = opts.TitleOpts(title = "主标题", subtitle = "  副标题"),  

                # 多图的 axis 连在一块
                axispointer_opts = opts.AxisPointerOpts(
                    is_show = True,
                    link=[{"xAxisIndex": "all"}],
                    label=opts.LabelOpts(background_color="#777"),
                ),
            )

        self.kline_chart = kline

    def overlap_trade(self):
        # 叠加成交开平仓点标记至K线图

        # Buy，深红色向上箭头
        trade_buy = (
            EffectScatter()
            .add_xaxis(xaxis_data = list(self.trade_data[(self.trade_data.direction == "多") & (self.trade_data.offset == "开")].index.strftime("%Y-%m-%d %H:%M")))
            .add_yaxis(
                series_name = "Buy", 
                y_axis = self.trade_data[(self.trade_data.direction == "多") & (self.trade_data.offset == "开")].trade_price.values.tolist(),
                symbol = SymbolType.ARROW,
                symbol_size = 12,
                symbol_rotate = 0,
                label_opts = opts.LabelOpts(is_show=False),
                itemstyle_opts = opts.ItemStyleOpts(color="#8A0000", opacity=0.8),
            )
            .set_global_opts(legend_opts = opts.LegendOpts(is_show = False))
        )

        # Sell，深红色向下箭头
        trade_sell = (
            EffectScatter()
            .add_xaxis(xaxis_data = list(self.trade_data[(self.trade_data.direction == "空") & (self.trade_data.offset == "平")].index.strftime("%Y-%m-%d %H:%M")))
            .add_yaxis(
                series_name = "Sell", 
                y_axis = self.trade_data[(self.trade_data.direction == "空") & (self.trade_data.offset == "平")].trade_price.values.tolist(),
                symbol = SymbolType.ARROW,
                symbol_size = 12,
                symbol_rotate = 180,
                label_opts = opts.LabelOpts(is_show=False),
                itemstyle_opts = opts.ItemStyleOpts(color="#8A0000", opacity=0.8),
            )
            .set_global_opts(legend_opts = opts.LegendOpts(is_show = False))
        )

        # Short，深绿色向下箭头
        trade_short = (
            EffectScatter()
            .add_xaxis(xaxis_data = list(self.trade_data[(self.trade_data.direction == "空") & (self.trade_data.offset == "开")].index.strftime("%Y-%m-%d %H:%M")))
            .add_yaxis(
                series_name = "Short", 
                y_axis = self.trade_data[(self.trade_data.direction == "空") & (self.trade_data.offset == "开")].trade_price.values.tolist(),
                symbol = SymbolType.ARROW,
                symbol_size = 12,
                symbol_rotate = 180,
                label_opts = opts.LabelOpts(is_show=False),
                itemstyle_opts = opts.ItemStyleOpts(color="#008F28", opacity=0.8),
            )
            .set_global_opts(legend_opts = opts.LegendOpts(is_show = False))
        )

        # Cover，深绿色向上箭头
        trade_cover = (
            EffectScatter()
            .add_xaxis(xaxis_data = list(self.trade_data[(self.trade_data.direction == "多") & (self.trade_data.offset == "平")].index.strftime("%Y-%m-%d %H:%M")))
            .add_yaxis(
                series_name = "Cover", 
                y_axis = self.trade_data[(self.trade_data.direction == "多") & (self.trade_data.offset == "平")].trade_price.values.tolist(),
                symbol = SymbolType.ARROW,
                symbol_size = 12,
                symbol_rotate = 0,
                label_opts = opts.LabelOpts(is_show=False),
                itemstyle_opts = opts.ItemStyleOpts(color="#008F28", opacity=0.8),
            )
            .set_global_opts(legend_opts = opts.LegendOpts(is_show = False))
        )

        self.kline_chart.overlap(trade_buy)
        self.kline_chart.overlap(trade_sell)
        self.kline_chart.overlap(trade_short)
        self.kline_chart.overlap(trade_cover)

    def overlap_net_pnl(self):
        # 在K线图绘制累计净盈亏曲线(注：kline的datazoom_opts为单图形模式)

        if self.grid == True:
            print("grid应为False！")
            return 

        bar_data = pd.merge(self.bar_data, self.trade_data.net_pnl.cumsum(), how="outer", left_index=True, right_index=True)
        bar_data["net_pnl"].fillna(method="ffill", inplace=True)
        bar_data["net_pnl"].fillna(value=0, inplace=True)

        self.kline_chart.extend_axis(yaxis = opts.AxisOpts(position="right"))

        net_pnl_line = (
            Line()
            .add_xaxis(xaxis_data = self.bar_data_datetime)
            .add_yaxis(
                series_name = "Net_pnl",
                y_axis = bar_data.net_pnl.apply(lambda x: round(x, 2)).values.tolist(),
                yaxis_index = 1,
                label_opts = opts.LabelOpts(is_show=False),
                is_symbol_show = False,
                linestyle_opts = opts.LineStyleOpts(width=3, opacity=0.5, type_="solid", color="#8A0000"),
            ) 
        )

        signal_scatter = (
            EffectScatter()
            .add_xaxis(xaxis_data = list(self.trade_data.index.strftime("%Y-%m-%d %H:%M")))
            .add_yaxis(
                series_name = "Signal",
                y_axis = self.trade_data.signal.values.tolist(),
                yaxis_index = 1,
                symbol_size = 5,
                label_opts = opts.LabelOpts(is_show=True, font_size=9),
                itemstyle_opts = opts.ItemStyleOpts(color="#0000FF", opacity=0.9),
            ) 
            .set_global_opts(legend_opts = opts.LegendOpts(is_show = False))
        )

        self.kline_chart.overlap(signal_scatter)
        self.kline_chart.overlap(net_pnl_line)

    def overlap_balance(self, capital=100000):
        # 在K线图绘制收益曲线(注：kline的datazoom_opts为单图形模式)

        if self.grid == True:
            print("grid应为False！")
            return 

        bar_data = pd.merge(self.bar_data, self.trade_data.balance, how="outer", left_index=True, right_index=True)
        bar_data["balance"].fillna(method="ffill", inplace=True)
        bar_data["balance"].fillna(value=capital, inplace=True)

        self.kline_chart.extend_axis(yaxis = opts.AxisOpts(position="right"))

        balance_line = (
            Line()
            .add_xaxis(xaxis_data = self.bar_data_datetime)
            .add_yaxis(
                series_name = "Balance",
                y_axis = bar_data.balance.apply(lambda x: round(x, 2)).values.tolist(),
                yaxis_index = 1,
                label_opts = opts.LabelOpts(is_show=False),
                is_symbol_show = False,
                linestyle_opts = opts.LineStyleOpts(width=3, opacity=0.5, type_="solid", color="#8A0000"),
            ) 
        )

        self.kline_chart.overlap(balance_line)

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

    def overlap_ema(self, ema_series:list = [5, 10, 20, 60]):
        # 在K线图上绘制 SMA 均线
        EMA_series = ema_series

        EMA_line = (
            Line()
            .add_xaxis(xaxis_data=self.bar_data_datetime)
            .set_global_opts(xaxis_opts=opts.AxisOpts(type_="category"))
        )

        for i in EMA_series:
            ema_value = ta.EMA(np.array(self.bar_data["close_price"]), i)
            EMA_line.add_yaxis(
                series_name = f"EMA_{i}",
                y_axis = ema_value.tolist(),
                is_symbol_show = False,
                is_smooth = True,
                is_hover_animation = False,
                linestyle_opts = opts.LineStyleOpts(width=2, opacity=0.8),
                label_opts = opts.LabelOpts(is_show=False),
                tooltip_opts = opts.TooltipOpts(is_show=False),
            )

        self.kline_chart.overlap(EMA_line)

    def overlap_boll(self, timeperiod=14, nbdevup=2, nbdevdn=2, matype=0):
        # 在K线图上绘制 布林带

        boll_up, boll_mid, boll_low = ta.BBANDS(np.array(self.bar_data["close_price"]),\
                                                timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)

        BOLL_line = (
            Line()
            .add_xaxis(xaxis_data=self.bar_data_datetime) 
            .add_yaxis(
                series_name = "boll_up",
                y_axis = boll_up.tolist(),
                is_symbol_show = False,
                is_smooth = True,
                is_hover_animation = False,
                linestyle_opts = opts.LineStyleOpts(width=2, opacity=0.8),
                label_opts = opts.LabelOpts(is_show=False),
                tooltip_opts = opts.TooltipOpts(is_show=False),
            )
            .add_yaxis(
                series_name = "boll_mid",
                y_axis = boll_mid.tolist(),
                is_symbol_show = False,
                is_smooth = True,
                is_hover_animation = False,
                linestyle_opts = opts.LineStyleOpts(width=2, opacity=0.8),
                label_opts = opts.LabelOpts(is_show=False),
                tooltip_opts = opts.TooltipOpts(is_show=False),
            )
            .add_yaxis(
                series_name = "boll_low",
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

    def grid_macd(self, fastperiod=12, slowperiod=26, signalperiod=9, grid_index=1):
        # 在K线图上绘制 MACD

        macd_dif, macd_dea, macd_bar = ta.MACD(np.array(self.bar_data["close_price"]), \
                                               fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        # 国内MACD
        macd_bar = macd_bar * 2

        # MACD快慢线
        macd_line = (
            Line()
            .add_xaxis(xaxis_data=self.bar_data_datetime)
            .add_yaxis(
                series_name="DIF",
                y_axis=macd_dif.tolist(),
                xaxis_index=grid_index,
                yaxis_index=grid_index,
                is_symbol_show = False,
                is_smooth = True,
                is_hover_animation = False,
                linestyle_opts = opts.LineStyleOpts(width=2, opacity=0.8),
                label_opts = opts.LabelOpts(is_show=False),
                tooltip_opts = opts.TooltipOpts(is_show=False),
            )
            .add_yaxis(
                series_name="DEA",
                y_axis=macd_dea.tolist(),
                xaxis_index=grid_index,
                yaxis_index=grid_index,
                is_symbol_show = False,
                is_smooth = True,
                is_hover_animation = False,
                linestyle_opts = opts.LineStyleOpts(width=2, opacity=0.8),
                label_opts = opts.LabelOpts(is_show=False),
                tooltip_opts = opts.TooltipOpts(is_show=False),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    grid_index=grid_index,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                yaxis_opts=opts.AxisOpts(
                    grid_index=grid_index,
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    axislabel_opts=opts.LabelOpts(is_show=True),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )

        # MACD柱状线
        macd_bar = (
            Bar()
            .add_xaxis(xaxis_data = self.bar_data_datetime)
            .add_yaxis(
                series_name = "MACD",
                y_axis = macd_bar.tolist(),
                xaxis_index = grid_index,
                yaxis_index = grid_index,
                label_opts = opts.LabelOpts(is_show=False),
                tooltip_opts = opts.TooltipOpts(is_show=False),
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
                    opacity=0.6,
                ),
            )
            .set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
        )

        # 层叠MACD快慢线与柱状线
        overlap_macd = macd_line.overlap(macd_bar)

        return overlap_macd



    def grid_graph(self, grid_graph_1 = None, grid_graph_2 = None):
        # 创建组合图表画布对象
        self.grid_chart = Grid(init_opts=opts.InitOpts(width="1900px", height="900px", chart_id=self.chart_id))

        if self.grid_quantity == 0:
            self.grid_chart.add(
                chart = self.kline_chart,
                grid_opts = opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="10%", height="85%"),
                is_control_axis_index = True
            )
            return self.grid_chart

        elif self.grid_quantity == 1:
            self.grid_chart.add(
                chart = self.kline_chart,
                grid_index = 0,
                grid_opts = opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="10%", height="60%"),
            )
            self.grid_chart.add(
                chart = grid_graph_1,
                grid_index = 1,
                grid_opts = opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="75%", height="20%"),
            )
            return self.grid_chart

        elif self.grid_quantity == 2:
            self.grid_chart.add(
                chart = self.kline_chart,
                grid_index = 0,
                grid_opts = opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="10%", height="40%"),
            )
            self.grid_chart.add(
                chart = grid_graph_1,
                grid_index = 1,
                grid_opts = opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="55%", height="17.5%"),
            )
            self.grid_chart.add(
                chart = grid_graph_2,
                grid_index = 2,
                grid_opts = opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="77.5%", height="17.5%"),
            )
            return self.grid_chart


    def render(self, html_name="render.html"):
        return self.grid_chart.render(html_name)

