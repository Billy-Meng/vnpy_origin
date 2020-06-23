# encoding: UTF-8

import copy
import csv
import decimal
import math
import os
import sys
import traceback
import talib as ta
import numpy as np

from collections import OrderedDict
from datetime import datetime, timedelta
from pykalman import KalmanFilter

from vnpy.trader.object import RenkoBarData
from vnpy.trader.utility import round_to
from vnpy.trader.constant import Direction, Color
from vnpy.component.cta_period import CtaPeriod, Period


class CtaRenkoBar(object):
    """CTA 砖型K线"""

    CB_ON_BAR = 'cb_on_bar'
    CB_FAST_RETURN = 'cb_fast_return'
    CB_ON_PERIOD = 'cb_on_period'

    # 参数列表，保存了参数的名称
    param_list = ['symbol']

    def __init__(self, strategy, cb_on_bar, setting=None):

        self.cb_on_bar = cb_on_bar

        # 参数列表
        self.param_list.append('para_pre_len')
        self.param_list.append('para_ma1_len')
        self.param_list.append('para_ma2_len')
        self.param_list.append('para_ma3_len')
        self.param_list.append('para_ema1_len')
        self.param_list.append('para_ema2_len')
        self.param_list.append('para_ema3_len')
        self.param_list.append('para_ama_len')
        self.param_list.append('para_rsi1_len')
        self.param_list.append('para_rsi2_len')

        self.param_list.append('para_cci_len')

        self.param_list.append('para_dmi_len')
        self.param_list.append('para_dmi_max')
        self.param_list.append('para_atr1_len')
        self.param_list.append('para_atr2_len')
        self.param_list.append('para_atr3_len')
        self.param_list.append('para_boll_len')
        self.param_list.append('para_boll_std_rate')
        self.param_list.append('para_boll2_len')
        self.param_list.append('para_boll2_std_rate')
        self.param_list.append('height')
        self.param_list.append('kilo_height')

        self.param_list.append('price_tick')
        self.param_list.append('underlying_symbol')
        self.param_list.append('activate_ma_tick')
        self.param_list.append('activate_kf_tick')
        self.param_list.append('avg_price_len')
        self.param_list.append('activate_period')

        self.param_list.append('name')
        self.param_list.append('para_cmi_len')
        self.param_list.append('para_kdj_len')
        self.param_list.append('para_kdj_tb_len')
        self.param_list.append('para_kdj_slow_len')
        self.param_list.append('para_kdj_smooth_len')

        self.param_list.append('para_active_kf')  # 卡尔曼均线

        self.param_list.append('para_active_skd')  # 摆动指标
        self.param_list.append('para_skd_fast_len')
        self.param_list.append('para_skd_slow_len')
        self.param_list.append('para_skd_low')
        self.param_list.append('para_skd_high')

        self.param_list.append('para_macd_fast_len')
        self.param_list.append('para_macd_slow_len')
        self.param_list.append('para_macd_signal_len')

        self.param_list.append('para_active_yb')  # 重心线
        self.param_list.append('para_yb_len')
        self.param_list.append('para_yb_ref')

        self.param_list.append('para_golden_n')  # 黄金分割

        # 输入参数

        self.name = u'RenkoBar'

        self.price_tick = 1  # 商品的最小价格单位
        self.height = 3 * self.price_tick  # 砖块的高度
        self.kilo_height = 0  # 使用价格的千分之一作为基准 >0 才有效,
        self.round_n = 4

        self.activate_ma_tick = False
        self.activate_kf_tick = False
        self.last_price_list = []
        self.avg_price_len = 20
        self.tick_kf = None
        self.line_tick_statemean = []
        self.line_tick_statecovar = []

        self.para_pre_len = 0  # 1

        self.para_ma1_len = 0  # 10
        self.para_ma2_len = 0  # 20
        self.para_ma3_len = 0  # 120

        self.para_ema1_len = 0  # 7
        self.para_ema2_len = 0  # 60
        self.para_ema3_len = 0  # 120

        self.para_ama_len = 0  # 10

        self.para_rsi1_len = 0  # 7     # RSI 相对强弱指数（快曲线）
        self.para_rsi2_len = 0  # 14    # RSI 相对强弱指数（慢曲线）

        self.para_dmi_len = 0  # 14           # DMI的计算周期
        self.para_dmi_max = 0  # 30           # Dpi和Mdi的突破阈值

        self.para_atr1_len = 0  # 10           # ATR波动率的计算周期(近端）
        self.para_atr2_len = 0  # 26           # ATR波动率的计算周期（常用）
        self.para_atr3_len = 0  # 50           # ATR波动率的计算周期（远端）

        self.underlying_symbol = ''  # 商品的短代码, 大写

        # 当前的Tick
        self.cur_tick = None

        # K 线服务的策略
        self.strategy = strategy

        # K线保存数据
        self.base_price = 0  # 基准价格
        self.cur_price = 0
        self.cur_bar = None  # K线数据对象,代表当前还没有走完的renko bar
        self.line_bar = []  # renko bar缓存数据队列
        self.bar_len = 0
        self.max_hold_bars = 3000

        # K 线的相关计算结果数据
        self.line_pre_high = []  # K线的前para_pre_len的的最高
        self.line_pre_low = []  # K线的前para_pre_len的的最低
        self.line_ma1 = []  # K线的MA1均线，周期是InputMaLen1，不包含当前bar
        self.line_ma2 = []  # K线的MA2均线，周期是InputMaLen2，不包含当前bar
        self.line_ma3 = []  # K线的MA2均线，周期是InputMaLen2，不包含当前bar
        self._rt_ma1 = None
        self._rt_ma2 = None
        self._rt_ma3 = None
        self.line_ma1_atan = []
        self.line_ma2_atan = []
        self.line_ma3_atan = []
        self._rt_ma1_atan = None
        self._rt_ma2_atan = None
        self._rt_ma3_atan = None
        self.ma12_count = 0  # ma1 与 ma2 ,金叉/死叉后第几根bar
        self.ma13_count = 0  # ma1 与 ma3 ,金叉/死叉后第几根bar
        self.ma23_count = 0  # ma2 与 ma3 ,金叉/死叉后第几根bar

        self.line_ema1 = []  # K线的EMA1均线，周期是InputEmaLen1，包含当前bar
        self.line_ema1_mtm3_rate = []  # K线的EMA1均线 的momentum(3) 动能
        self.line_ema1_mtm12_rate = []  # K线的EMA1均线 的momentum(12) 动能

        self.line_ema2 = []  # K线的EMA2均线，周期是InputEmaLen2，包含当前bar
        self.line_ema2_mtm3_rate = []  # K线的EMA2均线 的momentum(3) 动能

        self.line_ema3 = []

        self.line_ama = []  # K线的AMA 均线，周期是para_ema1_len

        self.cur_ama_avg_diff = 0  # K线AMA均线的价格差变化率(2*para_ema1_len周期）的平均值
        self.cur_ama_pre_diff = 0  # K线AMA均线的价格差变化率(前一周期）
        self.line_ama_diff_rate = []  # 前后价差变化比率，(1+平均值除以单位价差）* 50， 连续低于 14 平空仓、连续高于86平多仓
        self.line_ama_mtm_rate = []  # 3个价差的斜率
        self.line_ama_er = []  # 变动速率:=整个周期价格的总体变动/每个周期价格变动的累加;

        # K线的DMI( Pdi，Mdi，ADX，Adxr) 计算数据
        self.cur_pdi = 0  # bar内的升动向指标，即做多的比率
        self.cur_mdi = 0  # bar内的下降动向指标，即做空的比率

        self.line_pdi = []  # 升动向指标，即做多的比率
        self.line_mdi = []  # 下降动向指标，即做空的比率

        self.line_dx = []  # 趋向指标列表，最大长度为inputM*2
        self.cur_adx = 0  # Bar内计算的平均趋向指标
        self.line_adx = []  # 平均趋向指标
        self.cur_adxr = 0  # 趋向平均值，为当日ADX值与M日前的ADX值的均值
        self.line_adxr = []  # 平均趋向变化指标

        # K线的基于DMI、ADX计算的结果
        self.cur_adx_trend = 0  # ADX值持续高于前一周期时，市场行情将维持原趋势
        self.cur_adxr_trend = 0  # ADXR值持续高于前一周期时,波动率比上一周期高

        self.signal_adx_long = False  # 多过滤器条件,做多趋势的判断，ADX高于前一天，上升动向> inputMM
        self.signal_adx_short = False  # 空过滤器条件,做空趋势的判断，ADXR高于前一天，下降动向> inputMM

        # K线的ATR技术数据
        self.line_atr1 = []  # K线的ATR1,周期为para_atr1_len
        self.line_atr2 = []  # K线的ATR2,周期为para_atr2_len
        self.line_atr3 = []  # K线的ATR3,周期为para_atr3_len

        self.cur_atr1 = 0
        self.cur_atr2 = 0
        self.cur_atr3 = 0

        # K线的RSI计算数据
        self.line_rsi1 = []  # 记录K线对应的RSI数值，只保留para_rsi1_len*8
        self.line_rsi2 = []  # 记录K线对应的RSI数值，只保留para_rsi2_len*8

        self.para_rsi_low = 30  # RSI的最低线
        self.para_rsi_high = 70  # RSI的最高线

        self.rsi_top_list = []  # 记录RSI的最高峰，只保留 inputRsiLen个
        self.rsi_buttom_list = []  # 记录RSI的最低谷，只保留 inputRsiLen个
        self.cur_rsi_top_buttom = None  # 最近的一个波峰/波谷

        # K线的CMI计算数据
        self.para_cmi_len = 0
        self.line_cmi = []  # 记录K线对应的Cmi数值，只保留para_cmi_len*8

        # K线的布林特计算数据
        self.para_boll_len = 0  # K线周期
        self.para_boll_std_rate = 2  # 两倍标准差

        self.line_boll_upper = []  # 上轨
        self.line_boll_middle = []  # 中线
        self.line_boll_lower = []  # 下轨
        self.line_boll_std = []  # 标准差

        self.line_upper_atan = []
        self.line_middle_atan = []
        self.line_lower_atan = []
        self._rt_upper = None
        self._rt_middle = None
        self._rt_lower = None

        self._rt_upper_atan = None
        self._rt_middle_atan = None
        self._rt_lower_atan = None

        self.cur_upper = 0  # 最后一根K的Boll上轨数值（与price_tick取整）
        self.last_middle = 0  # 最后一根K的Boll中轨数值（与price_tick取整）
        self.last_lower = 0  # 最后一根K的Boll下轨数值（与price_tick取整+1）

        self.para_boll2_len = 0  # K线周期
        self.para_boll2_std_rate = 2  # 两倍标准差
        self.line_boll2_upper = []  # 上轨
        self.line_boll2_middle = []  # 中线
        self.line_boll2_lower = []  # 下轨
        self.line_boll2_std = []  # 标准差
        self.line_upper2_atan = []
        self.line_middle2_atan = []
        self.line_lower2_atan = []
        self._rt_upper2 = None
        self._rt_middle2 = None
        self._rt_lower2 = None
        self._rt_upper2_atan = None
        self._rt_middle2_atan = None
        self._rt_lower2_atan = None

        self.cur_upper2 = 0  # 最后一根K的Boll2上轨数值（与price_tick取整）
        self.cur_middle2 = 0  # 最后一根K的Boll2中轨数值（与price_tick取整）
        self.cur_lower2 = 0  # 最后一根K的Boll2下轨数值（与price_tick取整+1）

        # K线的KDJ指标计算数据
        self.para_kdj_len = 0  # KDJ指标的长度,缺省是9
        self.para_kdj_tb_len = 0  # KDJ指标的长度,缺省是9 ( for TB)
        self.para_kdj_slow_len = 0
        self.para_kdj_smooth_len = 0
        self.line_k = []  # K为快速指标
        self.line_d = []  # D为慢速指标
        self.line_j = []  #
        self.kdj_top_list = []  # 记录KDJ最高峰，只保留 para_kdj_len个
        self.kdj_buttom_list = []  # 记录KDJ的最低谷，只保留 para_kdj_len个
        self.line_rsv = []  # RSV
        self.cur_kdj_top_buttom = {}  # 最近的一个波峰/波谷
        self.cur_k = 0  # bar内计算时，最后一个未关闭的bar的实时K值
        self.cur_d = 0  # bar内计算时，最后一个未关闭的bar的实时值
        self.cur_j = 0  # bar内计算时，最后一个未关闭的bar的实时J值

        self.cur_kd_count = 0  # > 0, 金叉， < 0 死叉
        self.cur_kd_cross = 0  # 最近一次金叉/死叉的点位
        self.cur_kd_cross_price = 0  # 最近一次发生金叉/死叉的价格

        # K线的MACD计算数据(26,12,9)
        self.para_macd_fast_len = 0
        self.para_macd_slow_len = 0
        self.para_macd_signal_len = 0

        self.line_dif = []  # DIF = EMA12 - EMA26，即为talib-MACD返回值macd
        self.line_dea = []  # DEA = （前一日DEA X 8/10 + 今日DIF X 2/10），即为talib-MACD返回值
        self.line_macd = []  # (dif-dea)*2，但是talib中MACD的计算是bar = (dif-dea)*1,国内一般是乘以2
        self.macd_segment_list = []  # macd 金叉/死叉的段列表，记录价格的最高/最低，Dif的最高，最低，Macd的最高/最低，Macd面接
        self._rt_dif = None
        self._rt_dea = None
        self._rt_macd = None
        self.cur_macd_count = 0
        self.cur_macd_cross = 0  # 最近一次金叉/死叉的点位
        self.cur_macd_cross_price = 0  # 最近一次发生金叉/死叉的价格
        self.rt_macd_count = 0  # 实时金叉/死叉, default = 0； -1 实时死叉； 1：实时金叉
        self.rt_macd_cross = 0  # 实时金叉/死叉的位置
        self.rt_macd_cross_price = 0  # 发生实时金叉死叉时的价格
        self.dif_top_divergence = False  # mcad dif 与price 顶背离
        self.dif_buttom_divergence = False  # mcad dif 与price 底背离
        self.macd_top_divergence = False  # mcad 面积 与price 顶背离
        self.macd_buttom_divergence = False  # mcad 面积 与price 底背离

        # K 线的CCI计算数据
        self.para_cci_len = 0
        self.line_cci = []
        self.cur_cci = None
        self._rt_cci = None

        # 卡尔曼过滤器
        self.para_active_kf = False
        self.kf = None
        self.line_state_mean = []
        self.line_state_covar = []

        # SAR
        self.para_sar_step = 0
        self.para_sar_limit = 0
        self.cur_sar_direction = ''
        self.line_sar = []
        self.line_sar_top = []
        self.line_sar_buttom = []
        self.line_sar_sr_up = []
        self.line_sar_ep_up = []
        self.line_sar_af_up = []
        self.line_sar_sr_down = []
        self.line_sar_ep_down = []
        self.line_sar_af_down = []
        self.cur_sar_count = 0  # SAR 上升下降变化后累加

        # 周期
        self.cur_atan = None
        self.line_atan = []
        self.para_active_period = False
        self.cur_period = None  # 当前所在周期
        self.period_list = []

        # 优化的多空动量线
        self.para_active_skd = False
        self.para_skd_fast_len = 13  # 周期1
        self.para_skd_slow_len = 8  # 周期2
        self.line_skd_rsi = []  # 参照的RSI
        self.line_skd_sto = []  # 根据RSI演算的STO
        self.line_sk = []  # 快线
        self.line_sd = []  # 慢线
        self.para_skd_low = 30
        self.para_skd_high = 70
        self.cur_skd_count = 0  # 当前金叉/死叉后累加
        self._rt_sk = None  # 实时SK值
        self._rt_sd = None  # 实时SD值
        self.cur_skd_divergence = 0  # 背离，>0,底背离， < 0 顶背离
        self.skd_top_list = []  # SK 高位
        self.skd_buttom_list = []  # SK 低位
        self.cur_skd_cross = 0  # 最近一次金叉/死叉的点位
        self.cur_skd_cross_price = 0  # 最近一次发生金叉/死叉的价格
        self.rt_skd_count = 0  # 实时金叉/死叉, default = 0； -1 实时死叉； 1：实时金叉
        self.rt_skd_cross = 0  # 实时金叉/死叉的位置
        self.rt_skd_cross_price = 0  # 发生实时金叉死叉时的价格

        # 多空趋势线
        self.para_active_yb = False
        self.line_yb = []
        self.para_yb_ref = 1
        self.para_yb_len = 10
        self.cur_yb_count = 0  # 当前黄/蓝累加
        self._rt_yb = None

        self.para_golden_n = 0  # 黄金分割（一般设置为60）
        self.cur_p192 = None  # HH-(HH-LL) * 0.192;
        self.cur_p382 = None  # HH-(HH-LL) * 0.382;
        self.cur_p500 = None  # (HH+LL)/2;
        self.cur_p618 = None  # HH-(HH-LL) * 0.618;
        self.cur_p809 = None  # HH-(HH-LL) * 0.809;

        # 是否7x24小时运行（ 一般为数字货币）
        self.is_7x24 = False

        # (实时运行时，或者addbar小于bar得周期时，不包含最后一根Bar）
        self.open_array = np.zeros(self.max_hold_bars)  # 与lineBar一致得开仓价清单
        self.open_array[:] = np.nan
        self.high_array = np.zeros(self.max_hold_bars)  # 与lineBar一致得最高价清单
        self.high_array[:] = np.nan
        self.low_array = np.zeros(self.max_hold_bars)  # 与lineBar一致得最低价清单
        self.low_array[:] = np.nan
        self.close_array = np.zeros(self.max_hold_bars)  # 与lineBar一致得收盘价清单
        self.close_array[:] = np.nan

        self.mid3_array = np.zeros(self.max_hold_bars)  # 收盘价/最高/最低价 的平均价
        self.mid3_array[:] = np.nan
        self.mid4_array = np.zeros(self.max_hold_bars)  # 收盘价*2/最高/最低价 的平均价
        self.mid4_array[:] = np.nan
        self.mid5_array = np.zeros(self.max_hold_bars)  # 收盘价*2/开仓价/最高/最低价 的平均价
        self.mid5_array[:] = np.nan

        # 导出到csv文件
        self.export_filename = None
        self.export_fields = []

        # 启动实时得函数
        self.rt_funcs = set()

        # 当前时间
        self.cur_datetime = None

        # 事件回调函数
        self.cb_dict = {}
        if setting:
            self.setParam(setting)

            # 修正精度
            if self.price_tick < 1:
                exponent = decimal.Decimal(str(self.price_tick))
                self.round_n = max(abs(exponent.as_tuple().exponent) + 2, 4)

            # 使用千分之n高度，修改第一个值
            if self.kilo_height > 0:
                self.height = self.price_tick * self.kilo_height

    def __getstate__(self):
        """移除Pickle dump()时不支持的Attribute"""
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        state.pop('strategy', None)
        state.pop('cb_on_bar', None)
        state.pop('cb_dict', None)
        return state

    def __setstate__(self, state):
        """Pickle load()"""
        self.__dict__.update(state)

    def restore(self, state):
        """从Pickle中恢复数据"""
        for key in state.__dict__.keys():
            self.__dict__[key] = state.__dict__[key]

    def setParam(self, setting):
        """设置参数"""
        d = self.__dict__
        for key in self.param_list:
            if key in setting:
                d[key] = setting[key]

    def register_call_back(self, event_type, cb_func):
        """注册事件回调函数"""
        self.cb_dict.update({event_type: cb_func})

    def get_avg_tick_lastprice(self, price, price_time):
        """获取平滑后的价格"""
        if not self.activate_ma_tick:
            return price

        # 添加到最后
        len_list = len(self.last_price_list)
        if len_list > self.avg_price_len:
            self.last_price_list.pop(0)
        if len_list == 0:
            self.last_price_list.append(price)
            return price
        else:
            if price_time == self.cur_datetime:
                # 时间相同，直接更新最后价格(针对成交数据，同一时间存在多个tick）
                self.last_price_list[-1] = price
            else:
                # 时间不同，添加
                self.last_price_list.append(price)
        len_list = len(self.last_price_list)
        # avg_price = ta.MA(np.array(self.last_price_list, dtype=float), len_list)[-1]
        avg_price = sum(self.last_price_list) / len_list
        if np.isnan(avg_price):
            self.write_log(u'计算tick 均值失败')
            return avg_price
        avg_price = round_to(target=self.price_tick, value=avg_price)
        return avg_price

    def get_kf_tick_lastprice(self, price, price_time):
        """获取卡尔曼平滑后的价格"""
        if not self.activate_kf_tick:
            return price

        # 添加到最后
        len_list = len(self.last_price_list)
        if len_list > self.avg_price_len:
            self.last_price_list.pop(0)
        if len_list == 0:
            self.last_price_list.append(price)

            return price
        else:
            self.last_price_list.append(price)
        len_list = len(self.last_price_list)

        if len(self.line_tick_statemean) == 0 or len(self.line_tick_statecovar) == 0:
            self.tick_kf = KalmanFilter(transition_matrices=[1],
                                        observation_matrices=[1],
                                        initial_state_mean=self.last_price_list[-1],
                                        initial_state_covariance=1,
                                        transition_covariance=0.01)
            state_means, state_covariances = self.tick_kf.filter(np.array(self.last_price_list, dtype=float))
            m = state_means[-1].item()
            c = state_covariances[-1].item()
            self.line_tick_statemean.append(m)
            self.line_tick_statecovar.append(c)
        else:
            # 增量计算
            m = self.line_tick_statemean[-1]
            c = self.line_tick_statecovar[-1]
            state_means, state_covariances = self.tick_kf.filter_update(filtered_state_mean=m,
                                                                        filtered_state_covariance=c,
                                                                        observation=np.array(self.last_price_list[-1:],
                                                                                             dtype=float))

            m = state_means[-1].item()
            c = state_covariances[-1].item()
            if np.isnan(m):
                self.write_log(u'计算tick KF均值失败')
                return price
            if len(self.line_tick_statemean) > self.avg_price_len:
                del self.line_tick_statemean[0]
            if len(self.line_tick_statecovar) > self.avg_price_len:
                del self.line_tick_statecovar[0]

            self.line_tick_statemean.append(m)
            self.line_tick_statecovar.append(c)

        kf_price = round_to(target=self.price_tick, value=m)
        return kf_price

    def adjust_tick(self, tick):
        """修正tick的价格，取平均值"""
        # 修正最新价
        if tick.last_price is None or tick.last_price == 0:
            if tick.ask_price1 == 0 and tick.bid_price1 == 0:
                return None
            tick.last_price = round_to(target=self.price_tick, value=(tick.ask_price1 + tick.bid_price1) / 2)

        if self.activate_kf_tick:
            avg_price = self.get_kf_tick_lastprice(tick.last_price, tick.datetime)
        else:
            avg_price = self.get_avg_tick_lastprice(tick.last_price, tick.datetime)

        if avg_price != tick.last_price:
            tick.last_price = avg_price
            if tick.ask_price1 != 0:
                tick.ask_price1 = tick.last_price + self.price_tick
            if tick.bid_price1 != 0:
                tick.bid_price1 = tick.last_price - self.price_tick
        return tick

    def on_tick(self, tick):
        """行情更新
        :type tick: object
        """
        # 排除比bar的时间还早的tick
        if len(self.line_bar) > 0:
            if self.line_bar[-1].datetime > tick.datetime:
                return

        avg_tick = self.adjust_tick(tick)
        if avg_tick is None:
            return
        self.cur_tick = avg_tick
        self.cur_datetime = tick.datetime
        self.cur_price = tick.last_price

        # 更新周期的Close价格
        self.update_period_price(avg_tick.last_price)

        # 3.生成x K线，若形成新Bar，则触发OnBar事件
        self.__draw_line_bar(avg_tick)

    def add_bar(self, bar, *args, **kwargs):
        """仅用于添加renko Bar，不做指标运行计算"""
        is_init = kwargs.get('is_init', False)

        if is_init:
            # 添加renko Bar，并做指标运行计算
            self.on_bar(bar)
            return

        self.update_renko_height(bar.close_price, self.height)

        if bar is None or not isinstance(bar.datetime, datetime):
            return

        if len(self.line_bar) > 0:
            # 新添加得bar比现有得bar时间晚，不添加
            if bar.datetime < self.line_bar[-1].datetime:
                return

        # 更新最后价格
        self.cur_price = bar.close_price
        self.cur_datetime = bar.datetime

        # 计算相关数据
        bar.open_price = round(bar.open_price, self.round_n)
        bar.high_price = round(bar.high_price, self.round_n)
        bar.low_price = round(bar.low_price, self.round_n)
        bar.close_price = round(bar.close_price, self.round_n)

        bar_mid3 = round((bar.close_price + bar.high_price + bar.low_price) / 3, self.round_n)
        bar_mid4 = round((2 * bar.close_price + bar.high_price + bar.low_price) / 4, self.round_n)
        bar_mid5 = round((2 * bar.close_price + bar.open_price + bar.high_price + bar.low_price) / 5, self.round_n)

        # 扩展open,close,high,low numpy array列表
        self.open_array[:-1] = self.open_array[1:]
        self.open_array[-1] = bar.open_price

        self.high_array[:-1] = self.high_array[1:]
        self.high_array[-1] = bar.high_price

        self.low_array[:-1] = self.low_array[1:]
        self.low_array[-1] = bar.low_price

        self.close_array[:-1] = self.close_array[1:]
        self.close_array[-1] = bar.close_price

        self.mid3_array[:-1] = self.mid3_array[1:]
        self.mid3_array[-1] = bar_mid3

        self.mid4_array[:-1] = self.mid4_array[1:]
        self.mid4_array[-1] = bar_mid4

        self.mid5_array[:-1] = self.mid5_array[1:]
        self.mid5_array[-1] = bar_mid5
        self.line_bar.append(bar)
        self.bar_len = len(self.line_bar)

    def on_bar(self, bar):
        """OnBar事件"""
        # 计算相关数据
        # 计算相关数据
        bar_mid3 = round((bar.close_price + bar.high_price + bar.low_price) / 3, self.round_n)
        bar_mid4 = round((2 * bar.close_price + bar.high_price + bar.low_price) / 4, self.round_n)
        bar_mid5 = round((2 * bar.close_price + bar.open_price + bar.high_price + bar.low_price) / 5, self.round_n)

        self.bar_len = len(self.line_bar)

        if bar.close_price > bar.open_price:
            bar.color = Color.RED
        elif bar.close_price < bar.open_price:
            bar.color = Color.BLUE

        # 扩展open,close,high,low 列表
        self.open_array[:-1] = self.open_array[1:]
        self.open_array[-1] = bar.open_price

        self.high_array[:-1] = self.high_array[1:]
        self.high_array[-1] = bar.high_price

        self.low_array[:-1] = self.low_array[1:]
        self.low_array[-1] = bar.low_price

        self.close_array[:-1] = self.close_array[1:]
        self.close_array[-1] = bar.close_price

        self.mid3_array[:-1] = self.mid3_array[1:]
        self.mid3_array[-1] = bar_mid3

        self.mid4_array[:-1] = self.mid4_array[1:]
        self.mid4_array[-1] = bar_mid4

        self.mid5_array[:-1] = self.mid5_array[1:]
        self.mid5_array[-1] = bar_mid5

        # 添加bar=>lineBar
        self.line_bar.append(bar)

        bar_close_time = bar.datetime + timedelta(seconds=bar.seconds)
        if self.cur_datetime is None or self.cur_datetime < bar_close_time:
            self.cur_datetime = bar_close_time

        self.__count_pre_high_low()
        self.__count_sar()
        self.__count_ma()
        self.__count_ema()
        self.__count_atr()
        self.__count_rsi()
        self.__count_ama()

        self.__count_dmi()
        self.__count_cmi()
        self.__count_kdj()
        self.__count_kdj_tb()
        self.__count_boll()
        self.__count_macd()
        self.__count_cci()
        self.__count_kf()
        self.__count_skd()
        self.__count_yb()
        self.__count_golden_section()

        self.__count_period(bar)

        self.export_to_csv(bar)

        self.runtime_recount()

        # 回调上层调用者
        self.cb_on_bar(bar, self.name)

        self.update_renko_height(bar.close_price, self.height)

    def update_renko_height(self, cur_price, height):
        """更新砖块高度"""
        if self.kilo_height > 0:
            # 调整新砖块高度  max(价格的千分之一, 最小跳动）* 高度数量
            new_height = int(
                max(cur_price / 1000, self.price_tick) * self.kilo_height / self.price_tick) * self.price_tick
            if new_height != self.height:
                self.write_log(u'修改:{}砖块高度:{}=>{}'.format(self.name, self.height, new_height))
                self.height = new_height
        elif height != self.height:
            self.write_log(u'修改:{}砖块高度:{}=>{}'.format(self.name, self.height, height))
            self.height = height

    def runtime_recount(self):
        """
        根据实时计算得要求，执行实时指标计算
        :return:
        """
        for func in list(self.rt_funcs):
            try:
                func()
            except Exception as ex:
                print(u'{}调用实时计算,异常:{},{}'.format(self.name, str(ex), traceback.format_exc()), file=sys.stderr)

    def update_period_price(self, price):
        """更新周期的Close价格"""

        if len(self.period_list) < 1:
            return

        self.period_list[-1].update_price(price)

    def get_last_bar_str(self):
        """显示最后一个Bar的信息"""
        msg = u'[' + self.name + u']'

        if len(self.line_bar) < 2:
            return msg

        display_bar = self.line_bar[-1]

        msg = msg + u'ad:{} o:{};h:{};l:{};c:{},v:{},{}'. \
            format(display_bar.datetime, display_bar.open_price, display_bar.high_price,
                   display_bar.low_price, display_bar.close_price, display_bar.volume, display_bar.color)
        if self.para_ma1_len > 0 and len(self.line_ma1) > 0:
            msg = msg + u',MA({0}):{1}'.format(self.para_ma1_len, self.line_ma1[-1])

        if self.para_ma2_len > 0 and len(self.line_ma2) > 0:
            msg = msg + u',MA({0}):{1}'.format(self.para_ma2_len, self.line_ma2[-1])
            if self.ma12_count == 1:
                msg = msg + u'MA{}金叉MA{}'.format(self.para_ma1_len, self.para_ma2_len)
            elif self.ma12_count == -1:
                msg = msg + u'MA{}死叉MA{}'.format(self.para_ma1_len, self.para_ma2_len)

        if self.para_ma3_len > 0 and len(self.line_ma3) > 0:
            msg = msg + u',MA({0}):{1}'.format(self.para_ma3_len, self.line_ma3[-1])
            if self.ma13_count == 1:
                msg = msg + u'MA{}金叉MA{}'.format(self.para_ma1_len, self.para_ma3_len)
            elif self.ma13_count == -1:
                msg = msg + u'MA{}死叉MA{}'.format(self.para_ma1_len, self.para_ma3_len)

            if self.ma23_count == 1:
                msg = msg + u'MA{}金叉MA{}'.format(self.para_ma2_len, self.para_ma3_len)
            elif self.ma23_count == -1:
                msg = msg + u'MA{}死叉MA{}'.format(self.para_ma2_len, self.para_ma3_len)

        if self.para_ema1_len > 0 and len(self.line_ema1) > 0:
            msg = msg + u',EMA({0}):{1}'.format(self.para_ema1_len, self.line_ema1[-1])
        if self.para_ema2_len > 0 and len(self.line_ema2) > 0:
            msg = msg + u',EMA({0}):{1}'.format(self.para_ema2_len, self.line_ema2[-1])
        if self.para_ema3_len > 0 and len(self.line_ema3) > 0:
            msg = msg + u',EMA({0}):{1}'.format(self.para_ema3_len, self.line_ema3[-1])

        if self.para_dmi_len > 0 and len(self.line_pdi) > 0:
            msg = msg + u',Pdi:{1};Mdi:{1};Adx:{2}'.format(self.line_pdi[-1], self.line_mdi[-1], self.line_adx[-1])

        if self.para_atr1_len > 0 and len(self.line_atr1) > 0:
            msg = msg + u',Atr({0}):{1}'.format(self.para_atr1_len, self.line_atr1[-1])

        if self.para_atr2_len > 0 and len(self.line_atr2) > 0:
            msg = msg + u',Atr({0}):{1}'.format(self.para_atr2_len, self.line_atr2[-1])

        if self.para_atr3_len > 0 and len(self.line_atr3) > 0:
            msg = msg + u',Atr({0}):{1}'.format(self.para_atr3_len, self.line_atr3[-1])

        if self.para_rsi1_len > 0 and len(self.line_rsi1) > 0:
            msg = msg + u',Rsi({0}):{1}'.format(self.para_rsi1_len, self.line_rsi1[-1])

        if self.para_rsi2_len > 0 and len(self.line_rsi2) > 0:
            msg = msg + u',Rsi({0}):{1}'.format(self.para_rsi2_len, self.line_rsi2[-1])

        if self.para_kdj_len > 0 and len(self.line_k) > 0:
            msg = msg + u',KDJ({},{}):{},{},{}'.format(self.para_kdj_len,
                                                       self.para_kdj_slow_len,
                                                       round(self.line_k[-1], self.round_n),
                                                       round(self.line_d[-1], self.round_n),
                                                       round(self.line_j[-1], self.round_n))

        if self.para_kdj_tb_len > 0 and len(self.line_k) > 0:
            msg = msg + u',KDJ_TB({},{}):K:{},D:{},J:{},kd_count:{},cross_k:{},cross_price:{}' \
                .format(self.para_kdj_tb_len,
                        self.para_kdj_slow_len,
                        round(self.line_k[-1], self.round_n),
                        round(self.line_d[-1], self.round_n),
                        round(self.line_j[-1], self.round_n),
                        self.cur_kd_count, self.cur_kd_cross, self.cur_kd_cross_price)

        if self.para_cci_len > 0 and len(self.line_cci) > 0:
            msg = msg + u',Cci({0}):{1}'.format(self.para_cci_len, self.line_cci[-1])

        if self.para_boll_len > 0 and len(self.line_boll_upper) > 0:
            msg = msg + u',Boll({}):std:{},mid:{},up:{},low:{},Atan:[mid:{},up:{},low:{}]'. \
                format(self.para_boll_len, round(self.line_boll_upper[-1], self.round_n),
                       round(self.line_boll_middle[-1], self.round_n), round(self.line_boll_lower[-1], self.round_n),
                       round(self.line_boll_std[-1], self.round_n),
                       round(self.line_middle_atan[-1], self.round_n) if len(self.line_middle_atan) > 0 else 0,
                       round(self.line_upper_atan[-1], self.round_n) if len(self.line_upper_atan) > 0 else 0,
                       round(self.line_lower_atan[-1], self.round_n) if len(self.line_lower_atan) > 0 else 0)

        if self.para_boll2_len > 0 and len(self.line_boll2_upper) > 0:
            msg = msg + u',Boll2({0}):std:{4},m:{2},u:{1},l:{3}'. \
                format(self.para_boll2_len, round(self.line_boll2_upper[-1], self.round_n),
                       round(self.line_boll2_middle[-1], self.round_n), round(self.line_boll2_lower[-1], self.round_n),
                       round(self.line_boll_std[-1], self.round_n))

        if self.para_macd_fast_len > 0 and len(self.line_dif) > 0:
            msg = msg + u',MACD({0},{1},{2}):Dif:{3},Dea{4},Macd:{5}'. \
                format(self.para_macd_fast_len, self.para_macd_slow_len, self.para_macd_signal_len,
                       round(self.line_dif[-1], self.round_n),
                       round(self.line_dea[-1], self.round_n),
                       round(self.line_macd[-1], self.round_n))
            if len(self.line_macd) > 2:
                if self.line_macd[-2] < 0 < self.line_macd[-1]:
                    msg = msg + u'金叉 '
                elif self.line_macd[-2] > 0 > self.line_macd[-1]:
                    msg = msg + u'死叉 '

                if self.dif_top_divergence:
                    msg = msg + u'Dif顶背离 '
                if self.macd_top_divergence:
                    msg = msg + u'MACD顶背离 '
                if self.dif_buttom_divergence:
                    msg = msg + u'Dif底背离 '
                if self.macd_buttom_divergence:
                    msg = msg + u'MACD低背离 '

        if self.para_active_kf and len(self.line_state_mean) > 0:
            msg = msg + u',Kalman:{0}'.format(self.line_state_mean[-1])

        if self.para_active_skd and len(self.line_sk) > 1 and len(self.line_sd) > 1:

            msg = msg + u',SK:{}/SD:{}{}{},count:{}' \
                .format(round(self.line_sk[-1], self.round_n),
                        round(self.line_sd[-1], self.round_n),
                        u'金叉' if self.cur_skd_count == 1 else u'',
                        u'死叉' if self.cur_skd_count == -1 else u'',
                        self.cur_skd_count)

            if self.cur_skd_divergence == 1:
                msg = msg + u'底背离'
            elif self.cur_skd_divergence == -1:
                msg = msg + u'顶背离'

        if self.para_active_yb and len(self.line_yb) > 1:
            c = 'Blue' if self.line_yb[-1] < self.line_yb[-2] else 'Yellow'
            msg = msg + u',YB:{},[{}({})]'.format(self.line_yb[-1], c, self.cur_yb_count)

        return msg

    # ----------------------------------------------------------------------
    def __first_tick(self, tick, open=0, source=None):
        """当前砖块的第一个tick数据"""

        self.cur_bar = RenkoBarData(gateway_name='',
                                    symbol=tick.symbol,
                                    exchange=tick.exchange,
                                    datetime=tick.datetime)  # 创建新的K线
        self.cur_bar.symbol = tick.symbol
        # 取最新的RenkoHeight
        self.cur_bar.height = self.height

        if open > 0:
            self.cur_bar.open_price = open
            self.cur_bar.high_price = max(open, tick.last_price)
            self.cur_bar.low_price = min(open, tick.last_price)
        else:
            self.cur_bar.open_price = round(tick.last_price, self.round_n)
            self.cur_bar.high_price = tick.last_price
            self.cur_bar.low_price = tick.last_price

        self.cur_bar.close_price = round(tick.last_price, self.round_n)

        # K线的日期时间，
        self.cur_bar.date = tick.date
        self.cur_bar.time = tick.time

        self.cur_bar.trading_day = tick.trading_day

        # K线的上限线和下限线
        if source == Color.RED:
            self.cur_bar.up_band = self.cur_bar.open_price
            self.cur_bar.down_band = round(self.cur_bar.open_price - self.cur_bar.height, self.round_n)

        elif source == Color.BLUE:
            self.cur_bar.up_band = round(self.cur_bar.open_price + self.cur_bar.height, self.round_n)
            self.cur_bar.down_band = self.cur_bar.open_price
        else:
            self.cur_bar.up_band = round(
                int((self.cur_bar.open_price + self.cur_bar.height / 2) / self.price_tick) * self.price_tick,
                self.round_n)
            self.cur_bar.down_band = round(self.cur_bar.up_band - self.cur_bar.height, self.round_n)

        # 初始化进入high区域和low区域的首次时间
        if tick.last_price < self.cur_bar.down_band:
            self.cur_bar.low_time = tick.datetime
        elif tick.last_price > self.cur_bar.up_band:
            self.cur_bar.high_time = tick.datetime

        self.cur_bar.volume = tick.volume
        self.cur_bar.open_interest = tick.open_interest

    # ----------------------------------------------------------------------
    def __draw_line_bar(self, tick):
        """生成 Renko Bar
        """
        # 2、处理第一个tick数据
        if self.base_price == 0:
            # 第一个数据的价格作为基准价格
            self.base_price = tick.last_price

            source = ''
            if len(self.line_bar) > 0:
                last_bar = self.line_bar[-1]
                source = last_bar.color
            # 当前分钟的第一个tick
            self.__first_tick(tick=tick, open=self.base_price, source=source)
            return

        if (tick.datetime.hour == 9 or tick.datetime.hour == 13 or tick.datetime.hour == 21) \
                and tick.datetime.minute == 0 and tick.datetime.second == 0:
            self.cur_bar.datetime = tick.datetime
            if tick.last_price > self.cur_bar.up_band:
                self.cur_bar.high_time = tick.datetime
                self.cur_bar.low_time = None
            elif tick.last_price < self.cur_bar.down_band:
                self.cur_bar.high_time = None
                self.cur_bar.low_time = tick.datetime
            else:
                self.cur_bar.low_time = None
                self.cur_bar.high_time = None

        bar = copy.copy(self.cur_bar)

        # 3、当前没有砖块
        if len(self.line_bar) == 0:

            up_price = int((self.base_price + bar.height / 2) / self.price_tick) * self.price_tick
            down_price = round(up_price - bar.height, self.round_n)

            # 超出上限值
            if tick.last_price >= up_price:
                self.__append_red_bar(up_price, bar, tick)

            # 超出下限值
            elif tick.last_price <= down_price:
                self.__append_blue_bar(down_price, bar, tick)

            # 更新self.barRenko
            else:
                # self.barRenko.datetime = tick.datetime
                self.cur_bar.high_price = max(tick.last_price, self.cur_bar.high_price)
                self.cur_bar.low_price = min(tick.last_price, self.cur_bar.low_price)
                self.cur_bar.close_price = tick.last_price

                self.cur_bar.volume = self.cur_bar.volume + tick.volume
                self.cur_bar.open_interest = tick.open_interest

            # 仅为第一个bar，后续逻辑无意义，退出
            return

        # 4、当前有砖块，比较最后一个砖块
        lastBar = self.line_bar[-1]

        if lastBar.color == Color.BLUE:
            up_price = round(lastBar.open_price + self.cur_bar.height, self.round_n)
            down_price = round(lastBar.close_price - self.cur_bar.height, self.round_n)
        else:
            up_price = round(lastBar.close_price + self.cur_bar.height, self.round_n)
            down_price = round(lastBar.open_price - self.cur_bar.height, self.round_n)

        # 超出上限值
        if tick.last_price >= up_price:
            # self.writeCtaLog(u'{0},price:{1}'.format(tick.datetime, tick.last_price))
            self.__append_red_bar(up_price - self.cur_bar.height, bar, tick)

        # 低于下限值
        elif tick.last_price <= down_price:
            # self.writeCtaLog(u'{0},price:{1}'.format(tick.datetime, tick.last_price))
            self.__append_blue_bar(down_price + self.cur_bar.height, bar, tick)

        # 更新self.barRenko
        else:
            # 首先更新上限/下限区域的时间
            if tick.last_price > self.cur_bar.up_band \
                    and self.cur_bar.close_price <= self.cur_bar.up_band:
                self.cur_bar.high_time = tick.datetime
            if tick.last_price < self.cur_bar.down_band \
                    and self.cur_bar.close_price >= self.cur_bar.down_band:
                self.cur_bar.low_time = tick.datetime

            self.cur_bar.high_price = max(tick.last_price, self.cur_bar.high_price)
            self.cur_bar.low_price = min(tick.last_price, self.cur_bar.low_price)
            self.cur_bar.close_price = tick.last_price

            self.cur_bar.volume = self.cur_bar.volume + tick.volume
            self.cur_bar.open_interest = tick.open_interest

            # 实时计算临时砖块的颜色
            if self.cur_bar.close_price > self.cur_bar.open_price:
                self.cur_bar.color = Color.RED
            elif self.cur_bar.close_price < self.cur_bar.open_price:
                self.cur_bar.color = Color.BLUE
            else:
                self.cur_bar.color = Color.EQUAL

            if self.line_bar[-1].color == Color.BLUE and self.cur_bar.color == Color.RED \
                    and self.cur_bar.low_price < self.line_bar[-1].close_price \
                    and self.cur_bar.close_price > self.line_bar[-1].open_price \
                    and (tick.datetime - self.cur_bar.datetime).total_seconds() < 5:
                self.write_log(u'Fast Return Bule=>Red in {0}s,l:{1}=>c{2},'
                               .format((tick.datetime - self.cur_bar.datetime).seconds,
                                       self.cur_bar.low_price, self.line_bar[-1].close_price))

                func = self.cb_dict.get(self.CB_FAST_RETURN, None)
                if func:
                    try:
                        func(old_color=Color.BLUE,
                             new_color=Color.RED,
                             seconds=(tick.datetime - self.cur_bar.datetime).total_seconds())

                    except Exception as ex:
                        self.write_log(u'call back event{} exception:{}'.format(self.CB_FAST_RETURN, str(ex)))
                        self.write_log(u'traceback:{}'.format(traceback.format_exc()))

            elif self.line_bar[-1].color == Color.RED and self.cur_bar.color == Color.BLUE \
                    and self.cur_bar.high_price > self.line_bar[-1].close_price \
                    and self.cur_bar.close_price < self.line_bar[-1].open_price \
                    and (tick.datetime - self.cur_bar.datetime).seconds < 5:
                self.write_log(u'Fast Return Red=>Blue in {0}s,l:{1}=>c{2},'
                               .format((tick.datetime - self.cur_bar.datetime).seconds,
                                       self.cur_bar.high_price, self.line_bar[-1].close_price))
                func = self.cb_dict.get(self.CB_FAST_RETURN, None)
                if func:
                    try:
                        func(old_color=Color.RED,
                             new_color=Color.BLUE,
                             seconds=(tick.datetime - self.cur_bar.datetime).total_seconds())
                    except Exception as ex:
                        self.write_log(u'call back event{} exception:{}'.format(self.CB_FAST_RETURN, str(ex)))
                        self.write_log(u'traceback:{}'.format(traceback.format_exc()))
        return False

    # ----------------------------------------------------------------------
    def __append_red_bar(self, basePrice, bar, tick):
        """增加红色的Big Renko Bar
        basePrice,基准价,=Open & Low Price
        bar，由 barRenko复制的k线数据
        tick 行情数据"""

        # 将之前的barRenko 复制带颜色的bar推入K线
        bar.high_price = round(basePrice + bar.height, self.round_n)
        bar.close_price = round(basePrice + bar.height, self.round_n)
        # bar.low = basePrice 保留下影线
        bar.open_price = round(basePrice, self.round_n)

        bar.color = Color.RED

        bar.seconds = (tick.datetime - bar.datetime).seconds
        if bar.high_time:
            bar.high_seconds = (tick.datetime - bar.high_time).seconds
        if bar.low_time:
            bar.low_seconds = (tick.datetime - bar.low_time).seconds

        self.on_bar(bar)

        # 更新 barRenko,,开仓价为上一个bar的close,颜色为红色
        self.__first_tick(tick, open=bar.close_price, source=bar.color)

        # 若价格仍超出新砖块一个砖块的高度差，递归增加
        if tick.last_price >= basePrice + bar.height * 2:
            bar2 = copy.copy(bar)
            bar2.low_price = bar2.close_price
            bar2.datetime = tick.datetime
            bar2.highTime = tick.datetime

            self.__append_red_bar(basePrice + bar.height, bar2, tick)

    # ----------------------------------------------------------------------
    def __append_blue_bar(self, basePrice, bar, tick):
        """增加蓝色的Big Renko Bar

        basePrice,基准价,Open & High Price
        bar，由 barRenko复制的k线数据
        tick 行情数据"""

        # 将之前的barRenko 复制带颜色的bar推入K线

        bar.open_price = round(basePrice, self.round_n)
        bar.low_price = round(basePrice - bar.height, self.round_n)
        bar.close_price = round(basePrice - bar.height, self.round_n)

        bar.color = Color.BLUE

        bar.seconds = (tick.datetime - bar.datetime).seconds
        if bar.high_time:
            bar.high_seconds = (tick.datetime - bar.high_time).seconds
        if bar.low_time:
            bar.low_seconds = (tick.datetime - bar.low_time).seconds

        self.on_bar(bar)

        # 更新 barRenko,开仓价为上一个bar的close,颜色为蓝色
        self.__first_tick(tick, open=bar.close_price, source=bar.color)

        # 若价格仍低于新砖块的一个砖块高度差，递归增加
        if tick.last_price <= basePrice - bar.height * 2:
            bar2 = copy.copy(bar)
            bar2.high_price = bar2.close_price
            bar2.datetime = tick.datetime
            bar2.lowTime = tick.datetime

            self.__append_blue_bar(basePrice - bar.height, bar2, tick)

    # ----------------------------------------------------------------------
    def __count_pre_high_low(self):
        """计算 K线的前周期最高和最低"""
        if self.para_pre_len <= 0:  # 不计算
            return

        # 1、lineBar满足长度才执行计算
        if self.bar_len < self.para_pre_len:
            self.write_log(u'数据未充分,当前Bar数据数量：{0}，计算High、Low需要：{1}'.
                           format(len(self.line_bar), self.para_pre_len))
            return

        # 2.计算前para_pre_len周期内(不包含当前周期）的Bar高点和低点
        pre_high = max(self.high_array[-self.para_pre_len:])
        pre_low = min(self.low_array[-self.para_pre_len:])

        # 保存
        if len(self.line_pre_high) > self.max_hold_bars:
            del self.line_pre_high[0]
        self.line_pre_high.append(pre_high)

        # 保存
        if len(self.line_pre_low) > self.max_hold_bars:
            del self.line_pre_low[0]
        self.line_pre_low.append(pre_low)

    # ----------------------------------------------------------------------
    def __count_dmi(self):
        """计算K线的DMI数据和条件"""
        if self.para_dmi_len <= 0:  # 不计算
            return

        # 1、lineMx满足长度才执行计算
        if self.bar_len < self.para_dmi_len + 1:
            self.write_log(u'数据未充分,当前Bar数据数量：{0}，计算DMI需要：{1}'.format(len(self.line_bar), self.para_dmi_len + 1))
            return

        # 2、根据当前High，Low，(包含当前周期）重新计算TR1，PDM，MDM和ATR
        barTr1 = 0  # 获取InputP周期内的价差最大值之和
        barPdm = 0  # InputP周期内的做多价差之和
        barMdm = 0  # InputP周期内的做空价差之和

        for i in range(len(self.line_bar) - 1, len(self.line_bar) - 1 - self.para_dmi_len, -1):  # 周期 para_dmi_len
            # 3.1、计算TR1

            # 当前周期最高与最低的价差
            high_low_spread = self.line_bar[i].high_price - self.line_bar[i].low_price
            # 当前周期最高与昨收价的价差
            high_preclose_spread = abs(self.line_bar[i].high_price - self.line_bar[i - 1].close_price)
            # 当前周期最低与昨收价的价差
            low_preclose_spread = abs(self.line_bar[i].low_price - self.line_bar[i - 1].close_price)

            # 最大价差
            max_spread = max(high_low_spread, high_preclose_spread, low_preclose_spread)
            barTr1 = barTr1 + float(max_spread)

            # 今高与昨高的价差
            high_prehigh_spread = self.line_bar[i].high_price - self.line_bar[i - 1].high_price
            # 昨低与今低的价差
            low_prelow_spread = self.line_bar[i - 1].low_price - self.line_bar[i].low_price

            # 3.2、计算周期内的做多价差之和
            if high_prehigh_spread > 0 and high_prehigh_spread > low_prelow_spread:
                barPdm = barPdm + high_prehigh_spread

            # 3.3、计算周期内的做空价差之和
            if low_prelow_spread > 0 and low_prelow_spread > high_prehigh_spread:
                barMdm = barMdm + low_prelow_spread

        # 6、计算上升动向指标，即做多的比率
        if barTr1 == 0:
            self.cur_pdi = 0
        else:
            self.cur_pdi = barPdm * 100 / barTr1

        if len(self.line_pdi) > self.max_hold_bars:
            del self.line_pdi[0]

        self.line_pdi.append(self.cur_pdi)

        # 7、计算下降动向指标，即做空的比率
        if barTr1 == 0:
            self.cur_mdi = 0
        else:
            self.cur_mdi = barMdm * 100 / barTr1

        # 8、计算平均趋向指标 Adx，Adxr
        if self.cur_mdi + self.cur_pdi == 0:
            dx = 0
        else:
            dx = 100 * abs(self.cur_mdi - self.cur_pdi) / (self.cur_mdi + self.cur_pdi)
        if len(self.line_mdi) > self.max_hold_bars:
            del self.line_mdi[0]
        self.line_mdi.append(self.cur_mdi)

        if len(self.line_dx) > self.max_hold_bars:
            del self.line_dx[0]
        self.line_dx.append(dx)

        # 平均趋向指标，MA计算
        if len(self.line_dx) < self.para_dmi_len + 1:
            self.cur_adx = dx
        else:
            self.cur_adx = ta.EMA(np.array(self.line_dx, dtype=float), self.para_dmi_len)[-1]

        # 保存Adx值
        if len(self.line_adx) > self.max_hold_bars:
            del self.line_adx[0]
        self.line_adx.append(self.cur_adx)

        # 趋向平均值，为当日ADX值与1周期前的ADX值的均值
        if len(self.line_adx) == 1:
            self.cur_adxr = self.line_adx[-1]
        else:
            self.cur_adxr = (self.line_adx[-1] + self.line_adx[-2]) / 2

        # 保存Adxr值
        if len(self.line_adxr) > self.max_hold_bars:
            del self.line_adxr[0]
        self.line_adxr.append(self.cur_adxr)

        # 7、计算A，ADX值持续高于前一周期时，市场行情将维持原趋势
        if len(self.line_adx) < 2:
            self.cur_adx_trend = False
        elif self.line_adx[-1] > self.line_adx[-2]:
            self.cur_adx_trend = True
        else:
            self.cur_adx_trend = False

        # ADXR值持续高于前一周期时,波动率比上一周期高
        if len(self.line_adxr) < 2:
            self.cur_adxr_trend = False
        elif self.line_adxr[-1] > self.line_adxr[-2]:
            self.cur_adxr_trend = True
        else:
            self.cur_adxr_trend = False

        # 多过滤器条件,做多趋势，ADX高于前一天，上升动向> para_dmi_max
        if self.cur_pdi > self.cur_mdi and self.cur_adx_trend and self.cur_adxr_trend and self.cur_pdi >= self.para_dmi_max:
            self.signal_adx_long = True
            self.write_log(u'{0}[DEBUG]Buy Signal On Renko Bar,Pdi:{1}>Mdi:{2},adx[-1]:{3}>Adx[-2]:{4}'
                           .format(self.cur_tick.datetime, self.cur_pdi, self.cur_mdi, self.line_adx[-1],
                                   self.line_adx[-2]))
        else:
            self.signal_adx_long = False

        # 空过滤器条件 做空趋势，ADXR高于前一天，下降动向> inputMM
        if self.cur_pdi < self.cur_mdi and self.cur_adx_trend and self.cur_adxr_trend and self.cur_mdi >= self.para_dmi_max:
            self.signal_adx_short = True
            self.write_log(u'{0}[DEBUG]Short Signal On Renko Bar,Pdi:{1}<Mdi:{2},adx[-1]:{3}>Adx[-2]:{4}'
                           .format(self.cur_tick.datetime, self.cur_pdi, self.cur_mdi, self.line_adx[-1],
                                   self.line_adx[-2]))
        else:
            self.signal_adx_short = False

    def get_sar(self, direction, cur_sar, cur_af=0, sar_limit=0.2, sar_step=0.02):
        """
        抛物线计算方法
        :param direction: Direction
        :param cur_sar: 当前抛物线价格
        :param cur_af: 当前抛物线价格
        :param sar_limit: 最大加速范围
        :param sar_step: 加速因子
        :return: 新的
        """
        if np.isnan(self.high_array[-1]):
            return cur_sar, cur_af
        # 向上抛物线
        if direction == Direction.LONG:
            af = min(sar_limit, cur_af + sar_step)
            ep = self.high_array[-1]
            sar = cur_sar + af * (ep - cur_sar)
            return sar, af
        # 向下抛物线
        elif direction == Direction.SHORT:
            af = min(sar_limit, cur_af + sar_step)
            ep = self.low_array[-1]
            sar = cur_sar + af * (ep - cur_sar)
            return sar, af
        else:
            return cur_sar, cur_af

    def __count_sar(self):
        """计算K线的SAR"""

        if self.bar_len < 5:
            return

        if not (self.para_sar_step > 0 or self.para_sar_limit > self.para_sar_step):  # 不计算
            return

        if len(self.line_sar_sr_up) == 0 and len(self.line_sar_sr_down) == 0:
            if self.line_bar[-2].close_price > self.line_bar[-5].close_price:
                # 标记为上涨趋势
                sr0 = min(self.low_array[0:])
                af0 = 0
                ep0 = self.high_array[-1]
                self.line_sar_sr_up.append(sr0)
                self.line_sar_ep_up.append(ep0)
                self.line_sar_af_up.append(af0)
                self.line_sar.append(sr0)
                self.cur_sar_direction = 'up'
                self.cur_sar_count = 0
            else:
                # 标记为下跌趋势
                sr0 = max(self.high_array[0:])
                af0 = 0
                ep0 = self.low_array[-1]
                self.line_sar_sr_down.append(sr0)
                self.line_sar_ep_down.append(ep0)
                self.line_sar_af_down.append(af0)
                self.line_sar.append(sr0)
                self.cur_sar_direction = 'down'
                self.cur_sar_count = 0
            self.line_sar_top.append(self.line_bar[-2].high_price)  # SAR的谷顶
            self.line_sar_buttom.append(self.line_bar[-2].low_price)  # SAR的波底

        # 当前处于上升抛物线
        elif len(self.line_sar_sr_up) > 0:

            # # K线low，仍然在上升抛物线上方，延续
            if self.low_array[-1] > self.line_sar_sr_up[-1]:

                sr0 = self.line_sar_sr_up[-1]
                ep0 = self.high_array[-1]  # 文华使用前一个K线的最高价
                af0 = min(self.para_sar_limit,
                          self.line_sar_af_up[-1] + self.para_sar_step)  # 文华的af随着K线的数目增加而递增，没有判断新高
                # 计算出新的抛物线价格
                sr = sr0 + af0 * (ep0 - sr0)

                self.line_sar_sr_up.append(sr)
                self.line_sar_ep_up.append(ep0)
                self.line_sar_af_up.append(af0)
                self.line_sar.append(sr)
                self.cur_sar_count += 1
                # self.write_log('Up: sr0={},ep0={},af0={},sr={}'.format(sr0, ep0, af0, sr))

            # K线最低，触碰上升的抛物线 =》 转为 下降抛物线
            elif self.low_array[-1] <= self.line_sar_sr_up[-1]:
                ep0 = max(self.high_array[-len(self.line_sar_sr_up):])
                sr0 = ep0
                af0 = 0
                self.line_sar_sr_down.append(sr0)
                self.line_sar_ep_down.append(ep0)
                self.line_sar_af_down.append(af0)
                self.line_sar.append(sr0)
                self.cur_sar_direction = 'down'
                # self.write_log('Up->Down: sr0={},ep0={},af0={},sr={}'.format(sr0, ep0, af0, sr0))
                # self.write_log('lineSarTop={}, lineSarButtom={}, len={}'.format(self.lineSarTop[-1], self.lineSarButtom[-1],len(self.lineSarSrUp)))
                self.line_sar_top.append(self.line_bar[-2].high_price)
                self.line_sar_buttom.append(self.line_bar[-2].low_price)
                self.line_sar_sr_up = []
                self.line_sar_ep_up = []
                self.line_sar_af_up = []
                sr0 = self.line_sar_sr_down[-1]
                ep0 = self.low_array[-1]  # 文华使用前一个K线的最低价
                af0 = min(self.para_sar_limit,
                          self.line_sar_af_down[-1] + self.para_sar_step)  # 文华的af随着K线的数目增加而递增，没有判断新高
                sr = sr0 + af0 * (ep0 - sr0)
                self.line_sar_sr_down.append(sr)
                self.line_sar_ep_down.append(ep0)
                self.line_sar_af_down.append(af0)
                self.line_sar.append(sr)
                self.cur_sar_count = 0
                # self.write_log('Down: sr0={},ep0={},af0={},sr={}'.format(sr0, ep0, af0, sr))
        elif len(self.line_sar_sr_down) > 0:
            if self.high_array[-1] < self.line_sar_sr_down[-1]:
                sr0 = self.line_sar_sr_down[-1]
                ep0 = self.low_array[-1]  # 文华使用前一个K线的最低价
                af0 = min(self.para_sar_limit,
                          self.line_sar_af_down[-1] + self.para_sar_step)  # 文华的af随着K线的数目增加而递增，没有判断新高
                sr = sr0 + af0 * (ep0 - sr0)
                self.line_sar_sr_down.append(sr)
                self.line_sar_ep_down.append(ep0)
                self.line_sar_af_down.append(af0)
                self.line_sar.append(sr)
                self.cur_sar_count -= 1
                # self.write_log('Down: sr0={},ep0={},af0={},sr={}'.format(sr0, ep0, af0, sr))
            elif self.high_array[-1] >= self.line_sar_sr_down[-1]:
                ep0 = min(self.low_array[-len(self.line_sar_sr_down):])
                sr0 = ep0
                af0 = 0
                self.line_sar_sr_up.append(sr0)
                self.line_sar_ep_up.append(ep0)
                self.line_sar_af_up.append(af0)
                self.line_sar.append(sr0)
                self.cur_sar_direction = 'up'
                # self.write_log('Down->Up: sr0={},ep0={},af0={},sr={}'.format(sr0, ep0, af0, sr0))
                # self.write_log('lineSarTop={}, lineSarButtom={}, len={}'.format(self.lineSarTop[-1], self.lineSarButtom[-1],len(self.lineSarSrDown)))
                self.line_sar_top.append(self.line_bar[-2].high_price)
                self.line_sar_buttom.append(self.line_bar[-2].low_price)
                self.line_sar_sr_down = []
                self.line_sar_ep_down = []
                self.line_sar_af_down = []
                sr0 = self.line_sar_sr_up[-1]
                ep0 = self.high_array[-1]  # 文华使用前一个K线的最高价
                af0 = min(self.para_sar_limit,
                          self.line_sar_af_up[-1] + self.para_sar_step)  # 文华的af随着K线的数目增加而递增，没有判断新高
                sr = sr0 + af0 * (ep0 - sr0)
                self.line_sar_sr_up.append(sr)
                self.line_sar_ep_up.append(ep0)
                self.line_sar_af_up.append(af0)
                self.line_sar.append(sr)
                self.cur_sar_count = 0
                self.write_log('Up: sr0={},ep0={},af0={},sr={}'.format(sr0, ep0, af0, sr))

        # 更新抛物线的最高值和最低值
        if self.line_sar_top[-1] < self.high_array[-1]:
            self.line_sar_top[-1] = self.high_array[-1]
        if self.line_sar_buttom[-1] > self.low_array[-1]:
            self.line_sar_buttom[-1] = self.low_array[-1]

        if len(self.line_sar) > self.max_hold_bars:
            del self.line_sar[0]

    # ----------------------------------------------------------------------
    def __count_ma(self):
        """计算K线的MA1 和MA2"""

        if not (self.para_ma1_len > 0 or self.para_ma2_len > 0 or self.para_ma3_len > 0):  # 不计算
            return

        # 1、lineBar满足长度才执行计算
        if self.bar_len < min(7, self.para_ma1_len, self.para_ma2_len, self.para_ma3_len) + 2:
            self.write_log(u'数据未充分,当前Bar数据数量：{0}，计算MA需要：{1}'.
                           format(self.bar_len,
                                  min(7, self.para_ma1_len, self.para_ma2_len, self.para_ma3_len) + 2))
            return

        # 计算第一条MA均线
        if self.para_ma1_len > 0:
            count_len = min(self.para_ma1_len, self.bar_len)
            bar_ma1 = ta.MA(self.close_array[-count_len:], count_len)[-1]
            bar_ma1 = round(float(bar_ma1), self.round_n)

            if len(self.line_ma1) > self.max_hold_bars:
                del self.line_ma1[0]
            self.line_ma1.append(bar_ma1)

            # 计算斜率
            if len(self.line_ma1) > 2 and self.line_ma1[-2] != 0:
                ma1_atan = math.atan((self.line_ma1[-1] / self.line_ma1[-2] - 1) * 100) * 180 / math.pi
                ma1_atan = round(ma1_atan, self.round_n)
                if len(self.line_ma1_atan) > self.max_hold_bars:
                    del self.line_ma1_atan[0]
                self.line_ma1_atan.append(ma1_atan)

        # 计算第二条MA均线
        if self.para_ma2_len > 0:
            count_len = min(self.para_ma2_len, self.bar_len)
            bar_ma2 = ta.MA(self.close_array[-count_len:], count_len)[-1]
            bar_ma2 = round(float(bar_ma2), self.round_n)

            if len(self.line_ma2) > self.max_hold_bars:
                del self.line_ma2[0]
            self.line_ma2.append(bar_ma2)

            # 计算斜率
            if len(self.line_ma2) > 2 and self.line_ma2[-2] != 0:
                ma2_atan = math.atan((self.line_ma2[-1] / self.line_ma2[-2] - 1) * 100) * 180 / math.pi
                ma2_atan = round(ma2_atan, self.round_n)
                if len(self.line_ma2_atan) > self.max_hold_bars:
                    del self.line_ma2_atan[0]
                self.line_ma2_atan.append(ma2_atan)

        # 计算第三条MA均线
        if self.para_ma3_len > 0:
            count_len = min(self.para_ma3_len, self.bar_len)
            bar_ma3 = ta.MA(self.close_array[-count_len:], count_len)[-1]
            bar_ma3 = round(float(bar_ma3), self.round_n)

            if len(self.line_ma3) > self.max_hold_bars:
                del self.line_ma3[0]
            self.line_ma3.append(bar_ma3)

            # 计算斜率
            if len(self.line_ma3) > 2 and self.line_ma3[-2] != 0:
                ma3_atan = math.atan((self.line_ma3[-1] / self.line_ma3[-2] - 1) * 100) * 180 / math.pi
                ma3_atan = round(ma3_atan, self.round_n)
                if len(self.line_ma3_atan) > self.max_hold_bars:
                    del self.line_ma3_atan[0]
                self.line_ma3_atan.append(ma3_atan)

        # 计算MA1，MA2，MA3的金叉死叉
        if len(self.line_ma1) >= 2 and len(self.line_ma2) > 2:
            golden_cross = False
            dead_cross = False
            if self.line_ma1[-1] > self.line_ma1[-2] \
                    and self.line_ma1[-1] > self.line_ma2[-1] \
                    and self.line_ma1[-2] <= self.line_ma2[-2]:
                golden_cross = True
            if self.line_ma1[-1] < self.line_ma1[-2] \
                    and self.line_ma1[-1] < self.line_ma2[-1] \
                    and self.line_ma1[-2] >= self.line_ma2[-2]:
                dead_cross = True
            if self.ma12_count <= 0:
                if golden_cross:
                    self.ma12_count = 1
                elif self.line_ma1[-1] < self.line_ma2[-1]:
                    self.ma12_count -= 1

            elif self.ma12_count >= 0:
                if dead_cross:
                    self.ma12_count = -1
                elif self.line_ma1[-1] > self.line_ma2[-1]:
                    self.ma12_count += 1

        if len(self.line_ma2) >= 2 and len(self.line_ma3) > 2:
            golden_cross = False
            dead_cross = False
            if self.line_ma2[-1] > self.line_ma2[-2] \
                    and self.line_ma2[-1] > self.line_ma3[-1] \
                    and self.line_ma2[-2] <= self.line_ma3[-2]:
                golden_cross = True

            if self.line_ma2[-1] < self.line_ma2[-2] \
                    and self.line_ma2[-1] < self.line_ma3[-1] \
                    and self.line_ma2[-2] >= self.line_ma3[-2]:
                dead_cross = True

            if self.ma23_count <= 0:
                if golden_cross:
                    self.ma23_count = 1
                elif self.line_ma2[-1] < self.line_ma3[-1]:
                    self.ma23_count -= 1

            elif self.ma23_count >= 0:
                if dead_cross:
                    self.ma23_count = -1
                elif self.line_ma2[-1] > self.line_ma3[-1]:
                    self.ma23_count += 1

        if len(self.line_ma1) >= 2 and len(self.line_ma3) > 2:
            golden_cross = False
            dead_cross = False
            if self.line_ma1[-1] > self.line_ma1[-2] \
                    and self.line_ma1[-1] > self.line_ma3[-1] \
                    and self.line_ma1[-2] <= self.line_ma3[-2]:
                golden_cross = True
            if self.line_ma1[-1] < self.line_ma1[-2] \
                    and self.line_ma1[-1] < self.line_ma3[-1] \
                    and self.line_ma1[-2] >= self.line_ma3[-2]:
                dead_cross = True
            if self.ma13_count <= 0:
                if golden_cross:
                    self.ma13_count = 1
                elif self.line_ma1[-1] < self.line_ma3[-1]:
                    self.ma13_count -= 1

            elif self.ma13_count >= 0:
                if dead_cross:
                    self.ma13_count = -1
                elif self.line_ma1[-1] > self.line_ma3[-1]:
                    self.ma13_count += 1

    def rt_count_ma(self):
        """
        实时计算MA得值
        :param ma_num:第几条均线, 1，对应para_ma1_len,,,,
        :return:
        """
        if self.para_ma1_len <=0 and self.para_ma2_len <=0 and self.para_ma3_len <= 0:
            return
        if self.cur_bar:
            rt_close_array = np.append(self.close_array, [self.cur_bar.close_price])
        else:
            rt_close_array = self.close_array

        if self.para_ma1_len > 0:
            count_len = min(self.bar_len, self.para_ma1_len)
            if count_len > 0:
                close_ma_array = ta.MA(rt_close_array[-count_len:], count_len)
                self._rt_ma1 = round(float(close_ma_array[-1]), self.round_n)

                # 计算斜率
                if len(close_ma_array) > 2 and close_ma_array[-2] != 0:
                    self._rt_ma1_atan = round(
                        math.atan((close_ma_array[-1] / close_ma_array[-2] - 1) * 100) * 180 / math.pi, self.round_n)

        if self.para_ma2_len > 0:
            count_len = min(self.bar_len, self.para_ma2_len)
            if count_len > 0:
                close_ma_array = ta.MA(rt_close_array[-count_len:], count_len)
                self._rt_ma2 = round(float(close_ma_array[-1]), self.round_n)

                # 计算斜率
                if len(close_ma_array) > 2 and close_ma_array[-2] != 0:
                    self._rt_ma2_atan = round(
                        math.atan((close_ma_array[-1] / close_ma_array[-2] - 1) * 100) * 180 / math.pi, self.round_n)

        if self.para_ma3_len > 0:
            count_len = min(self.bar_len, self.para_ma3_len)
            if count_len > 0:
                close_ma_array = ta.MA(rt_close_array[-count_len:], count_len)
                self._rt_ma3 = round(float(close_ma_array[-1]), self.round_n)

                # 计算斜率
                if len(close_ma_array) > 2 and close_ma_array[-2] != 0:
                    self._rt_ma3_atan = round(
                        math.atan((close_ma_array[-1] / close_ma_array[-2] - 1) * 100) * 180 / math.pi, self.round_n)

    @property
    def rt_ma1(self):
        self.check_rt_funcs(self.rt_count_ma)
        if self._rt_ma1 is None and len(self.line_ma1) > 0:
            return self.line_ma1[-1]
        return self._rt_ma1

    @property
    def rt_ma2(self):
        self.check_rt_funcs(self.rt_count_ma)
        if self._rt_ma2 is None and len(self.line_ma2) > 0:
            return self.line_ma2[-1]
        return self._rt_ma2

    @property
    def rt_ma3(self):
        self.check_rt_funcs(self.rt_count_ma)
        if self._rt_ma3 is None and len(self.line_ma3) > 0:
            return self.line_ma3[-1]
        return self._rt_ma3

    @property
    def rt_ma1_atan(self):
        self.check_rt_funcs(self.rt_count_ma)
        if self._rt_ma1_atan is None and len(self.line_ma1_atan) > 0:
            return self.line_ma1_atan[-1]
        return self._rt_ma1_atan

    @property
    def rt_ma2_atan(self):
        self.check_rt_funcs(self.rt_count_ma)
        if self._rt_ma2_atan is None and len(self.line_ma2_atan) > 0:
            return self.line_ma2_atan[-1]
        return self._rt_ma2_atan

    @property
    def rt_ma3_atan(self):
        self.check_rt_funcs(self.rt_count_ma)
        if self._rt_ma3_atan is None and len(self.line_ma3_atan) > 0:
            return self.line_ma3_atan[-1]
        return self._rt_ma3_atan

    # ----------------------------------------------------------------------
    def __count_ema(self):
        """计算K线的EMA1 和EMA2"""

        if not (self.para_ema1_len > 0 or self.para_ema2_len > 0 or self.para_ema3_len > 0):  # 不计算
            return

        ema1_data_len = min(self.para_ema1_len * 4, self.para_ema1_len + 40) if self.para_ema1_len > 0 else 0
        ema2_data_len = min(self.para_ema2_len * 4, self.para_ema2_len + 40) if self.para_ema2_len > 0 else 0
        ema3_data_len = min(self.para_ema3_len * 4, self.para_ema3_len + 40) if self.para_ema3_len > 0 else 0
        max_data_len = max(ema1_data_len, ema2_data_len, ema3_data_len)
        # 1、lineBar满足长度才执行计算
        if self.bar_len < max_data_len:
            self.write_log(u'数据未充分,当前Bar数据数量：{0}，计算EMA需要：{1}'.
                           format(len(self.line_bar), max_data_len))
            return

        # 计算第一条EMA均线
        if self.para_ema1_len > 0:
            count_len = min(self.para_ema1_len, self.bar_len)

            # 3、获取前InputN周期(不包含当前周期）的K线
            bar_ema1 = ta.EMA(self.close_array[-ema1_data_len:], count_len)[-1]
            bar_ema1 = round(float(bar_ema1), self.round_n)

            if len(self.line_ema1) > self.max_hold_bars:
                del self.line_ema1[0]
            self.line_ema1.append(bar_ema1)

        # 计算第二条EMA均线
        if self.para_ema2_len > 0:
            count_len = min(self.bar_len, self.para_ema2_len)

            # 3、获取前InputN周期(不包含当前周期）的自适应均线
            bar_ema2 = ta.EMA(self.close_array[-ema2_data_len:], count_len)[-1]
            bar_ema2 = round(float(bar_ema2), self.round_n)

            if len(self.line_ema2) > self.max_hold_bars:
                del self.line_ema2[0]
            self.line_ema2.append(bar_ema2)

        # 计算第三条EMA均线
        if self.para_ema3_len > 0:
            count_len = min(self.bar_len, self.para_ema3_len)

            # 3、获取前InputN周期(不包含当前周期）的自适应均线
            bar_ema3 = ta.EMA(self.close_array[-ema3_data_len:], count_len)[-1]
            bar_ema3 = round(float(bar_ema3), self.round_n)

            if len(self.line_ema3) > self.max_hold_bars:
                del self.line_ema3[0]
            self.line_ema3.append(bar_ema3)

    def __count_ama(self):
        """计算K线的卡夫曼自适应AMA1
        如何测量价格变动的速率。
    采用的方法是，在一定的周期内，计算每个周期价格的变动的累加，用整个周期的总体价格变动除以每个周期价格变动的累加，我们采用这个数字作为价格变化的速率。如果股票持续上涨或下跌，那么变动的速率就是1；如果股票在一定周期内涨跌的幅度为0，那么价格的变动速率就是0。变动速率为1，对应的最快速的均线－2日的EMA；变动速率为0 ，则对应最慢速的均线－30日EMA。
    以通达信软件的公式为例（其他软件也可以用）：
    每个周期价格变动的累加:=sum(abs(close-ref(close,1)),n);
    整个周期价格的总体变动:=abs(close-ref(close,n));
    变动速率:=整个周期价格的总体变动/每个周期价格变动的累加;
    在本文中，一般采用周期n=10。
    ·使用10周期去指定一个从非常慢到非常快的趋势；
    ·在10周期内当价格方向不明确的时候，自适应均线应该是横向移动；
        """

        if self.para_ama_len <= 0:
            return

        # 1、lineBar满足长度才执行计算
        if len(self.line_bar) < self.para_ama_len + 2:
            self.write_log(u'数据未充分,当前Bar数据数量：{0}，计算AMA需要：{1}'.
                           format(len(self.line_bar), self.para_ama_len))
            return

        # 3、para_ema1_len(包含当前周期）的自适应均线
        bar_ama1 = ta.KAMA(self.close_array[-self.para_ama_len:], self.para_ama_len)[-1]
        bar_ama1 = round(float(bar_ama1), self.round_n)

        # 删除多余的数据
        if len(self.line_ama) > self.max_hold_bars:
            del self.line_ama[0]

        # 添加新数据
        self.line_ama.append(bar_ama1)

        # 整个周期价格的总体变动:=abs(close-ref(close,n));
        dir = abs(self.line_bar[-1].close_price - self.line_bar[-self.para_ama_len - 1].close_price)
        vir = 0

        for i in range(len(self.line_bar) - 1, len(self.line_bar) - 1 - self.para_ama_len, -1):  # 周期 para_ama_len
            v1 = abs(self.line_bar[i].close_price - self.line_bar[i - 1].close_price)
            vir = vir + v1

        er = round(float(dir) / vir, self.round_n)

        if len(self.line_ama_er) > self.max_hold_bars:
            del self.line_ama_er[0]

        self.line_ama_er.append(er)

        # 数据长度不足时的计算
        if 1 < len(self.line_ama) < self.para_ama_len * 2:
            # 平均价差变化值，取最后一个和第一个的价差的平均值作为变化值
            self.cur_ama_avg_diff = (self.line_ama[-1] - self.line_ama[0]) / (self.para_ama_len * 2)

            # 前后价差变化值，取平均值
            listPreDiff = [abs(self.line_ama[n] - self.line_ama[n - 1]) for n, x in enumerate(self.line_ama)]
            self.cur_ama_pre_diff = ta.MA(np.array(listPreDiff, dtype=float), timeperiod=self.para_ama_len)[-1]

        elif len(self.line_ama) >= self.para_ama_len * 2:
            # 前后价差
            preDiff = self.line_ama[-1] - self.line_ama[-2]

            # 周期价差变化
            self.cur_ama_avg_diff = self.cur_ama_avg_diff + (preDiff - self.cur_ama_avg_diff) / (self.para_ama_len * 2)
            # 前后价差变化
            self.cur_ama_pre_diff += (abs(preDiff) - self.cur_ama_pre_diff) / (self.para_ama_len * 2)

        if self.cur_ama_pre_diff != 0:
            diffRate = round((1 + self.cur_ama_avg_diff / self.cur_ama_pre_diff) * 50, self.round_n)
        else:
            diffRate = 0

        if len(self.line_ama_diff_rate) > self.max_hold_bars:
            del self.line_ama_diff_rate[0]

        self.line_ama_diff_rate.append(diffRate)

        if len(self.line_ama) > 2:
            mtmDiff = self.line_ama[-1] - self.line_ama[-3]
            mtmRate = round(mtmDiff / (self.height * self.price_tick), self.round_n)

            if len(self.line_ama_mtm_rate) > self.max_hold_bars:
                del self.line_ama_mtm_rate[0]

            self.line_ama_mtm_rate.append(mtmRate)

    def __count_atr(self):
        """计算Mx K线的各类数据和条件"""

        # 1、lineMx满足长度才执行计算
        maxAtrLen = max(self.para_atr1_len, self.para_atr2_len, self.para_atr3_len)

        if maxAtrLen <= 0:  # 不计算
            return

        data_need_len = min(7, maxAtrLen)

        if self.bar_len < data_need_len:
            self.write_log(u'数据未充分,当前Bar数据数量：{0}，计算ATR需要：{1}'.
                           format(self.bar_len, data_need_len))
            return

        # 计算 ATR
        if self.para_atr1_len > 0:
            count_len = min(self.bar_len, self.para_atr1_len)
            self.cur_atr1 = ta.ATR(self.high_array[-count_len:], self.low_array[-count_len:],
                                   self.close_array[-count_len:], count_len)
            self.cur_atr1 = round(self.cur_atr1, self.round_n)
            if len(self.line_atr1) > self.max_hold_bars:
                del self.line_atr1[0]
            self.line_atr1.append(self.cur_atr1)

        if self.para_atr2_len > 0:
            count_len = min(self.bar_len, self.para_atr2_len)
            self.cur_atr2 = ta.ATR(self.high_array[-count_len:], self.low_array[-count_len:],
                                   self.close_array[-count_len:], count_len)
            self.cur_atr2 = round(self.cur_atr2, self.round_n)
            if len(self.line_atr2) > self.max_hold_bars:
                del self.line_atr2[0]
            self.line_atr2.append(self.cur_atr2)

        if self.para_atr3_len > 0:
            count_len = min(self.bar_len, self.para_atr3_len)
            self.cur_atr3 = ta.ATR(self.high_array[-count_len:], self.low_array[-count_len:],
                                   self.close_array[-count_len:], count_len)
            self.cur_atr3 = round(self.cur_atr3, self.round_n)

            if len(self.line_atr3) > self.max_hold_bars:
                del self.line_atr3[0]

            self.line_atr3.append(self.cur_atr3)

    def __count_rsi(self):
        """计算K线的RSI"""

        if self.para_rsi1_len <= 0 and self.para_rsi2_len <= 0:
            return

        # 1、lineBar满足长度才执行计算
        if len(self.line_bar) < self.para_rsi1_len + 2:
            self.write_log(u'数据未充分,当前Bar数据数量：{0}，计算RSI需要：{1}'.
                           format(len(self.line_bar), self.para_rsi1_len + 2))
            return

        # 计算第1根RSI曲线
        # 3、para_rsi1_len(包含当前周期）的相对强弱
        bar_rsi = ta.RSI(self.close_array[-2 * self.para_rsi1_len:], self.para_rsi1_len)[-1]
        bar_rsi = round(float(bar_rsi), self.round_n)

        if len(self.line_rsi1) > self.max_hold_bars:
            del self.line_rsi1[0]

        self.line_rsi1.append(bar_rsi)

        if len(self.line_rsi1) > 3:
            # 峰
            if self.line_rsi1[-1] < self.line_rsi1[-2] and self.line_rsi1[-3] < self.line_rsi1[-2]:
                t = {}
                t["Type"] = u'T'
                t["RSI"] = self.line_rsi1[-2]
                t["Close"] = self.close_array[-1]

                if len(self.rsi_top_list) > self.max_hold_bars:
                    del self.rsi_top_list[0]

                self.rsi_top_list.append(t)
                self.cur_rsi_top_buttom = self.rsi_top_list[-1]

            # 谷
            elif self.line_rsi1[-1] > self.line_rsi1[-2] and self.line_rsi1[-3] > self.line_rsi1[-2]:
                b = {}
                b["Type"] = u'B'
                b["RSI"] = self.line_rsi1[-2]
                b["Close"] = self.close_array[-1]

                if len(self.rsi_buttom_list) > self.max_hold_bars:
                    del self.rsi_buttom_list[0]
                self.rsi_buttom_list.append(b)
                self.cur_rsi_top_buttom = self.rsi_buttom_list[-1]

        # 计算第二根RSI曲线
        if self.para_rsi2_len > 0:
            if self.bar_len < self.para_rsi2_len + 2:
                return

            bar_rsi = ta.RSI(self.close_array[-2 * self.para_rsi2_len:], self.para_rsi2_len)[-1]
            bar_rsi = round(float(bar_rsi), self.round_n)

            if len(self.line_rsi2) > self.max_hold_bars:
                del self.line_rsi2[0]

            self.line_rsi2.append(bar_rsi)

    def __count_cmi(self):
        """市场波动指数（Choppy Market Index，CMI）是一个用来判断市场走势类型的技术分析指标。
        它通过计算当前收盘价与一定周期前的收盘价的差值与这段时间内价格波动的范围的比值，来判断目前的股价走势是趋势还是盘整。
        市场波动指数CMI的计算公式：
        CMI=(Abs(Close-ref(close,(n-1)))*100/(HHV(high,n)-LLV(low,n))
        其中，Abs是绝对值。
        n是周期数，例如３０。
        市场波动指数CMI的使用方法：
        这个指标的重要用途是来区分目前的股价走势类型：盘整，趋势。当CMI指标小于２０时，市场走势是盘整；当CMI指标大于２０时，市场在趋势期。
        CMI指标还可以用于预测股价走势类型的转变。因为物极必反，当CMI长期处于０附近，此时，股价走势很可能从盘整转为趋势；当CMI长期处于１００附近，此时，股价趋势很可能变弱，形成盘整。
        """

        if self.para_cmi_len <= 0:
            return

        # 1、lineBar满足长度才执行计算
        if self.bar_len < self.para_cmi_len:
            self.write_log(u'数据未充分,当前Bar数据数量：{0}，计算CMI需要：{1}'.
                           format(len(self.line_bar), self.para_cmi_len))
            return

        hhv = max(self.close_array[-self.para_cmi_len:])
        llv = min(self.close_array[-self.para_cmi_len:])

        if hhv == llv:
            cmi = 100
        else:
            cmi = abs(self.close_array[-1] - self.close_array[-self.para_cmi_len]) * 100 / (hhv - llv)

        cmi = round(cmi, self.round_n)

        if len(self.line_cmi) > self.max_hold_bars:
            del self.line_cmi[0]

        self.line_cmi.append(cmi)

    def __count_boll(self):
        """布林特线"""
        if not (self.para_boll_len > 0 or self.para_boll2_len > 0):  # 不计算
            return

        if self.para_boll_len > 0:
            if self.bar_len < min(7, self.para_boll_len):
                self.write_log(u'数据未充分,当前Bar数据数量：{0}，计算Boll需要：{1}'.
                               format(len(self.line_bar), min(14, self.para_boll_len) + 1))
            else:
                bollLen = min(self.bar_len, self.para_boll_len)

                # 不包含当前最新的Bar
                upper_list, middle_list, lower_list = ta.BBANDS(self.close_array,
                                                                timeperiod=bollLen, nbdevup=self.para_boll_std_rate,
                                                                nbdevdn=self.para_boll_std_rate, matype=0)
                if len(self.line_boll_upper) > self.max_hold_bars:
                    del self.line_boll_upper[0]
                if len(self.line_boll_middle) > self.max_hold_bars:
                    del self.line_boll_middle[0]
                if len(self.line_boll_lower) > self.max_hold_bars:
                    del self.line_boll_lower[0]
                if len(self.line_boll_std) > self.max_hold_bars:
                    del self.line_boll_std[0]

                # 1标准差
                std = (upper_list[-1] - lower_list[-1]) / (self.para_boll_std_rate * 2)
                self.line_boll_std.append(std)

                upper = round(upper_list[-1], self.round_n)
                self.line_boll_upper.append(upper)  # 上轨
                self.cur_upper = round_to(upper, self.price_tick)  # 上轨取整

                middle = round(middle_list[-1], self.round_n)
                self.line_boll_middle.append(middle)  # 中轨
                self.last_middle = round_to(middle, self.price_tick)  # 中轨取整

                lower = round(lower_list[-1], self.round_n)
                self.line_boll_lower.append(lower)  # 下轨
                self.last_lower = round_to(lower, self.price_tick)  # 下轨取整

                # 计算斜率
                if len(self.line_boll_upper) > 2 and self.line_boll_upper[-2] != 0:
                    up_atan = math.atan((self.line_boll_upper[-1] / self.line_boll_upper[-2] - 1) * 100) * 180 / math.pi
                    up_atan = round(up_atan, self.round_n)
                    if len(self.line_upper_atan) > self.max_hold_bars:
                        del self.line_upper_atan[0]
                    self.line_upper_atan.append(up_atan)
                if len(self.line_boll_middle) > 2 and self.line_boll_middle[-2] != 0:
                    mid_atan = math.atan(
                        (self.line_boll_middle[-1] / self.line_boll_middle[-2] - 1) * 100) * 180 / math.pi
                    mid_atan = round(mid_atan, self.round_n)
                    if len(self.line_middle_atan) > self.max_hold_bars:
                        del self.line_middle_atan[0]
                    self.line_middle_atan.append(mid_atan)
                if len(self.line_boll_lower) > 2 and self.line_boll_lower[-2] != 0:
                    low_atan = math.atan(
                        (self.line_boll_lower[-1] / self.line_boll_lower[-2] - 1) * 100) * 180 / math.pi
                    low_atan = round(low_atan, self.round_n)
                    if len(self.line_lower_atan) > self.max_hold_bars:
                        del self.line_lower_atan[0]
                    self.line_lower_atan.append(low_atan)

        if self.para_boll2_len > 0:
            if self.bar_len < min(14, self.para_boll2_len) + 1:
                self.write_log(u'数据未充分,当前Bar数据数量：{0}，计算Boll2需要：{1}'.
                               format(len(self.line_bar), min(14, self.para_boll2_len) + 1))
            else:
                boll2Len = min(self.bar_len, self.para_boll2_len)

                # 不包含当前最新的Bar
                upper_list, middle_list, lower_list = ta.BBANDS(self.close_array[-2 * self.para_boll2_len],
                                                                timeperiod=boll2Len, nbdevup=self.para_boll2_std_rate,
                                                                nbdevdn=self.para_boll2_std_rate, matype=0)
                if len(self.line_boll2_upper) > self.max_hold_bars:
                    del self.line_boll2_upper[0]
                if len(self.line_boll2_middle) > self.max_hold_bars:
                    del self.line_boll2_middle[0]
                if len(self.line_boll2_lower) > self.max_hold_bars:
                    del self.line_boll2_lower[0]
                if len(self.line_boll2_std) > self.max_hold_bars:
                    del self.line_boll2_std[0]

                # 1标准差
                std = (upper_list[-1] - lower_list[-1]) / (self.para_boll2_std_rate * 2)
                self.line_boll2_std.append(std)

                upper = round(upper_list[-1], self.round_n)
                self.line_boll2_upper.append(upper)  # 上轨
                self.cur_upper2 = round_to(upper, self.price_tick)  # 上轨取整

                middle = round(middle_list[-1], self.round_n)
                self.line_boll2_middle.append(middle)  # 中轨
                self.cur_middle2 = round_to(middle, self.price_tick)  # 中轨取整

                lower = round(lower_list[-1], self.round_n)
                self.line_boll2_lower.append(lower)  # 下轨
                self.cur_lower2 = round_to(lower, self.price_tick)  # 下轨取整

                # 计算斜率
                if len(self.line_boll2_upper) > 2 and self.line_boll2_upper[-2] != 0:
                    up_atan = math.atan(
                        (self.line_boll2_upper[-1] / self.line_boll2_upper[-2] - 1) * 100) * 180 / math.pi
                    up_atan = round(up_atan, self.round_n)
                    if len(self.line_upper2_atan) > self.max_hold_bars:
                        del self.line_upper2_atan[0]
                    self.line_upper2_atan.append(up_atan)
                if len(self.line_boll2_middle) > 2 and self.line_boll2_middle[-2] != 0:
                    mid_atan = math.atan(
                        (self.line_boll2_middle[-1] / self.line_boll2_middle[-2] - 1) * 100) * 180 / math.pi
                    mid_atan = round(mid_atan, self.round_n)
                    if len(self.line_middle2_atan) > self.max_hold_bars:
                        del self.line_middle2_atan[0]
                    self.line_middle2_atan.append(mid_atan)
                if len(self.line_boll2_lower) > 2 and self.line_boll2_lower[-2] != 0:
                    low_atan = math.atan(
                        (self.line_boll2_lower[-1] / self.line_boll2_lower[-2] - 1) * 100) * 180 / math.pi
                    low_atan = round(low_atan, self.round_n)
                    if len(self.line_lower2_atan) > self.max_hold_bars:
                        del self.line_lower2_atan[0]
                    self.line_lower2_atan.append(low_atan)

    def rt_count_boll(self):
        """实时计算布林上下轨，斜率"""

        if not (self.para_boll_len > 0 or self.para_boll2_len > 0):  # 不计算
            return

        rt_close_array = np.append(self.close_array, [self.cur_bar.close_price])
        if self.para_boll_len > 0:
            bollLen = min(self.para_boll_len, self.bar_len)

            upper_list, middle_list, lower_list = ta.BBANDS(rt_close_array,
                                                            timeperiod=bollLen, nbdevup=self.para_boll_std_rate,
                                                            nbdevdn=self.para_boll_std_rate, matype=0)

            self._rt_middle = round(middle_list[-1], self.round_n)
            self._rt_upper = round(upper_list[-1], self.round_n)
            self._rt_lower = round(lower_list[-1], self.round_n)

            # 计算斜率
            if len(self.line_boll_upper) > 2 and self.line_boll_upper[-1] != 0:
                up_atan = math.atan((self._rt_upper / self.line_boll_upper[-1] - 1) * 100) * 180 / math.pi
                self._rt_upper_atan = round(up_atan, self.round_n)

            if len(self.line_boll_middle) > 2 and self.line_boll_middle[-1] != 0:
                mid_atan = math.atan((self._rt_middle / self.line_boll_middle[-1] - 1) * 100) * 180 / math.pi
                self._rt_middle_atan = round(mid_atan, self.round_n)

            if len(self.line_boll_lower) > 2 and self.line_boll_lower[-1] != 0:
                low_atan = math.atan((self._rt_lower / self.line_boll_lower[-1] - 1) * 100) * 180 / math.pi
                self._rt_lower_atan = round(low_atan, self.round_n)

        if self.para_boll2_len > 0:
            bollLen = min(self.para_boll2_len, self.bar_len)
            upper_list, middle_list, lower_list = ta.BBANDS(rt_close_array,
                                                            timeperiod=bollLen, nbdevup=self.para_boll2_std_rate,
                                                            nbdevdn=self.para_boll2_std_rate, matype=0)
            self._rt_middle2 = round(middle_list[-1], self.round_n)
            self._rt_upper2 = round(upper_list[-1], self.round_n)
            self._rt_lower2 = round(lower_list[-1], self.round_n)

            # 计算斜率
            if len(self.line_boll2_upper) > 2 and self.line_boll2_upper[-1] != 0:
                up_atan = math.atan((self._rt_upper2 / self.line_boll2_upper[-1] - 1) * 100) * 180 / math.pi
                self._rt_upper2_atan = round(up_atan, self.round_n)

            if len(self.line_boll2_middle) > 2 and self.line_boll2_middle[-1] != 0:
                mid_atan = math.atan((self._rt_middle2 / self.line_boll2_middle[-1] - 1) * 100) * 180 / math.pi
                self._rt_middle2_atan = round(mid_atan, self.round_n)

            if len(self.line_boll2_lower) > 2 and self.line_boll2_lower[-1] != 0:
                low_atan = math.atan((self._rt_lower2 / self.line_boll2_lower[-1] - 1) * 100) * 180 / math.pi
                self._rt_lower2_atan = round(low_atan, self.round_n)

    def check_rt_funcs(self, func):
        """检查调用函数名是否在实时计算函数清单中，如果没有，则添加并运行"""
        if func not in self.rt_funcs:
            self.write_log(u'{}添加{}到实时函数中'.format(self.name, str(func.__name__)))
            self.rt_funcs.add(func)
            func()

    @property
    def rt_upper(self):
        self.check_rt_funcs(self.rt_count_boll)
        if self._rt_upper is None and len(self.line_boll_upper) > 0:
            return self.line_boll_upper[-1]
        return self._rt_upper

    @property
    def rt_middle(self):
        self.check_rt_funcs(self.rt_count_boll)
        if self._rt_middle is None and len(self.line_boll_middle) > 0:
            return self.line_boll_middle[-1]
        return self._rt_middle

    @property
    def rt_lower(self):
        self.check_rt_funcs(self.rt_count_boll)
        if self._rt_lower is None and len(self.line_boll_lower) > 0:
            return self.line_boll_lower[-1]
        return self._rt_lower

    @property
    def rt_upper_atan(self):
        self.check_rt_funcs(self.rt_count_boll)
        if self._rt_upper_atan is None and len(self.line_upper_atan) > 0:
            return self.line_upper_atan[-1]
        return self._rt_upper_atan

    @property
    def rt_middle_atan(self):
        self.check_rt_funcs(self.rt_count_boll)
        if self._rt_middle_atan is None and len(self.line_middle_atan) > 0:
            return self.line_middle_atan[-1]
        return self._rt_middle_atan

    @property
    def rt_lower_atan(self):
        self.check_rt_funcs(self.rt_count_boll)
        if self._rt_lower_atan is None and len(self.line_lower_atan) > 0:
            return self.line_lower_atan[-1]
        return self._rt_lower_atan

    @property
    def rt_upper2(self):
        self.check_rt_funcs(self.rt_count_boll)
        if self._rt_upper2 is None and len(self.line_boll2_upper) > 0:
            return self.line_boll2_upper[-1]
        return self._rt_upper2

    @property
    def rt_middle2(self):
        self.check_rt_funcs(self.rt_count_boll)
        if self._rt_middle2 is None and len(self.line_boll2_middle) > 0:
            return self.line_boll2_middle[-1]
        return self._rt_middle2

    @property
    def rt_lower2(self):
        self.check_rt_funcs(self.rt_count_boll)
        if self._rt_lower2 is None and len(self.line_boll2_lower) > 0:
            return self.line_boll2_lower[-1]
        return self._rt_lower2

    @property
    def rt_upper2_atan(self):
        self.check_rt_funcs(self.rt_count_boll)
        if self._rt_upper2_atan is None and len(self.line_upper2_atan) > 0:
            return self.line_upper2_atan[-1]
        return self._rt_upper2_atan

    @property
    def rt_middle2_atan(self):
        self.check_rt_funcs(self.rt_count_boll)
        if self._rt_middle2_atan is None and len(self.line_middle2_atan) > 0:
            return self.line_middle2_atan[-1]
        return self._rt_middle2_atan

    @property
    def rt_lower2_atan(self):
        self.check_rt_funcs(self.rt_count_boll)
        if self._rt_lower2_atan is None and len(self.line_lower2_atan) > 0:
            return self.line_lower2_atan[-1]
        return self._rt_lower2_atan

    def __count_kdj(self):
        """KDJ指标"""
        """
        KDJ指标的中文名称又叫随机指标，是一个超买超卖指标,最早起源于期货市场，由乔治·莱恩（George Lane）首创。
        随机指标KDJ最早是以KD指标的形式出现，而KD指标是在威廉指标的基础上发展起来的。
        不过KD指标只判断股票的超买超卖的现象，在KDJ指标中则融合了移动平均线速度上的观念，形成比较准确的买卖信号依据。在实践中，K线与D线配合J线组成KDJ指标来使用。
        KDJ指标在设计过程中主要是研究最高价、最低价和收盘价之间的关系，同时也融合了动量观念、强弱指标和移动平均线的一些优点。
        因此，能够比较迅速、快捷、直观地研判行情，被广泛用于股市的中短期趋势分析，是期货和股票市场上最常用的技术分析工具。
 
        第一步 计算RSV：即未成熟随机值（Raw Stochastic Value）。
        RSV 指标主要用来分析市场是处于“超买”还是“超卖”状态：
            - RSV高于80%时候市场即为超买状况，行情即将见顶，应当考虑出仓；
            - RSV低于20%时候，市场为超卖状况，行情即将见底，此时可以考虑加仓。
        N日RSV=(N日收盘价-N日内最低价）÷(N日内最高价-N日内最低价）×100%
        第二步 计算K值：当日K值 = 2/3前1日K值 + 1/3当日RSV ; 
        第三步 计算D值：当日D值 = 2/3前1日D值 + 1/3当日K值； 
        第四步 计算J值：当日J值 = 3当日K值 - 2当日D值. 
        """

        if self.para_kdj_len <= 0:
            return

        if len(self.line_bar) < self.para_kdj_len + 1:
            self.write_log(u'数据未充分,当前Bar数据数量：{0}，计算KDJ需要：{1}'.format(len(self.line_bar), self.para_kdj_len + 1))
            return

        if self.para_kdj_slow_len == 0:
            self.para_kdj_slow_len = 3
        if self.para_kdj_smooth_len == 0:
            self.para_kdj_smooth_len = 3

        para_kdj_len = min(self.para_kdj_len, self.bar_len)

        hhv = max(self.high_array[-para_kdj_len:])
        llv = min(self.low_array[-para_kdj_len:])

        if len(self.line_k) > 0:
            last_k = self.line_k[-1]
        else:
            last_k = 0

        if len(self.line_d) > 0:
            last_d = self.line_d[-1]
        else:
            last_d = 0

        if hhv == llv:
            rsv = 50
        else:
            rsv = (self.close_array[-1] - llv) / (hhv - llv) * 100

        self.line_rsv.append(rsv)

        k = (self.para_kdj_slow_len - 1) * last_k / self.para_kdj_slow_len + rsv / self.para_kdj_slow_len
        if k < 0:
            k = 0
        if k > 100:
            k = 100

        d = (self.para_kdj_smooth_len - 1) * last_d / self.para_kdj_smooth_len + k / self.para_kdj_smooth_len
        if d < 0:
            d = 0
        if d > 100:
            d = 100

        j = self.para_kdj_smooth_len * k - (self.para_kdj_smooth_len - 1) * d

        if len(self.line_k) > self.max_hold_bars:
            del self.line_k[0]
        self.line_k.append(k)

        if len(self.line_d) > self.max_hold_bars:
            del self.line_d[0]
        self.line_d.append(d)

        if len(self.line_j) > self.max_hold_bars:
            del self.line_j[0]
        self.line_j.append(j)

        # 增加KDJ的J谷顶和波底
        if len(self.line_j) > 3:
            # 峰
            if self.line_j[-1] < self.line_j[-2] and self.line_j[-3] <= self.line_j[-2]:
                t = {
                    'Type': 'T',
                    'J': self.line_j[-2],
                    'Close': self.close_array[-1]}

                if len(self.kdj_top_list) > self.max_hold_bars:
                    del self.kdj_top_list[0]

                self.kdj_top_list.append(t)
                self.cur_kdj_top_buttom = self.kdj_top_list[-1]

            # 谷
            elif self.line_j[-1] > self.line_j[-2] and self.line_j[-3] >= self.line_j[-2]:
                b = {
                    'Type': u'B',
                    'J': self.line_j[-2],
                    'Close': self.close_array[-1]
                }
                if len(self.kdj_buttom_list) > self.max_hold_bars:
                    del self.kdj_buttom_list[0]
                self.kdj_buttom_list.append(b)
                self.cur_kdj_top_buttom = self.kdj_buttom_list[-1]

        self.__update_kdj_cross()

    def __count_kdj_tb(self):
        """KDJ指标"""
        """
        KDJ指标的中文名称又叫随机指标，是一个超买超卖指标,最早起源于期货市场，由乔治·莱恩（George Lane）首创。
        随机指标KDJ最早是以KD指标的形式出现，而KD指标是在威廉指标的基础上发展起来的。
        不过KD指标只判断股票的超买超卖的现象，在KDJ指标中则融合了移动平均线速度上的观念，形成比较准确的买卖信号依据。在实践中，K线与D线配合J线组成KDJ指标来使用。
        KDJ指标在设计过程中主要是研究最高价、最低价和收盘价之间的关系，同时也融合了动量观念、强弱指标和移动平均线的一些优点。
        因此，能够比较迅速、快捷、直观地研判行情，被广泛用于股市的中短期趋势分析，是期货和股票市场上最常用的技术分析工具。
 
        第一步 计算RSV：即未成熟随机值（Raw Stochastic Value）。
        RSV 指标主要用来分析市场是处于“超买”还是“超卖”状态：
            - RSV高于80%时候市场即为超买状况，行情即将见顶，应当考虑出仓；
            - RSV低于20%时候，市场为超卖状况，行情即将见底，此时可以考虑加仓。
        N日RSV=(N日收盘价-N日内最低价）÷(N日内最高价-N日内最低价）×100%
        第二步 计算K值：当日K值 = 2/3前1日K值 + 1/3当日RSV ; 
        第三步 计算D值：当日D值 = 2/3前1日D值 + 1/3当日K值； 
        第四步 计算J值：当日J值 = 3当日K值 - 2当日D值. 

        """
        if self.para_kdj_tb_len <= 0:
            return

        if self.para_kdj_tb_len + self.para_kdj_smooth_len > self.max_hold_bars:
            self.max_hold_bars = self.para_kdj_tb_len + self.para_kdj_smooth_len + 1

        # if len(self.lineBar) < self.para_kdj_tb_len + 1:
        #     if not countInBar:
        #         self.write_log(u'数据未充分,当前Bar数据数量：{0}，计算KDJ需要：{1}'.format(len(self.lineBar), self.para_kdj_tb_len + 1))
        #     return

        if self.para_kdj_slow_len == 0:
            self.para_kdj_slow_len = 3
        if self.para_kdj_smooth_len == 0:
            self.para_kdj_smooth_len = 3

        if len(self.line_bar) < 3:
            return

        data_len = min(self.bar_len, self.para_kdj_tb_len)

        hhv = max(self.high_array[-data_len:])
        llv = min(self.low_array[-data_len:])

        if len(self.line_k) > 0:
            last_k = self.line_k[-1]
        else:
            last_k = 0

        if len(self.line_d) > 0:
            last_d = self.line_d[-1]
        else:
            last_d = 0

        if hhv == llv:
            rsv = 50
        else:
            rsv = (self.close_array[-1] - llv) / (hhv - llv) * 100

        self.line_rsv.append(rsv)

        k = (self.para_kdj_slow_len - 1) * last_k / self.para_kdj_slow_len + rsv / self.para_kdj_slow_len
        if k < 0:
            k = 0
        if k > 100:
            k = 100

        d = (self.para_kdj_smooth_len - 1) * last_d / self.para_kdj_smooth_len + k / self.para_kdj_smooth_len
        if d < 0:
            d = 0
        if d > 100:
            d = 100

        j = self.para_kdj_smooth_len * k - (self.para_kdj_smooth_len - 1) * d

        if len(self.line_k) > self.max_hold_bars:
            del self.line_k[0]
        self.line_k.append(k)

        if len(self.line_d) > self.max_hold_bars:
            del self.line_d[0]
        self.line_d.append(d)

        if len(self.line_j) > self.max_hold_bars:
            del self.line_j[0]
        self.line_j.append(j)

        # 增加KDJ的J谷顶和波底
        if len(self.line_j) > 3:
            # 峰
            if self.line_j[-1] < self.line_j[-2] and self.line_j[-3] <= self.line_j[-2]:
                t = {
                    'Type': 'T',
                    'J': self.line_j[-2],
                    'Close': self.close_array[-1]
                }
                if len(self.kdj_top_list) > self.max_hold_bars:
                    del self.kdj_top_list[0]

                self.kdj_top_list.append(t)
                self.cur_kdj_top_buttom = self.kdj_top_list[-1]

            # 谷
            elif self.line_j[-1] > self.line_j[-2] and self.line_j[-3] >= self.line_j[-2]:

                b = {
                    'Type': 'B',
                    'J': self.line_j[-2],
                    'Close': self.close_array
                }
                if len(self.kdj_buttom_list) > self.max_hold_bars:
                    del self.kdj_buttom_list[0]
                self.kdj_buttom_list.append(b)
                self.cur_kdj_top_buttom = self.kdj_buttom_list[-1]

        self.__update_kdj_cross()

    def __update_kdj_cross(self):
        """更新KDJ金叉死叉"""
        if len(self.line_k) < 2 or len(self.line_d) < 2:
            return

        # K值大于D值
        if self.line_k[-1] > self.line_d[-1]:
            if self.line_k[-2] > self.line_d[-2]:
                # 延续金叉
                self.cur_kd_count = max(1, self.cur_kd_count) + 1
            else:
                # 发生金叉
                self.cur_kd_count = 1
                self.cur_kd_cross = round((self.line_k[-1] + self.line_k[-2]) / 2, self.round_n)
                self.cur_kd_cross_price = self.cur_price

        # K值小于D值
        else:
            if self.line_k[-2] < self.line_d[-2]:
                # 延续死叉
                self.cur_kd_count = min(-1, self.cur_kd_count) - 1
            else:
                # 发生死叉
                self.cur_kd_count = -1
                self.cur_kd_cross = round((self.line_k[-1] + self.line_k[-2]) / 2, self.round_n)
                self.cur_kd_cross_price =self.cur_price

    def __count_macd(self):
        """
        Macd计算方法：
        12日EMA的计算：EMA12 = 前一日EMA12 X 11/13 + 今日收盘 X 2/13
        26日EMA的计算：EMA26 = 前一日EMA26 X 25/27 + 今日收盘 X 2/27
        差离值（DIF）的计算： DIF = EMA12 - EMA26，即为talib-MACD返回值macd
        根据差离值计算其9日的EMA，即离差平均值，是所求的DEA值。
        今日DEA = （前一日DEA X 8/10 + 今日DIF X 2/10），即为talib-MACD返回值signal
        DIF与它自己的移动平均之间差距的大小一般BAR=（DIF-DEA)*2，即为MACD柱状图。
        但是talib中MACD的计算是bar = (dif-dea)*1
        """

        if self.para_macd_fast_len <= 0 or self.para_macd_slow_len <= 0 or self.para_macd_signal_len <= 0:
            return

        max_len = max(self.para_macd_fast_len, self.para_macd_slow_len) + self.para_macd_signal_len + 1

        if self.bar_len < max_len:
            self.write_log(u'数据未充分,当前Bar数据数量：{0}，计算MACD需要：{1}'.format(len(self.line_bar), max_len))
            return

        dif_list, dea_list, macd_list = ta.MACD(self.close_array[-max_len * 3:], fastperiod=self.para_macd_fast_len,
                                                slowperiod=self.para_macd_slow_len,
                                                signalperiod=self.para_macd_signal_len)

        if len(self.line_dif) > self.max_hold_bars:
            del self.line_dif[0]
        self.line_dif.append(round(dif_list[-1], self.round_n))

        if len(self.line_dea) > self.max_hold_bars:
            del self.line_dea[0]
        self.line_dea.append(round(dea_list[-1], self.round_n))

        if len(self.line_macd) > self.max_hold_bars:
            del self.line_macd[0]
        self.line_macd.append(round(macd_list[-1] * 2, self.round_n))  # 国内一般是2倍

        # 更新 “段”（金叉-》死叉；或 死叉-》金叉)
        segment = self.macd_segment_list[-1] if len(self.macd_segment_list) > 0 else {}

        # 创建新的段
        if (self.line_macd[-1] > 0 and self.cur_macd_count <= 0) or \
                (self.line_macd[-1] < 0 and self.cur_macd_count >= 0):
            segment = {}
            # 金叉/死叉，更新位置&价格
            self.cur_macd_count, self.rt_macd_count = (1, 1) if self.line_macd[-1] > 0 else (-1, -1)
            self.cur_macd_cross = round((self.line_dif[-1] + self.line_dea[-1]) / 2, self.round_n)
            self.cur_macd_cross_price = self.close_array[-1]
            self.rt_macd_cross = self.cur_macd_cross
            self.rt_macd_cross_price = self.cur_macd_cross_price
            # 更新段
            segment.update({
                'macd_count': self.cur_macd_count,
                'max_price': self.high_array[-1],
                'min_price': self.low_array[-1],
                'max_dif': self.line_dif[-1],
                'min_dif': self.line_dif[-1],
                'macd_area': abs(self.line_macd[-1])})
            self.macd_segment_list.append(segment)

            # 新得能量柱>0，判断是否有底背离，同时，取消原有顶背离
            if self.line_macd[-1] > 0:
                self.dif_buttom_divergence = self.is_dif_divergence(direction=Direction.SHORT)
                self.macd_buttom_divergence = self.is_macd_divergence(direction=Direction.SHORT)
                self.dif_top_divergence = False
                self.macd_top_divergence = False

            # 新得能量柱<0，判断是否有顶背离，同时，取消原有底背离
            elif self.line_macd[-1] < 0:
                self.dif_buttom_divergence = False
                self.macd_buttom_divergence = False
                self.dif_top_divergence = self.is_dif_divergence(direction=Direction.LONG)
                self.macd_top_divergence = self.is_macd_divergence(direction=Direction.LONG)

        else:
            # 继续金叉
            if self.line_macd[-1] > 0 and self.cur_macd_count > 0:
                self.cur_macd_count += 1

                segment.update({
                    'macd_count': self.cur_macd_count,
                    'max_price': max(segment.get('max_price', self.high_array[-1]), self.high_array[-1]),
                    'min_price': min(segment.get('min_price', self.low_array[-1]), self.low_array[-1]),
                    'max_dif': max(segment.get('max_dif', self.line_dif[-1]), self.line_dif[-1]),
                    'min_dif': min(segment.get('min_dif', self.line_dif[-1]), self.line_dif[-1]),
                    'macd_area': segment.get('macd_area', 0) + abs(self.line_macd[-1])
                })
                # 取消实时得记录
                self.rt_macd_count = 0
                self.rt_macd_cross = 0
                self.rt_macd_cross_price = 0

            # 继续死叉
            elif self.line_macd[-1] > 0 and self.cur_macd_count > 0:
                self.cur_macd_count -= 1
                segment.update({
                    'macd_count': self.cur_macd_count,
                    'max_price': max(segment.get('max_price', self.high_array[-1]), self.high_array[-1]),
                    'min_price': min(segment.get('min_price', self.low_array[-1]), self.low_array[-1]),
                    'max_dif': max(segment.get('max_dif', self.line_dif[-1]), self.line_dif[-1]),
                    'min_dif': min(segment.get('min_dif', self.line_dif[-1]), self.line_dif[-1]),
                    'macd_area': segment.get('macd_area', 0) + abs(self.line_macd[-1])
                })
                # 取消实时得记录
                self.rt_macd_count = 0
                self.rt_macd_cross = 0
                self.rt_macd_cross_price = 0

        # 删除超过10个的macd段
        if len(self.macd_segment_list) > 10:
            self.macd_segment_list.pop(0)

    def rt_count_macd(self):
        """
        (实时）Macd计算方法：
        12日EMA的计算：EMA12 = 前一日EMA12 X 11/13 + 今日收盘 X 2/13
        26日EMA的计算：EMA26 = 前一日EMA26 X 25/27 + 今日收盘 X 2/27
        差离值（DIF）的计算： DIF = EMA12 - EMA26，即为talib-MACD返回值macd
        根据差离值计算其9日的EMA，即离差平均值，是所求的DEA值。
        今日DEA = （前一日DEA X 8/10 + 今日DIF X 2/10），即为talib-MACD返回值signal
        DIF与它自己的移动平均之间差距的大小一般BAR=（DIF-DEA)*2，即为MACD柱状图。
        但是talib中MACD的计算是bar = (dif-dea)*1
        """

        if self.para_macd_fast_len <= 0 or self.para_macd_slow_len <= 0 or self.para_macd_signal_len <= 0:
            return

        maxLen = max(self.para_macd_fast_len, self.para_macd_slow_len) + self.para_macd_signal_len + 1

        # maxLen = maxLen * 3             # 注：数据长度需要足够，才能准确。测试过，3倍长度才可以与国内的文华等软件一致

        if self.bar_len < maxLen:
            self.write_log(u'数据未充分,当前Bar数据数量：{0}，计算MACD需要：{1}'.format(len(self.line_bar), maxLen))
            return

        dif_list, dea_list, macd_list = ta.MACD(self.close_array[-3 * maxLen:], fastperiod=self.para_macd_fast_len,
                                                slowperiod=self.para_macd_slow_len,
                                                signalperiod=self.para_macd_signal_len)

        if len(self.line_dif) > self.max_hold_bars:
            del self.line_dif[0]
        self.line_dif.append(round(dif_list[-1], self.round_n))

        if len(self.line_dea) > self.max_hold_bars:
            del self.line_dea[0]
        self.line_dea.append(round(dea_list[-1], self.round_n))

        if len(self.line_macd) > self.max_hold_bars:
            del self.line_macd[0]
        self.line_macd.append(round(macd_list[-1] * 2, self.round_n))  # 国内一般是2倍

    @property
    def rt_dif(self):
        self.check_rt_funcs(self.rt_count_macd)
        if self._rt_dif is None and len(self.line_dif) > 0:
            return self.line_dif[-1]
        return self._rt_dif

    @property
    def rt_dea(self):
        self.check_rt_funcs(self.rt_count_macd)
        if self._rt_dea is None and len(self.line_dea) > 0:
            return self.line_dea[-1]
        return self._rt_dea

    @property
    def rt_macd(self):
        self.check_rt_funcs(self.rt_count_macd)
        if self._rt_macd is None and len(self.line_macd) > 0:
            return self.line_macd[-1]
        return self._rt_macd

    def is_dif_divergence(self, direction):
        """
        检查MACD DIF是否与价格有背离
        :param: direction，多：检查是否有顶背离，空，检查是否有底背离
        """
        s1, s2 = None, None  # s1,倒数的一个匹配段；s2，倒数第二个匹配段
        for seg in reversed(self.macd_segment_list):

            if direction == Direction.LONG:
                if seg.get('macd_count', 0) > 0:
                    if s1 is None:
                        s1 = seg
                        continue
                    elif s2 is None:
                        s2 = seg
                        break
            else:
                if seg.get('macd_count', 0) < 0:
                    if s1 is None:
                        s1 = seg
                        continue
                    elif s2 is None:
                        s2 = seg
                        break

        if not all([s1, s2]):
            return False

        if direction == Direction.LONG:
            s1_max_price = s1.get('max_price', None)
            s2_max_price = s2.get('max_price', None)
            s1_dif_max = s1.get('max_dif', None)
            s2_dif_max = s2.get('max_dif', None)
            if s1_max_price is None or s2_max_price is None or s1_dif_max is None and s2_dif_max is None:
                return False

            # 顶背离，只能在零轴上方才判断
            if s1_dif_max < 0 or s2_dif_max < 0:
                return False

            # 价格创新高（超过前高得0.99）；dif指标没有创新高
            if s1_max_price >= s2_max_price * 0.99 and s1_dif_max < s2_dif_max:
                return True

        if direction == Direction.SHORT:
            s1_min_price = s1.get('min_price', None)
            s2_min_price = s2.get('min_price', None)
            s1_dif_min = s1.get('min_dif', None)
            s2_dif_min = s2.get('min_dif', None)
            if s1_min_price is None or s2_min_price is None or s1_dif_min is None and s2_dif_min is None:
                return False

            # 底部背离，只能在零轴下方才判断
            if s1_dif_min > 0 or s1_dif_min > 0:
                return False

            # 价格创新低，dif没有创新低
            if s1_min_price <= s2_min_price * 1.01 and s1_dif_min > s2_dif_min:
                return True

        return False

    def is_macd_divergence(self, direction):
        """
        检查MACD 能量柱是否与价格有背离
        :param: direction，多：检查是否有顶背离，空，检查是否有底背离
        """
        s1, s2 = None, None  # s1,倒数的一个匹配段；s2，倒数第二个匹配段
        for seg in reversed(self.macd_segment_list):

            if direction == Direction.LONG:
                if seg.get('macd_count', 0) > 0:
                    if s1 is None:
                        s1 = seg
                        continue
                    elif s2 is None:
                        s2 = seg
                        break
            else:
                if seg.get('macd_count', 0) < 0:
                    if s1 is None:
                        s1 = seg
                        continue
                    elif s2 is None:
                        s2 = seg
                        break

        if not all([s1, s2]):
            return False

        if direction == Direction.LONG:
            s1_max_price = s1.get('max_price', None)
            s2_max_price = s2.get('max_price', None)
            s1_area = s1.get('macd_area', None)
            s2_area = s2.get('macd_area', None)
            if s1_max_price is None or s2_max_price is None or s1_area is None and s2_area is None:
                return False

            # 价格创新高（超过前高得0.99）；MACD能量柱没有创更大面积
            if s1_max_price >= s2_max_price * 0.99 and s1_area < s2_area:
                return True

        if direction == Direction.SHORT:
            s1_min_price = s1.get('min_price', None)
            s2_min_price = s2.get('min_price', None)
            s1_area = s1.get('macd_area', None)
            s2_area = s2.get('macd_area', None)
            if s1_min_price is None or s2_min_price is None or s1_area is None and s2_area is None:
                return False

            # 价格创新低，MACD能量柱没有创更大面积
            if s1_min_price <= s2_min_price * 1.01 and s1_area < s2_area:
                return True

        return False

    def __count_cci(self):
        """CCI计算
        顺势指标又叫CCI指标，CCI指标是美国股市技术分析 家唐纳德·蓝伯特(Donald Lambert)于20世纪80年代提出的，专门测量股价、外汇或者贵金属交易
        是否已超出常态分布范围。属于超买超卖类指标中较特殊的一种。波动于正无穷大和负无穷大之间。但是，又不需要以0为中轴线，这一点也和波动于正无穷大
        和负无穷大的指标不同。
        它最早是用于期货市场的判断，后运用于股票市场的研判，并被广泛使用。与大多数单一利用股票的收盘价、开盘价、最高价或最低价而发明出的各种技术分析
        指标不同，CCI指标是根据统计学原理，引进价格与固定期间的股价平均区间的偏离程度的概念，强调股价平均绝对偏差在股市技术分析中的重要性，是一种比
        较独特的技术指标。
        它与其他超买超卖型指标又有自己比较独特之处。象KDJ、W%R等大多数超买超卖型指标都有“0-100”上下界限，因此，它们对待一般常态行情的研判比较适用
        ，而对于那些短期内暴涨暴跌的股票的价格走势时，就可能会发生指标钝化的现象。而CCI指标却是波动于正无穷大到负无穷大之间，因此不会出现指标钝化现
        象，这样就有利于投资者更好地研判行情，特别是那些短期内暴涨暴跌的非常态行情。
        http://baike.baidu.com/view/53690.htm?fromtitle=CCI%E6%8C%87%E6%A0%87&fromid=4316895&type=syn
        """

        if self.para_cci_len <= 0:
            return

        # 1、lineBar满足长度才执行计算
        if len(self.line_bar) < self.para_cci_len + 2:
            self.write_log(u'数据未充分,当前Bar数据数量：{0}，计算CCI需要：{1}'.
                           format(len(self.line_bar), self.para_cci_len + 2))
            return

        # 计算第1根CCI曲线
        bar_cci = ta.CCI(
            high=np.array(self.high_array[-self.para_cci_len:], dtype=float),
            low=np.array(self.low_array[-self.para_cci_len:], dtype=float),
            close=np.array(self.close_array[-self.para_cci_len:], dtype=float),
            timeperiod=self.para_cci_len)[-1]

        self.cur_cci = round(float(bar_cci), self.round_n)

        if len(self.line_cci) > self.max_hold_bars:
            del self.line_cci[0]
        self.line_cci.append(self.cur_cci)

    def rt_count_cci(self):
        """实时计算CCI值"""
        if self.para_cci_len <= 0:
            return

        # 1、lineBar满足长度才执行计算
        if len(self.line_bar) < self.para_cci_len + 2:
            self.write_log(u'数据未充分,当前Bar数据数量：{0}，计算CCI需要：{1}'.
                           format(len(self.line_bar), self.para_cci_len + 2))
            return

        self._rt_cci = ta.CCI(high=np.append(self.high_array[-2 * self.para_cci_len:], [self.cur_bar.high_price]),
                              low=np.append(self.low_array[-2 * self.para_cci_len:], [self.cur_bar.low_price]),
                              close=np.append(self.close_array[-2 * self.para_cci_len:],
                                              [self.cur_bar.close_price]),
                              timeperiod=self.para_cci_len)[-1]

    @property
    def rt_cci(self):
        self.check_rt_funcs(self.rt_count_cci)
        if self._rt_cci is None:
            return self.cur_cci
        return self._rt_cci

    def __count_kf(self):
        """计算卡尔曼过滤器均线"""
        if not self.para_active_kf or len(self.close_array) == 0:
            return

        if len(self.line_state_mean) == 0 or len(self.line_state_covar) == 0:
            # 首次计算
            try:
                self.kf = KalmanFilter(transition_matrices=[1],
                                       observation_matrices=[1],
                                       initial_state_mean=self.close_array[-1],
                                       initial_state_covariance=1,

                                       transition_covariance=0.01)
            except Exception:
                self.write_log(u'导入卡尔曼过滤器失败,需先安装 pip install pykalman')
                self.para_active_kf = False

            state_means, state_covariances = self.kf.filter(np.array(self.close_array, dtype=float))
            m = state_means[-1].item()
            c = state_covariances[-1].item()
        else:
            # 增量计算
            m = self.line_state_mean[-1]
            c = self.line_state_covar[-1]
            state_means, state_covariances = self.kf.filter_update(filtered_state_mean=m,
                                                                   filtered_state_covariance=c,
                                                                   observation=np.array(self.close_array[-1:],
                                                                                        dtype=float))
            m = state_means[-1].item()
            c = state_covariances[-1].item()

        if len(self.line_state_mean) > self.max_hold_bars:
            del self.line_state_mean[0]
        if len(self.line_state_covar) > self.max_hold_bars:
            del self.line_state_covar[0]

        self.line_state_mean.append(m)
        self.line_state_covar.append(c)

    def __count_skd(self):
        """
        改良得多空线(类似KDJ，RSI）
        :param bar:
        :return:
        """
        if not self.para_active_skd:
            return

        data_len = max(self.para_skd_fast_len * 2, self.para_skd_fast_len + 20)
        if len(self.line_bar) < data_len:
            return

        # 计算最后一根Bar的RSI指标
        last_rsi = ta.RSI(self.close_array[-data_len:], self.para_skd_fast_len)[-1]
        # 添加到lineSkdRSI队列
        if len(self.line_skd_rsi) > self.max_hold_bars:
            del self.line_skd_rsi[0]
        self.line_skd_rsi.append(last_rsi)

        if len(self.line_skd_rsi) < self.para_skd_slow_len:
            return

        # 计算最后根的最高价/最低价
        rsi_HHV = max(self.line_skd_rsi[-self.para_skd_slow_len:])
        rsi_LLV = min(self.line_skd_rsi[-self.para_skd_slow_len:])

        # 计算STO
        if rsi_HHV == rsi_LLV:
            sto = 0
        else:
            sto = 100 * (last_rsi - rsi_LLV) / (rsi_HHV - rsi_LLV)
        sto_len = len(self.line_skd_sto)
        if sto_len > self.max_hold_bars:
            del self.line_skd_sto[0]
        self.line_skd_sto.append(sto)

        # 根据STO，计算SK = EMA(STO,5)
        if sto_len < 5:
            return
        sk = ta.EMA(np.array(self.line_skd_sto, dtype=float), 5)[-1]
        if len(self.line_sk) > self.max_hold_bars:
            del self.line_sk[0]
        self.line_sk.append(sk)

        if len(self.line_sk) < 3:
            return

        sd = ta.EMA(np.array(self.line_sk, dtype=float), 3)[-1]
        if len(self.line_sd) > self.max_hold_bars:
            del self.line_sd[0]
        self.line_sd.append(sd)

        if len(self.line_sd) < 2:
            return

        for t in self.skd_top_list[-1:]:
            t['bars'] += 1

        for b in self.skd_buttom_list[-1:]:
            b['bars'] += 1

        #  记录所有SK的顶部和底部
        # 峰(顶部)
        if self.line_sk[-1] < self.line_sk[-2] and self.line_sk[-3] < self.line_sk[-2]:
            t = {}
            t['type'] = u'T'
            t['sk'] = self.line_sk[-2]
            t['price'] = max(self.high_array[-4:])
            t['time'] = self.line_bar[-1].datetime
            t['bars'] = 0
            if len(self.skd_top_list) > self.max_hold_bars:
                del self.skd_top_list[0]
            self.skd_top_list.append(t)
            if self.cur_skd_count > 0:
                # 检查是否有顶背离
                if self.is_skd_divergence(direction=Direction.LONG):
                    self.cur_skd_divergence = -1

        # 谷(底部)
        elif self.line_sk[-1] > self.line_sk[-2] and self.line_sk[-3] > self.line_sk[-2]:
            b = {}
            b['type'] = u'B'
            b['sk'] = self.line_sk[-2]
            b['price'] = min(self.low_array[-4:])
            b['time'] = self.line_bar[-1].datetime
            b['bars'] = 0
            if len(self.skd_buttom_list) > self.max_hold_bars:
                del self.skd_buttom_list[0]
            self.skd_buttom_list.append(b)
            if self.cur_skd_count < 0:
                # 检查是否有底背离
                if self.is_skd_divergence(direction=Direction.SHORT):
                    self.cur_skd_divergence = 1

        # 判断是否金叉和死叉
        golden_cross = False
        dead_cross = False

        if self.line_sk[-1] > self.line_sk[-2] \
                and self.line_sk[-2] < self.line_sd[-2] \
                and self.line_sk[-1] > self.line_sd[-1]:
            golden_cross = True

        if self.line_sk[-1] < self.line_sk[-2] \
                and self.line_sk[-2] > self.line_sd[-2] \
                and self.line_sk[-1] < self.line_sd[-1]:
            dead_cross = True

        if self.cur_skd_count <= 0:
            if golden_cross:
                # 金叉
                self.cur_skd_count = 1
                self.cur_skd_cross = (self.line_sk[-1] + self.line_sk[-2] + self.line_sd[-1] + self.line_sd[-2]) / 4
                self.rt_skd_count = self.cur_skd_count
                self.rt_skd_cross = self.cur_skd_cross
                if self.rt_skd_cross_price == 0 or self.cur_price < self.rt_skd_cross_price:
                    self.rt_skd_cross_price = self.cur_price
                self.cur_skd_cross_price = self.cur_price
                if self.cur_skd_divergence < 0:
                    # 若原来是顶背离，消失
                    self.cur_skd_divergence = 0
                self.write_log(
                    u'{} Skd Gold Cross:{} at {},{}'.format(self.name, self.cur_skd_cross, self.cur_skd_cross_price,
                                                            self.cur_datetime))
            else:  # if self.lineSK[-1] < self.lineSK[-2]:
                # 延续死叉
                self.cur_skd_count -= 1
                # 取消实时的数据
                self.rt_skd_count = 0
                self.rt_skd_last_cross = 0
                self.rt_skd_cross_price = 0

                # 延续顶背离
                if self.cur_skd_divergence < 0:
                    self.cur_skd_divergence -= 1
            return

        elif self.cur_skd_count >= 0:
            if dead_cross:
                self.cur_skd_count = -1
                self.cur_skd_cross = (self.line_sk[-1] + self.line_sk[-2] + self.line_sd[-1] + self.line_sd[-2]) / 4
                self.rt_skd_count = self.cur_skd_count
                self.rt_skd_last_cross = self.cur_skd_cross
                if self.rt_skd_cross_price == 0 or self.cur_price > self.rt_skd_cross_price:
                    self.rt_skd_cross_price = self.cur_price
                self.cur_skd_cross_price = self.cur_price

                # 若原来是底背离，消失
                if self.cur_skd_divergence > 0:
                    self.cur_skd_divergence = 0

                self.write_log(
                    u'{} Skd Dead Cross:{} at {},{}'.format(self.name, self.cur_skd_cross, self.cur_skd_cross_price,
                                                            self.cur_datetime))

            else:  # if self.lineSK[-1] > self.lineSK[-2]:
                # 延续金叉
                self.cur_skd_count += 1

                # 取消实时的数据
                self.rt_skd_count = 0
                self.rt_skd_cross = 0
                self.rt_skd_cross_price = 0

                # 延续底背离
                if self.cur_skd_divergence > 0:
                    self.cur_skd_divergence += 1

    def __get_2nd_item(self, line):
        """
        获取第二个合适的选项
        :param line:
        :return:
        """
        bars = 0
        for item in reversed(line):
            bars += item['bars']
            if bars > 5:
                return item

        return line[0]

    def is_skd_divergence(self, direction, runtime=False):
        """
        检查是否有背离
        :param:direction，多：检查是否有顶背离，空，检查是否有底背离
        :return:
        """
        if len(self.skd_top_list) < 2 or len(self.skd_buttom_list) < 2 or self._rt_sk is None or self._rt_sd is None:
            return False

        t1 = self.skd_top_list[-1]
        t2 = self.__get_2nd_item(self.skd_top_list[:-1])
        b1 = self.skd_buttom_list[-1]
        b2 = self.__get_2nd_item(self.skd_buttom_list[:-1])

        if runtime:
            # 峰(顶部)
            if self._rt_sk < self.line_sk[-1] and self.line_sk[-2] < self.line_sk[-1]:
                t1 = {}
                t1['type'] = u'T'
                t1['sk'] = self.line_sk[-1]
                t1['price'] = max(self.high_array[-4:])
                t1['time'] = self.line_bar[-1].datetime
                t1['bars'] = 0
                t2 = self.__get_2nd_item(self.skd_top_list)
            # 谷(底部)
            elif self._rt_sk > self.line_sk[-1] and self.line_sk[-2] > self.line_sk[-1]:
                b1 = {}
                b1['type'] = u'B'
                b1['sk'] = self.line_sk[-1]
                b1['price'] = min(self.low_array[-4:])
                b1['time'] = self.line_bar[-1].datetime
                b1['bars'] = 0
                b2 = self.__get_2nd_item(self.skd_buttom_list)

        # 检查顶背离
        if direction == Direction.LONG:
            t1_price = t1.get('price', 0)
            t2_price = t2.get('price', 0)
            t1_sk = t1.get('sk', 0)
            t2_sk = t2.get('sk', 0)
            b1_sk = b1.get('sk', 0)

            t2_t1_price_rate = ((t1_price - t2_price) / t2_price) if t2_price != 0 else 0
            t2_t1_sk_rate = ((t1_sk - t2_sk) / t2_sk) if t2_sk != 0 else 0
            # 背离：价格创新高，SK指标没有创新高
            if t2_t1_price_rate > 0 and t2_t1_sk_rate < 0 and b1_sk > self.para_skd_high:
                return True

        elif direction == Direction.SHORT:
            b1_price = b1.get('price', 0)
            b2_price = b2.get('price', 0)
            b1_sk = b1.get('sk', 0)
            b2_sk = b2.get('sk', 0)
            t1_sk = t1.get('sk', 0)
            b2_b1_price_rate = ((b1_price - b2_price) / b2_price) if b2_price != 0 else 0
            b2_b1_sk_rate = ((b1_sk - b2_sk) / b2_sk) if b2_sk != 0 else 0
            # 背离：价格创新低，指标没有创新低
            if b2_b1_price_rate < 0 and b2_b1_sk_rate > 0 and t1_sk < self.para_skd_low:
                return True

        return False

    def rt_count_sk_sd(self):
        """
        计算实时SKD
        :return:
        """
        if not self.para_active_skd:
            return

        # 准备得数据长度
        data_len = max(self.para_skd_fast_len * 2, self.para_skd_fast_len + 20)
        if len(self.line_bar) < data_len:
            return

        # 计算最后得动态RSI值
        last_rsi = ta.RSI(np.append(self.close_array[-data_len:], [self.cur_price]), self.para_skd_fast_len)[-1]

        # 所有RSI值长度不足计算标准
        if len(self.line_skd_rsi) < self.para_skd_slow_len:
            return

        # 拼接RSI list
        rsi_list = self.line_skd_rsi[1 - self.para_skd_slow_len:]
        rsi_list.append(last_rsi)

        # 获取 RSI得最高/最低值
        rsi_HHV = max(rsi_list)
        rsi_LLV = min(rsi_list)

        # 计算动态STO
        if rsi_HHV == rsi_LLV:
            sto = 0
        else:
            sto = 100 * (last_rsi - rsi_LLV) / (rsi_HHV - rsi_LLV)

        sto_len = len(self.line_skd_sto)
        if sto_len < 5:
            self._rt_sk = self.line_sk[-1] if len(self.line_sk) > 0 else 0
            self._rt_sd = self.line_sd[-1] if len(self.line_sd) > 0 else 0
            return

        # 历史STO
        sto_list = self.line_skd_sto[:]
        sto_list.append(sto)

        self._rt_sk = ta.EMA(np.array(sto_list, dtype=float), 5)[-1]

        sk_list = self.line_sk[:]
        sk_list.append(self._rt_sk)
        if len(sk_list) < 5:
            self._rt_sd = self.line_sd[-1] if len(self.line_sd) > 0 else 0
        else:
            self._rt_sd = ta.EMA(np.array(sk_list, dtype=float), 3)[-1]

    def is_skd_in_risk(self, direction, dist=15, runtime=False):
        """
        检查SDK的方向风险
        :return:
        """
        if not self.para_active_skd or len(self.line_sk) < 2 or self._rt_sk is None:
            return False

        if runtime:
            sk = self._rt_sk
        else:
            sk = self.line_sk[-1]
        if direction == Direction.LONG and sk >= 100 - dist:
            return True

        if direction == Direction.SHORT and sk <= dist:
            return True

        return False

    def is_skd_high_dead_cross(self, runtime=False, high_skd=None):
        """
        检查是否高位死叉
        :return:
        """
        if not self.para_active_skd or len(self.line_sk) < self.para_skd_slow_len:
            return False

        if high_skd is None:
            high_skd = self.para_skd_high

        if runtime:
            # 兼容写法，如果老策略没有配置实时运行，又用到实时数据，就添加
            if self.rt_count_skd not in self.rt_funcs:
                self.write_log(u'skd_is_high_dead_cross(),添加rt_countSkd到实时函数中')
                self.rt_funcs.add(self.rt_count_skd)
                self.rt_count_sk_sd()
            if self._rt_sk is None or self._rt_sd is None:
                return False

            # 判断是否实时死叉
            dead_cross = self._rt_sk < self.line_sk[-1] and self.line_sk[-1] > self.line_sd[
                -1] and self._rt_sk < self._rt_sd

            # 实时死叉
            if self.cur_skd_count >= 0 and dead_cross:
                skd_last_cross = (self._rt_sk + self.line_sk[-1] + self._rt_sd + self.line_sd[-1]) / 4
                # 记录bar内首次死叉后的值:交叉值，价格
                if self.rt_skd_count >= 0:
                    self.rt_skd_count = -1
                    self.rt_skd_cross = skd_last_cross
                    self.rt_skd_cross_price = self.cur_price
                    self.write_log(u'{} skd rt Dead Cross at:{} ,price:{}'
                                   .format(self.name, self.rt_skd_cross, self.rt_skd_cross_price))

                if skd_last_cross > high_skd:
                    return True

        # 非实时，高位死叉
        if self.cur_skd_count < 0 and self.cur_skd_cross > high_skd:
            return True

        return False

    def is_skd_low_golden_cross(self, runtime=False, low_skd=None):
        """
        检查是否低位金叉
        :return:
        """
        if not self.para_active_skd or len(self.line_sk) < self.para_skd_slow_len:
            return False
        if low_skd is None:
            low_skd = self.para_skd_low

        if runtime:
            # 兼容写法，如果老策略没有配置实时运行，又用到实时数据，就添加
            if self.rt_count_skd not in self.rt_funcs:
                self.write_log(u'skd_is_low_golden_cross添加rt_countSkd到实时函数中')
                self.rt_funcs.add(self.rt_count_skd)
                self.rt_count_sk_sd()

            if self._rt_sk is None or self._rt_sd is None:
                return False
            # 判断是否金叉和死叉
            golden_cross = self._rt_sk > self.line_sk[-1] and self.line_sk[-1] < self.line_sd[
                -1] and self._rt_sk > self._rt_sd

            if self.cur_skd_count <= 0 and golden_cross:
                # 实时金叉
                skd_last_cross = (self._rt_sk + self.line_sk[-1] + self._rt_sd + self.line_sd[-1]) / 4

                if self.rt_skd_count <= 0:
                    self.rt_skd_count = 1
                    self.rt_skd_cross = skd_last_cross
                    self.rt_skd_cross_price = self.cur_price
                    self.write_log(u'{} skd rt Gold Cross at:{} ,price:{}'
                                   .format(self.name, self.rt_skd_cross, self.rt_skd_cross_price))
                if skd_last_cross < low_skd:
                    return True

        # 非实时低位金叉
        if self.cur_skd_count > 0 and self.cur_skd_cross < low_skd:
            return True

        return False

    def rt_count_skd(self):
        """
        实时计算 SK,SD值，并且判断计算是否实时金叉/死叉
        :return:
        """
        if self.para_active_skd:
            # 计算实时指标 rt_SK, rt_SD
            self.rt_count_sk_sd()

            # 计算 实时金叉/死叉
            self.is_skd_high_dead_cross(runtime=True, high_skd=0)
            self.is_skd_low_golden_cross(runtime=True, low_skd=100)

    @property
    def rt_sk(self):
        self.check_rt_funcs(self.rt_count_skd)
        if self._rt_sk is None and len(self.line_sk) > 0:
            return self.line_sk[-1]
        return self._rt_sk

    @property
    def rt_sd(self):
        self.check_rt_funcs(self.rt_count_skd)
        if self._rt_sd is None and len(self.line_sd) > 0:
            return self.line_sd[-1]
        return self._rt_sd

    def __count_yb(self):
        """某种趋势线"""

        if not self.para_active_yb:
            return

        if self.para_yb_len < 1:
            return

        if self.para_yb_ref < 1:
            self.write_log(u'参数 self.para_yb_ref:{}不能低于1'.format(self.para_yb_ref))
            return

        ema_len = min(len(self.line_bar), self.para_yb_len)
        if ema_len < 3:
            self.write_log(u'数据未充分,当前Bar数据数量：{0}'.
                           format(len(self.line_bar)))
            return
        # 3、获取前InputN周期(不包含当前周期）的K线
        bar_mid3_ema10 = ta.EMA(self.mid3_array[-ema_len * 3:], ema_len)[-1]
        bar_mid3_ema10 = round(float(bar_mid3_ema10), self.round_n)

        if len(self.line_yb) > self.max_hold_bars:
            del self.line_yb[0]

        self.line_yb.append(bar_mid3_ema10)

        if len(self.line_yb) < self.para_yb_ref + 1:
            return

        if self.line_yb[-1] > self.line_yb[-1 - self.para_yb_ref]:
            self.cur_yb_count = self.cur_yb_count + 1 if self.cur_yb_count >= 0 else 1
        else:
            self.cur_yb_count = self.cur_yb_count - 1 if self.cur_yb_count <= 0 else -1

        if self.cur_yb_count == 1:
            self.write_log(u'YB => Yellow at {}, {}'.format(self.close_array[-1], self.cur_datetime))
        elif self.cur_yb_count == -1:
            self.write_log(u'YB => Blue at {}, {}'.format(self.close_array[-1], self.cur_datetime))

    def rt_count_yb(self):
        """
        实时计算黄蓝
        :return:
        """
        if not self.para_active_yb:
            return
        if self.para_yb_len < 1:
            return
        if self.para_yb_ref < 1:
            self.write_log(u'参数 self.para_yb_ref:{}不能低于1'.format(self.para_yb_ref))
            return

        ema_len = min(len(self.line_bar), self.para_yb_len)
        if ema_len < 3:
            self.write_log(u'数据未充分,当前Bar数据数量：{0}'.
                           format(len(self.line_bar)))
            return
        # 3、获取前InputN周期(包含当前周期）的K线

        last_bar_mid3 = (self.cur_bar.close_price + self.cur_bar.high_price + self.cur_bar.low_price) / 3

        bar_mid3_ema10 = ta.EMA(np.append(self.mid3_array[-ema_len * 3:], [last_bar_mid3]), ema_len)[-1]
        self._rt_yb = round(float(bar_mid3_ema10), self.round_n)

    @property
    def rt_yb(self):
        self.check_rt_funcs(self.rt_count_yb)
        return self._rt_yb

    def __count_golden_section(self):
        """
        重新计算黄金分割线
        :return:
        """
        if self.para_golden_n < 2 or len(self.high_array) == 0:
            return

        hhv = max(self.high_array[-self.para_golden_n - 1:])
        llv = min(self.low_array[-self.para_golden_n - 1:])

        # 纳入实时数据
        hhv = max(hhv, self.cur_bar.high_price)
        llv = min(llv, self.cur_bar.low_price)

        if hhv == llv:
            return

        self.cur_p192 = hhv - (hhv - llv) * 0.192
        self.cur_p382 = hhv - (hhv - llv) * 0.382
        self.cur_p500 = (hhv + llv) / 2
        self.cur_p618 = hhv - (hhv - llv) * 0.618
        self.cur_p809 = hhv - (hhv - llv) * 0.809

        # 根据最小跳动取整
        self.cur_p192 = self.cur_p192 - self.cur_p192 % self.price_tick
        self.cur_p382 = self.cur_p382 - self.cur_p382 % self.price_tick
        self.cur_p500 = self.cur_p500 - self.cur_p500 % self.price_tick
        self.cur_p618 = self.cur_p618 - self.cur_p618 % self.price_tick
        self.cur_p809 = self.cur_p809 - self.cur_p809 % self.price_tick

    def __count_period(self, bar):
        """
        重新计算周期
        利用三均线，长均线作为长趋势判断，三线同向，为多/空。三线缠绕，为震荡
        :param bar:
        :return:
        """
        if not self.para_active_period:
            return

        # 更新周期的Close价格
        self.update_period_price(bar.close_price)

        if len(self.line_ma2) <= 6 or len(self.line_rsi1) <= 2:
            return

        last_mid = self.line_ma2[-1]
        pre_5_mid = self.line_ma2[-5]

        if pre_5_mid == 0 or np.isnan(pre_5_mid):
            self.write_log(u'pre_5中轨取值异常')
            return

        # 当前均值,与前5值价差,除以5个砖块高度
        self.cur_atan = math.atan((last_mid - pre_5_mid) / (self.height * 5)) * 180 / math.pi

        self.cur_atan = round(self.cur_atan, self.round_n)

        if self.cur_period is None:
            self.write_log(u'初始化周期为震荡')
            self.cur_period = CtaPeriod(mode=Period.SHOCK, price=bar.close_price, pre_mode=Period.INIT, dt=bar.datetime)
            self.period_list.append(self.cur_period)

        if len(self.line_atan) > self.max_hold_bars:
            del self.line_atan[0]
        self.line_atan.append(self.cur_atan)

        # 当前期趋势是震荡
        if self.cur_period.mode == Period.SHOCK:
            # 震荡 -》 空
            if self.ma12_count < 0 and self.ma13_count < 0 and self.ma23_count < 0 and self.line_ma3[-1] < \
                    self.line_ma3[
                        -2]:
                self.__append_period(mode=Period.SHORT, price=bar.close_price)
            # 震荡 =》 多
            elif self.ma12_count > 0 and self.ma13_count > 0 and self.ma23_count > 0 and self.line_ma3[-1] > \
                    self.line_ma3[-2]:
                self.__append_period(mode=Period.LONG, price=bar.close_price)
            # 周期维持不变
            else:
                self.write_log(u'{} 角度维持，Atan:{},周期维持:{}'.
                               format(bar.datetime, self.cur_atan, self.cur_period.mode))
            return

        # 当前期趋势是空
        if self.cur_period.mode == Period.SHORT:
            # 空=》震荡
            if not (self.ma12_count < 0 and self.ma13_count < 0
                    and self.ma23_count < 0 and self.line_ma3[-1] < self.line_ma3[-2]):
                self.__append_period(mode=Period.SHOCK, price=bar.close_price)
            # 周期维持空
            else:
                self.write_log(u'{} 角度向下{},周期维持:{}'.
                               format(bar.datetime, self.cur_atan, self.cur_period.mode))
            return

        # 当前期趋势是多
        if self.cur_period.mode == Period.LONG:
            # 多=》震荡
            if not (self.ma12_count > 0 and self.ma13_count > 0
                    and self.ma23_count > 0 and self.line_ma3[-1] > self.line_ma3[-2]):
                self.__append_period(mode=Period.SHOCK, price=bar.close_price)
            # 周期保持多
            else:
                self.write_log(u'{} 角度向上,Atan:{},周期维持:{}'.
                               format(bar.datetime, self.cur_atan, self.cur_period.mode))
            return

    def __append_period(self, mode, price):
        """添加周期"""

        # 删除多出的周期
        if len(self.period_list) > 8:
            del self.period_list[0]
        old_period = {}
        if self.cur_period:
            self.write_log(u'{} O:{},H:{},L:{},C:{} => {}'
                           .format(self.cur_period.mode,
                                   self.cur_period.open,
                                   self.cur_period.high,
                                   self.cur_period.low,
                                   self.cur_period.close,
                                   mode))
            if self.CB_ON_PERIOD in self.cb_dict:
                old_period.update(self.cur_period.__dict__)
        # 新增周期
        self.cur_period = CtaPeriod(mode, price, pre_mode=old_period.get('mode', Period.INIT), dt=self.cur_datetime)

        self.period_list.append(self.cur_period)

        # 推动事件回调
        func = self.cb_dict.get(self.CB_ON_PERIOD, None)
        if func:
            try:
                new_period = {}
                new_period.update(self.cur_period.__dict__)
                func(old_period=old_period,
                     new_period=new_period)

            except Exception as ex:
                self.write_log(u'call back event{} exception:{}'.format(self.CB_ON_PERIOD, str(ex)))
                self.write_log(u'traceback:{}'.format(traceback.format_exc()))

    # ----------------------------------------------------------------------
    def write_log(self, content):
        """记录CTA日志"""
        if self.strategy:
            self.strategy.write_log(u'[' + self.name + u']' + content)
        else:
            print(u'[' + self.name + u']' + content)

    def export_to_csv(self, bar):
        """导出当前Bar到csv文件"""
        if self.export_filename is None or len(self.export_fields) == 0:
            return
        field_names = []
        save_dict = {}
        for field in self.export_fields:
            field_name = field.get('name', None)
            attr_name = field.get('attr', None)
            source = field.get('source', None)
            type_ = field.get('type_', None)
            if field_name is None or attr_name is None or source is None or type_ is None:
                continue
            field_names.append(field_name)
            if source == 'bar':
                save_dict[field_name] = getattr(bar, str(attr_name), None)
            else:
                if type_ == 'list':
                    list_obj = getattr(self, str(attr_name), None)
                    if list_obj is None or len(list_obj) == 0:
                        save_dict[field_name] = 0
                    else:
                        save_dict[field_name] = list_obj[-1]
                elif type_ == 'string':
                    save_dict[field_name] = getattr(self, str(attr_name), '')
                else:
                    save_dict[field_name] = getattr(self, str(attr_name), 0)

        if len(save_dict) > 0:
            self.append_data(file_name=self.export_filename, dict_data=save_dict, field_names=field_names)

    def load_from_csv(self):
        """
        从csv文件加载之前数据
        :return:
        """
        if self.export_filename is None or len(self.export_fields) == 0:
            return
        attr_list = []
        for field in self.export_fields:
            field_name = field.get('name', None)
            attr_name = field.get('attr', None)
            source = field.get('source', None)
            type_ = field.get('type_', None)
            if field_name is None or attr_name is None or source is None or type_ is None:
                continue
            if source == 'bar':
                attr_list.append(str(attr_name))

        if 'datetime' not in attr_list:
            return

        try:
            with open(self.export_filename, 'r', encoding='utf8') as f:
                reader = csv.DictReader(f, delimiter=",")
                self.write_log(u'加载{0}'.format(self.export_filename))
                for row in reader:
                    bar = RenkoBarData()
                    for attr in attr_list:
                        if attr in row:
                            setattr(bar, attr, row.get(attr, None))

                    if isinstance(bar.datetime):
                        self.add_bar(bar, is_init=True)

        except Exception as ex:
            self.write_log(u'加载k线csv文件异常:{}'.format(str(ex)))

    def append_data(self, file_name, dict_data, field_names=None):
        """
        添加数据到csv文件中
        :param file_name:  csv的文件全路径
        :param dict_data:  OrderedDict
        :return:
        """
        if not isinstance(dict_data, dict):
            print(u'{}.append_data，输入数据不是dict'.format(self.name), file=sys.stderr)
            return

        dict_fieldnames = list(dict_data.keys()) if field_names is None else field_names

        if not isinstance(dict_fieldnames, list):
            print(u'{}append_data，输入字段不是list'.format(self.name), file=sys.stderr)
            return
        try:
            if not os.path.exists(file_name):
                self.write_log(u'create csv file:{}'.format(file_name))
                with open(file_name, 'a', encoding='utf8', newline='') as csvWriteFile:
                    writer = csv.DictWriter(f=csvWriteFile, fieldnames=dict_fieldnames, dialect='excel')
                    self.write_log(u'write csv header:{}'.format(dict_fieldnames))
                    writer.writeheader()
                    writer.writerow(dict_data)
            else:
                dt = dict_data.get('datetime', None)
                if dt is not None:
                    dt_index = dict_fieldnames.index('datetime')
                    last_dt = self.get_csv_last_dt(file_name=file_name, dt_index=dt_index,
                                                   line_length=sys.getsizeof(dict_data) / 8 + 1)
                    if last_dt is not None and dt < last_dt:
                        print(u'{}新增数据时间{}比csv最后一条记录时间{}早，不插入'.format(self.name, dt, last_dt))
                        return

                with open(file_name, 'a', encoding='utf8', newline='') as csvWriteFile:
                    writer = csv.DictWriter(f=csvWriteFile, fieldnames=dict_fieldnames, dialect='excel',
                                            extrasaction='ignore')
                    writer.writerow(dict_data)
        except Exception as ex:
            print(u'{}.append_data exception:{}/{}'.format(self.name, str(ex), traceback.format_exc()))

    def get_csv_last_dt(self, file_name, dt_index=0, line_length=1000):
        """
        获取csv文件最后一行的日期数据(第dt_index个字段必须是 '%Y-%m-%d %H:%M:%S'格式
        :param file_name:文件名
        :param line_length: 行数据的长度
        :return: None，文件不存在，或者时间格式不正确
        """
        with open(file_name, 'r') as f:
            f_size = os.path.getsize(file_name)
            if f_size < line_length:
                line_length = f_size
            f.seek(f_size - line_length)  # 移动到最后1000个字节
            for row in f.readlines()[-1:]:

                datas = row.split(',')
                if len(datas) > dt_index + 1:
                    try:
                        last_dt = datetime.strptime(datas[dt_index], '%Y-%m-%d %H:%M:%S')
                        return last_dt
                    except Exception:
                        return None
            return None

    def get_data(self):
        """
        获取数据，供外部系统查看
        :return: dict:
        {
        name: []， # k线名称
        type: k线类型：renko
        interval: 周期 高度 或者千分几
        symbol: 品种,
        main_indicators: [] , 主图指标
        sub_indicators: []， 附图指标
        start_time: '', 开始时间
        end_time: ''，结束时间
        data_list: list of dict
        }

        """
        # 根据参数，生成主图指标和附图指标
        indicators = {}

        # 前高/前低（通道）
        if isinstance(self.para_pre_len, int) and self.para_pre_len > 0:
            indicator = {
                'name': 'preHigh{}'.format(self.para_pre_len),
                'attr_name': 'line_pre_high',
                'is_main': True,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})
            indicator = {
                'name': 'preLow{}'.format(self.para_pre_len),
                'attr_name': 'line_pre_low',
                'is_main': True,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})

        # EMA 均线（主图）
        if isinstance(self.para_ema1_len, int) and self.para_ema1_len > 0:
            indicator = {
                'name': 'EMA{}'.format(self.para_ema1_len),
                'attr_name': 'line_ema1',
                'is_main': True,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})
        if isinstance(self.para_ema2_len, int) and self.para_ema2_len > 0:
            indicator = {
                'name': 'EMA{}'.format(self.para_ema2_len),
                'attr_name': 'line_ema2',
                'is_main': True,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})
        if isinstance(self.para_ema3_len, int) and self.para_ema3_len > 0:
            indicator = {
                'name': 'EMA{}'.format(self.para_ema3_len),
                'attr_name': 'line_ema3',
                'is_main': True,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})

        # MA 均线 （主图）
        if isinstance(self.para_ma1_len, int) and self.para_ma1_len > 0:
            indicator = {
                'name': 'MA{}'.format(self.para_ma1_len),
                'attr_name': 'line_ma1',
                'is_main': True,
                'type': 'line'
            }
            indicators.update({indicator.get('name'): copy.copy(indicator)})
        if isinstance(self.para_ma2_len, int) and self.para_ma2_len > 0:
            indicator = {
                'name': 'MA{}'.format(self.para_ma2_len),
                'attr_name': 'line_ma2',
                'is_main': True,
                'type': 'line'
            }
            indicators.update({indicator.get('name'): copy.copy(indicator)})
        if isinstance(self.para_ma3_len, int) and self.para_ma3_len > 0:
            indicator = {
                'name': 'MA{}'.format(self.para_ma3_len),
                'attr_name': 'line_ma3',
                'is_main': True,
                'type': 'line'
            }
            indicators.update({indicator.get('name'): copy.copy(indicator)})

        if isinstance(self.para_ama_len, int) and self.para_ama_len > 0:
            indicator = {
                'name': 'AMA{}'.format(self.para_ama_len),
                'attr_name': 'line_ama',
                'is_main': True,
                'type': 'line'
            }
            indicators.update({indicator.get('name'): copy.copy(indicator)})

        # 动能指标（附图）
        if isinstance(self.para_dmi_len, int) and self.para_dmi_len > 0:
            indicator = {
                'name': 'ADX({})'.format(self.para_dmi_len),
                'attr_name': 'line_adx',
                'is_main': False,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})
            indicator = {
                'name': 'ADXR({})'.format(self.para_dmi_len),
                'attr_name': 'line_adxr',
                'is_main': False,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})

        # 平均波动率 (副图）
        if isinstance(self.para_atr1_len, int) and self.para_atr1_len > 0:
            indicator = {
                'name': 'ATR{}'.format(self.para_atr1_len),
                'attr_name': 'line_atr1',
                'is_main': False,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})
        if isinstance(self.para_atr2_len, int) and self.para_atr2_len > 0:
            indicator = {
                'name': 'ATR{}'.format(self.para_atr2_len),
                'attr_name': 'line_atr2',
                'is_main': False,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})
        if isinstance(self.para_atr3_len, int) and self.para_atr3_len > 0:
            indicator = {
                'name': 'ATR{}'.format(self.para_atr3_len),
                'attr_name': 'line_atr3',
                'is_main': False,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})

        # 摆动指标（附图）
        if isinstance(self.para_rsi1_len, int) and self.para_rsi1_len > 0:
            indicator = {
                'name': 'RSI({})'.format(self.para_rsi1_len),
                'attr_name': 'line_rsi1',
                'is_main': False,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})
        if isinstance(self.para_rsi2_len, int) and self.para_rsi2_len > 0:
            indicator = {
                'name': 'RSI({})'.format(self.para_rsi2_len),
                'attr_name': 'line_rsi2',
                'is_main': False,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})

        # 市场波动指数 （副图）
        if isinstance(self.para_cmi_len, int) and self.para_cmi_len > 0:
            indicator = {
                'name': 'CMI({})'.format(self.para_cmi_len),
                'attr_name': 'line_cmi',
                'is_main': False,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})

        # 布林通道 (主图)
        if (isinstance(self.para_boll_len, int) and self.para_boll_len > 0):
            indicator = {
                'name': 'BOLL_U'.format(self.para_boll_len),
                'attr_name': 'line_boll_upper',
                'is_main': True,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})
            indicator = {
                'name': 'BOLL({})_M'.format(self.para_boll_len),
                'attr_name': 'line_boll_middle',
                'is_main': True,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})
            indicator = {
                'name': 'BOLL({})_L'.format(self.para_boll_len),
                'attr_name': 'line_boll_lower',
                'is_main': True,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})

        # 布林通道 (主图)
        if (isinstance(self.para_boll2_len, int) and self.para_boll2_len > 0):
            indicator = {
                'name': 'BOLL_U'.format(self.para_boll2_len),
                'attr_name': 'line_boll2_upper',
                'is_main': True,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})
            indicator = {
                'name': 'BOLL({})_M'.format(self.para_boll2_len),
                'attr_name': 'line_boll2_middle',
                'is_main': True,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})
            indicator = {
                'name': 'BOLL({})_L'.format(self.para_boll2_len),
                'attr_name': 'line_boll2_lower',
                'is_main': True,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})

        # KDJ 摆动指标 (副图)
        if (isinstance(self.para_kdj_len, int) and self.para_kdj_len > 0) or (
                isinstance(self.para_kdj_tb_len, int) and self.para_kdj_tb_len > 0):
            kdj_len = max(self.para_kdj_tb_len, self.para_kdj_len)
            indicator = {
                'name': 'KDJ({})_K'.format(kdj_len),
                'attr_name': 'line_k',
                'is_main': False,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})
            indicator = {
                'name': 'KDJ({})_D'.format(kdj_len),
                'attr_name': 'line_d',
                'is_main': False,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})

        # CCI 动能指标 (副图)
        if isinstance(self.para_cci_len, int) and self.para_cci_len > 0:
            indicator = {
                'name': 'CCI({})'.format(self.para_cci_len),
                'attr_name': 'line_cci',
                'is_main': False,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})

        if isinstance(self.para_macd_fast_len, int) and self.para_macd_fast_len > 0:
            indicator = {
                'name': 'Dif',
                'attr_name': 'line_dif',
                'is_main': False,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})
            indicator = {
                'name': 'Dea',
                'attr_name': 'line_dea',
                'is_main': False,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})
            indicator = {
                'name': 'Macd',
                'attr_name': 'line_macd',
                'is_main': False,
                'type': 'bar'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})

        # 卡尔曼均线
        if self.para_active_kf:
            indicator = {
                'name': 'KF',
                'attr_name': 'line_state_mean',
                'is_main': True,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})

        # 摆动指标
        if self.para_active_skd:
            indicator = {
                'name': 'SK',
                'attr_name': 'line_sk',
                'is_main': False,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})
            indicator = {
                'name': 'SD',
                'attr_name': 'line_sd',
                'is_main': False,
                'type': 'line'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})

        # 重心线
        if self.para_active_yb:
            indicator = {
                'name': 'YB',
                'attr_name': 'line_yb',
                'is_main': True,
                'type': 'bar'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})

        # 抛物线 (主图指标)
        if self.para_sar_step > 0:
            indicator = {
                'name': 'SAR',
                'attr_name': 'line_sar',
                'is_main': True,
                'type': 'point'}
            indicators.update({indicator.get('name'): copy.copy(indicator)})

        # 逐一填充数据到pandas
        bar_list = [OrderedDict({
            'datetime': bar.datetime,
            'open': bar.open_price,
            'high': bar.high_price,
            'low': bar.low_price,
            'close': bar.close_price,
            'volume': bar.volume,
            'openInterest': bar.open_interest}) for bar in self.line_bar]

        bar_len = len(bar_list)
        if bar_len == 0:
            return {}

        # 补充数据
        main_indicators = []
        sub_indicators = []
        for k, v in indicators.items():
            attr_name = v.get('attr_name', None)
            if attr_name is None or not hasattr(self, attr_name):
                continue
            attr_data_list = getattr(self, attr_name, [])
            data_len = len(attr_data_list)
            if data_len == 0:
                continue
            if data_len > bar_len:
                attr_data_list = attr_data_list[-bar_len:]
            elif data_len < bar_len:
                first_data = attr_data_list[0]
                attr_data_list = [first_data] * (bar_len - data_len) + attr_data_list

            # 逐一增加到bar_list的每个dict中
            for i in range(bar_len):
                bar_list[i].update({k: attr_data_list[i]})

            if v.get('is_main', False):
                main_indicators.append({'name': k, 'type': v.get('type')})
            else:
                sub_indicators.append({'name': k, 'type': v.get('type')})

        return {
            'name': self.name,
            'type': 'renko',
            'interval': 'K{}'.format(self.kilo_height) if self.kilo_height > 0 else '{}'.format(
                int(self.height / self.price_tick)),
            'symbol': self.line_bar[-1].symbol,
            'main_indicators': list(sorted(main_indicators, key=lambda x: x['name'])),
            'sub_indicators': list(sorted(sub_indicators, key=lambda x: x['name'])),
            'start_time': bar_list[0].get('datetime'),
            'end_time': bar_list[-1].get('datetime'),
            'data_list': bar_list}
