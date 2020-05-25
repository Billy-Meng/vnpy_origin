# -*- coding: utf-8 -*-
# @Time    : 2020/5/25
# @Author  : Billy

"""数据源客户端"""
from vnpy.trader.datasource.dataapi import DataSourceApi
from vnpy.trader.datasource.jqdata import jqdata_client
from vnpy.trader.datasource.rqdata import rqdata_client
from vnpy.trader.datasource.tqdata import tqdata_client
from vnpy.trader.setting import SETTINGS

if SETTINGS["datasource.api"] == "jqdata":
    datasource_client: DataSourceApi = jqdata_client

elif SETTINGS["datasource.api"] == "rqdata":
    datasource_client: DataSourceApi = rqdata_client

elif SETTINGS["datasource.api"] == "tqdata":
    datasource_client: DataSourceApi = tqdata_client
