# -*- coding: utf-8 -*-
# @Time    : 2020/5/21
# @Author  : Billy

"""数据源客户端"""
from vnpy.trader.datasource.dataapi import DataSourceApi
from vnpy.trader.datasource.jqdata import jqdata_client
from vnpy.trader.datasource.rqdata import rqdata_client
from vnpy.trader.setting import SETTINGS

if SETTINGS["datasource.api"] == "jqdata":
    datasource_client: DataSourceApi = jqdata_client
else:
    datasource_client: DataSourceApi = rqdata_client
