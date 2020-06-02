# -*- coding: utf-8 -*-

"""数据源客户端"""
from vnpy.trader.datasource.dataapi import DataSourceApi
from vnpy.trader.datasource.jqdata import jqdata_client
from vnpy.trader.datasource.rqdata import rqdata_client
from vnpy.trader.datasource.tqdata import tqdata_client
from vnpy.trader.datasource.jjdata import jjdata_client
from vnpy.trader.setting import SETTINGS

if SETTINGS["datasource.api"] == "jqdata":
    datasource_client: DataSourceApi = jqdata_client

elif SETTINGS["datasource.api"] == "rqdata":
    datasource_client: DataSourceApi = rqdata_client

elif SETTINGS["datasource.api"] == "tqdata":
    datasource_client: DataSourceApi = tqdata_client

elif SETTINGS["datasource.api"] == "jjdata":
    datasource_client: DataSourceApi = jjdata_client
