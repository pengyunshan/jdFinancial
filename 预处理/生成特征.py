# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 20:39:54 2017

@author: Lenovo
"""
#点击表生成特征，点击时刻的平均值，点击页面的参数值
#订单表生成特征，一个月购买的总金额，
import pandas as pd
import os
import numpy as np
os.chdir("F:/研究生/京东金融2017/Loan_Forecasting_Qualification")

t_order = pd.read_csv("t_order.csv")
t_order['buy_time'] = t_order['buy_time'].astype(np.datetime64)
t_order['cost'] = t_order['price'] * t_order['qty'] - t_order['discount']
t_order['month'] = t_order['buy_time'].apply(lambda x: x.month)
t_order_copy = t_order[['uid','month','cost']]

t_order_8 = t_order_copy[t_order_copy['month']==8]
t_order_9 = t_order_copy[t_order_copy['month']==9]
t_order_10 = t_order_copy[t_order_copy['month']==10]
t_order_11 = t_order_copy[t_order_copy['month']==11]
t_order_10.to_csv("t_order_10.csv")
t_order_11.to_csv("t_order_11.csv")
t_order_9.to_csv("t_order_9.csv")
t_order_8.to_csv("t_order_8.csv")

t_order_copy.to_csv("t_order_process.csv")