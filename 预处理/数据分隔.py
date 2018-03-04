# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 20:17:33 2017

@author: Lenovo
"""
import numpy as np
import pandas as pd
import os
#reset workspace
os.chdir("F:\\研究生\\京东金融2017\\Loan_Forecasting_Qualification")
#input game data
t_order = pd.read_csv("t_order.csv")
t_click = pd.read_csv("t_click.csv")
t_user = pd.read_csv("t_user.csv")
#处理时间信息
t_order['buy_time'] = t_order['buy_time'].astype(np.dtype('datetime64[ns]'))
t_click['click_time'] = t_click['click_time'].astype(np.dtype('datetime64[ns]'))
t_user['active_date'] = t_user['active_date'].astype(np.dtype('datetime64[ns]'))
#将订单数据和点击数据按月拆分
t_order['month'] = t_order['buy_time'].apply(lambda x : x.month)
t_order_8 = t_order[t_order['month']==8]
t_order_9 = t_order[t_order['month']==9]
t_order_10 = t_order[t_order['month']==10]
t_order_11 = t_order[t_order['month']==11]
t_order_10.to_csv("t_order_10.csv")
t_order_11.to_csv("t_order_11.csv")
t_order_9.to_csv("t_order_9.csv")
t_order_8.to_csv("t_order_8.csv")
#apply这个大数据集尽量别用
t_click['month'] = t_click['click_time'].apply(lambda x: x.month)
t_click_8 = t_click[t_click['month'] == 8]
t_click_9 = t_click[t_click['month'] == 9]
t_click_10 = t_click[t_click['month'] == 10]
t_click_11 = t_click[t_click['month'] == 11]
t_click_8.to_csv("t_click_8.csv")
t_click_9.to_csv("t_click_9.csv")
t_click_10.to_csv("t_click_10.csv")
t_click_11.to_csv("t_click_11.csv")


import os
os.chdir("F:\\研究生\\京东金融2017\\Loan_Forecasting_Qualification\\reprocessData")
import numpy as np
import pandas as pd
t_order_process = pd.read_csv("t_order_process.csv")

