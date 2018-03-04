# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:53:34 2017

@author: Lenovo
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
os.chdir("F:\\研究生\\京东金融2017\\Loan_Forecasting_Qualification\\reprocessData\\")
data = pd.read_csv("object.csv")
data.columns=['uid',8,9,10,11]
data.fillna(0,inplace = True)
x = data[[8,9,10]]
y = data[[11]]
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 1)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
data_train = xgb.DMatrix(x_train,label = y_train)
data_test = xgb.DMatrix(x_test,label = y_test)
watch_list = [(data_test,'eval'),(data_train,'train')]
param = {'max_depth':2,'eta' :1, 'silent': 1,'objective':'reg:linear'}
num_round = 100
bst = xgb.train(param, data_train,num_round)
y_pred = bst.predict(data_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_pred, y_test)
rmse = np.sqrt(metrics.mean_absolute_error(y_pred, y_test))
print("mse的值：",mse)
print("rmse的值：",rmse)
#预测模型
data_12 = data[[9,10,11]]
data_12 = np.array(data_12)
data_12 = xgb.DMatrix(data_12)
data_12_pred = bst.predict(data_12)
data_12_pred = pd.DataFrame(data_12_pred,index = data['uid'])
#加载users表
os.chdir("F:\\研究生\\京东金融2017\\Loan_Forecasting_Qualification\\")
t_user = pd.read_csv("t_user.csv",parse_dates = True)
t_user['active_date'] = t_user['active_date'].astype(np.datetime64)
min_time = t_user['active_date'].min()
t_user['active_date'] = t_user['active_date'].apply(lambda x:x - min_time)
t_user = t_user['uid']
t_user = pd.DataFrame(t_user)
t_user = pd.merge(t_user,data_12_pred,left_on = 'uid',right_index = True,how = 'outer')
t_user.index  = t_user['uid']
t_user = t_user[[0]]
t_user.sort_index(inplace = True)
t_user.fillna(0.1,inplace = True)
t_user.replace({np.nan:0.1})
t_user.loc[t_user[0] < 2] = 0.1
t_user.to_csv("xgboost_with_table_loan.csv",header = False)