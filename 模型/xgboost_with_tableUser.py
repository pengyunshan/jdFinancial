# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 07:44:34 2017

@author: Lenovo
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
os.chdir("F:\\研究生\\京东金融2017\\Loan_Forecasting_Qualification\\reprocessData\\")
data = pd.read_csv("object.csv")
os.chdir("F:\\研究生\\京东金融2017\\Loan_Forecasting_Qualification\\")
t_user = pd.read_csv("t_user.csv")
t_user['active_date'] = t_user['active_date'].astype(np.datetime64)
min_time = t_user['active_date'].min()
t_user['active_date'] = t_user['active_date'].apply(lambda x:x - min_time) 
t_user['day'] = t_user['active_date'].apply(lambda x:x.days)
t_user.set_index('uid',inplace =True)
t_user.sort_index(inplace =True)
t_user = t_user[['age','sex','day','limit']]
data.columns=['uid',8,9,10,11]
data = data.set_index('uid')
data = pd.merge(t_user,data,left_index =True,right_index = True ,how = 'outer')
data.fillna(0,inplace = True)

X = data[['age','sex','limit',8,9,10]]
y = data[[11]]
x_train,x_test,y_train,y_test = train_test_split(X,y,random_state = 1)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
data_train = xgb.DMatrix(x_train,label = y_train)
data_test = xgb.DMatrix(x_test,label = y_test)

watch_list = [(data_test,'eval'),(data_train,'train')]
param = {'max_depth':6,'eta' :1, 'silent': 1,'objective':'reg:linear'}
num_round = 1000
bst = xgb.train(param, data_train,num_round)
y_pred = bst.predict(data_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_pred, y_test)
rmse = np.sqrt(metrics.mean_squared_error(y_pred, y_test))
print("mse的值：",mse)
print("rmse的值：",rmse)
#预测模型
data_pred = xgb.DMatrix(np.array(data[['age','sex','limit',9,10,11]]))
Dec_pred = bst.predict(data_pred)
Dec_pred = pd.DataFrame(Dec_pred,index = data.index)
Dec_pred.loc[Dec_pred[0]<0,0] = 0.000001
Dec_pred.to_csv("xgboost_with_user.csv",header = False)