# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 18:17:37 2017

@author: Frank
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
#reset workspace
os.chdir("F:\\研究生\\京东金融2017\\Loan_Forecasting_Qualification")
#input game data
t_user = pd.read_csv("t_user.csv")
t_loan = pd.read_csv("t_loan.csv")

def str_to_date(timestr):
	time = datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S")
	return time

#处理时间信息
t_loan['loan_time'] = t_loan['loan_time'].apply(str_to_date)
t_loan['loan_time'] = t_loan['loan_time'].astype(np.dtype('datetime64[ns]'))

#all the game data from 2016,so we only need the message of month
z = t_loan['loan_time'].apply(lambda x:x.year)
z.value_counts()
month = t_loan['loan_time'].apply(lambda x:x.month)
t_loan['loan_time'] = month
del z,month

#每个用户每个月可能会多次贷款
sum_aggr = t_loan.groupby(['uid','loan_time']).sum()
sum_aggr = sum_aggr['loan_amount']
sum_aggr = sum_aggr.unstack()
loan_index = sum_aggr.index.copy()
sum_aggr.to_csv('sum_aggr.csv',index = True)

#missing value filling
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = np.nan,strategy = 'mean',axis = 1)
imp.fit(sum_aggr)
sum_aggr = imp.transform(sum_aggr)

#linear modeling

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(sum_aggr[:,[0,1,2]],sum_aggr[:,3],random_state = 1)
from sklearn.linear_model import LinearRegression 
linreg = LinearRegression()
linreg.fit(x_train,y_train)
print("截距：",linreg.intercept_)
print("相似性：",linreg.coef_)
y_pred = linreg.predict(x_test)
#Assess the linear model
from sklearn import metrics
print(metrics.mean_squared_error(y_pred,y_test))
print("标准:",np.sqrt(metrics.mean_squared_error(y_pred,y_test)))
y_pred = pd.DataFrame(y_pred)
#use the linear model we got to forecast the next month
y_pred_12 = linreg.predict(sum_aggr[:,[1,2,3]])
y_pred_12 = pd.DataFrame(y_pred_12,index = loan_index)

#do not soomth the data
origin_sum = pd.read_csv('sum_aggr.csv',index_col ='uid')
origin_sum = origin_sum.fillna(0)
x_train,x_test,y_train,y_test = train_test_split(origin_sum.loc[:,['8','9','10']],origin_sum.loc[:,'11'],random_state = 1)
linreg = LinearRegression()
linreg.fit(x_train,y_train)
print("截距：",linreg.intercept_)
print("相似性：",linreg.coef_)
y_pred = linreg.predict(x_test)
#Assess the linear model
from sklearn import metrics
print(metrics.mean_squared_error(y_pred,y_test))
print("标准差:",np.sqrt(metrics.mean_squared_error(y_pred,y_test)))

