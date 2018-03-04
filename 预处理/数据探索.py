# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:19:02 2017

@author: Lenovo
"""
import numpy as np
import pandas as pd
import os
from datetime import datetime
#reset workspace
os.chdir("F:\\研究生\\京东金融2017\\Loan_Forecasting_Qualification")
#input game data
t_user = pd.read_csv("t_user.csv")
t_order = pd.read_csv("t_order.csv")
t_loan_sum = pd.read_csv("t_loan_sum.csv")
t_loan = pd.read_csv("t_loan.csv")
t_click = pd.read_csv("t_click.csv")

def strtime_to_datetime(timestr):
    """将字符串格式的时间 (含毫秒) 转为 datetiem 格式
    :param timestr: {str}'2016-02-25 20:21:04.242'
    :return: {datetime}2016-02-25 20:21:04.242000
    """
    local_datetime = datetime.strptime(timestr, "%Y-%m-%d")
    return local_datetime

def strtime_to_datetime2(timestr):
    """将字符串格式的时间 (含毫秒) 转为 datetiem 格式
    :param timestr: {str}'2016-02-25 20:21:04.242'
    :return: {datetime}2016-02-25 20:21:04.242000
    """
    local_datetime = datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S")
    return local_datetime

def strtime_to_datetime3(timestr):
    """将字符串格式的时间 (含毫秒) 转为 datetiem 格式
    :param timestr: {str}'2016-02-25 20:21:04.242'
    :return: {datetime}2016-02-25 20:21:04.242000
    """
    local_datetime = datetime.strptime(timestr, "%Y-%m")
    return local_datetime

def getMonth(timestr):
    datetime.strftime()
#处理时间信息
t_user.head(5)
t_user['active_date'].dtype
t_user['active_date'] = t_user['active_date'].apply(strtime_to_datetime)
t_user['active_date'] = t_user['active_date'].astype(np.dtype('datetime64[ns]'))


t_click.head(5)
t_click['click_time'].dtype
t_click['click_time'] = t_click['click_time'].apply(strtime_to_datetime2)
t_click['click_time'] = t_click['click_time'].astype(np.dtype('datetime64[ns]'))
#应该在导入数据的时候就定义好数据类型

t_order.head(5)
t_order['buy_time'].dtype
t_order['buy_time'] = t_order['buy_time'].apply(strtime_to_datetime)
t_order['buy_time'] = t_order['buy_time'].astype(np.dtype('datetime64[ns]'))


t_loan.head(5)
t_loan['loan_time'].dtype
t_loan['loan_time'] = t_loan['loan_time'].apply(strtime_to_datetime2)
t_loan['loan_time'] = t_loan['loan_time'].astype(np.dtype('datetime64[ns]'))


t_loan_sum.head(5)
t_loan_sum['month'].dtype
t_loan_sum['month'] = t_loan_sum['month'].apply(strtime_to_datetime3)
t_loan_sum['month'] = t_loan_sum['month'].astype(np.dtype('datetime64[ns]'))


#全是2016年的数据，所以我们只需要月份数据
z = t_loan['loan_time'].apply(lambda x:x.year)
z.value_counts()
month = t_loan['loan_time'].apply(lambda x:x.month)
t_loan['loan_time'] = month
del z,month
#每个用户每个月可能会多次贷款
sum_aggr = t_loan.groupby(['uid','loan_time']).sum()
sum_aggr = sum_aggr['loan_amount']
sum_aggr = sum_aggr.unstack()
sum_aggr.to_csv('sum_aggr.csv',index = True)

#现在提取到了用户的款的月份数据
#t_loan_nov = t_loan[t_loan['loan_time'] == 11]
#获取11月份的人的数据
nov_person = set(t_loan_sum['uid'])
all_person = set(t_loan.index)
unused_peson = all_person - nov_person
len(all_person)
len(nov_person)
len(unused_peson)
#group表里得到11月非空的数据
group[group[11].isnull()]
#对loan_sum表进行分析
nov_person_loan = t_loan[t_loan['uid'].apply(lambda x: x in nov_person)]
nov_person_click = t_click[t_click['uid'].apply(lambda x: x in nov_person)]
nov_person_order = t_order[t_order['uid'].apply(lambda x :x in nov_person)]
nov_person = t_user[t_user['uid'].apply(lambda x :x in nov_person)]
#删除其他无关变量
del t_click
del t_loan
del t_order
del t_user
"""
#根据11月分的贷款人数据汇总得到在11月有贷款信息的人的各种数据
nov_person.to_csv('nov_person.csv',index = False)
nov_person_click.to_csv('nov_person_click.csv',index = False)
nov_person_order.to_csv('nov_person_order.csv',index = False)
nov_person_loan.to_csv('nov_person_loan.csv',index = False)
"""
t_loan = t_loan[['uid','loan_time','loan_amount']]
sum_t_loan = t_loan.groupby(['uid','loan_time']).sum()
temp = sum_t_loan.unstack()
temp.to_csv("temp.csv",index = True)


from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = np.nan,strategy = 'mean',axis = 1)
imp.fit(group)
group2 = imp.transform(group)

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(group2.loc[:,[0,1,2]],group2.loc[:,3],random_state = 1)
from sklearn.linear_model import LinearRegression 
linreg = LinearRegression()
linreg.fit(x_train,y_train)

print("阶距：",linreg.intercept_)
print("相似性",linreg.coef_)
y_pred = linreg.predict(x_test)
from sklearn import metrics
print(metrics.mean_squared_error(y_pred,y_test))
print(np.sqrt(metrics.mean_squared_error(y_pred,y_test)))
y_pred = pd.DataFrame(y_pred,index = y_test.index)
y_pred_12 = linreg.predict(group2[[1,2,3]])
group2.index = group.index
y_pred_12 = pd.DataFrame(y_pred_12,index = group2.index)
#一共有90993个人
temp = np.zeros((90993,1))
temp = pd.DataFrame(temp)
temp2 = pd.merge(temp,y_pred_12,left_index = True,right_index = True,how = 'outer')
temp2 = temp2[['0_y']]
temp2 = temp2.fillna(1)
temp2.to_csv('Des_pre.csv',index = True,header = False)
x = [1 for i in range(90993)]



#对用户表进行分析
import numpy as np
import pandas as pd
import os
#reset workspace
os.chdir("F:\\研究生\\京东金融2017\\Loan_Forecasting_Qualification")
t_user = pd.read_csv("t_user.csv")
t_user['active_date'] = t_user['active_date'].astype(np.datetime64)
t_user['active_date'].describe()
"""
Out[10]: 
count                   90993
unique                    212
top       2016-01-28 00:00:00
freq                     4298
first     2015-12-02 00:00:00
last      2016-06-30 00:00:00
Name: active_date, dtype: object
"""
last_active_user = t_user[t_user['active_date']=='2016-06-30 00:00:00']
#导入最近借款表
temp = pd.merge(last_active_user,temp,left_on = 'uid',right_index = True,how = 'inner')
temp.to_csv("last-active&loan-detail.csv")