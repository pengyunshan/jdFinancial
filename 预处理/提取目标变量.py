import numpy as np
import pandas as pd
import os

os.chdir("F:\\研究生\\京东金融2017\\Loan_Forecasting_Qualification")
t_loan = pd.read_csv("t_loan.csv" ,parse_dates = True)
t_loan['loan_time'] = t_loan['loan_time'].astype(np.datetime64)
t_loan['month'] = t_loan['loan_time'].apply(lambda x:x.month)
temp = t_loan[['uid','month','loan_amount']]
temp = temp.groupby(['uid','month']).sum()
temp = temp.unstack()
temp.columns = ['8','9','10','11']
temp.to_csv("target.csv")

t_loan_sum = pd.read_csv('t_loan_sum.csv',parse_dates = True)
t_loan_sum['month'] = t_loan_sum['month'].astype(np.datetime64)
t_loan_sum.set_index(['uid'],inplace = True)
t_loan_sum = t_loan_sum['loan_sum']
t_loan_sum = pd.DataFrame(t_loan_sum)
temp2 = pd.merge(t_loan_sum,temp,left_index = True,right_index = True,how = 'outer')
temp2['chafen'] = temp2['loan_sum'] - temp2['11']

