import numpy as np
import pandas as pd
from collections import Counter

data_path = '../data/'
user_path = data_path + 't_user.csv'
order_path = data_path + 't_order.csv'
click_path = data_path + 't_click.csv'
loan_path = data_path + 't_loan.csv'
loan_sum_path = data_path + 't_loan_sum.csv'

user = pd.read_csv(user_path)
order = pd.read_csv(order_path)
click = pd.read_csv(click_path)
loan = pd.read_csv(loan_path)
loan_sum = pd.read_csv(loan_sum_path)

# 对金额进行处理
user['limit'] = np.round(5**(user['limit'])-1,2)
order['price'] = np.round(5**(order['price'])-1,2)
loan['loan_amount'] = np.round(5**(loan['loan_amount'])-1,2)
loan_sum['loan_sum'] = np.round(5**(loan_sum['loan_sum'])-1,2)

a = loan[loan['loan_time']>='2016-11-03 00:00:00'].groupby('uid',as_index=False)['loan_amount'].sum()
b = loan_sum.merge(a,on='uid',how='outer')
c = b[b['loan_sum']!=b['loan_amount']]
print(c.shape)  #不同的个数
















