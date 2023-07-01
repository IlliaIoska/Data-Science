import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
import seaborn as sb
sb.set()

data = pd.read_csv('mupltiple_linear_regression.csv')

print(data.head())

x = data[['SAT', 'Rand 1,2,3']]
y = data['GPA']

linReg = LinearRegression()
linReg.fit(x,y)

print('SAT coefficient = ', linReg.coef_[0])
print('Rands 1,2,3 coefficient = ', linReg.coef_[1])
print('intercept = ', linReg.intercept_)

r_squared = linReg.score(x,y)

observations_num = x.shape[0]
params_num = x.shape[1]

adjusted_r_squared = 1 - (1 - r_squared)*(observations_num-1)/(observations_num-params_num-1)

print('adjusted R^2 = ', adjusted_r_squared)

p_values = f_regression(x,y)[1]

linRegSummary = pd.DataFrame(data= x.columns.values, columns=['Features'])

linRegSummary['P-values'] = p_values.round(3)

print(linRegSummary)






 


