import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sb
sb.set()

data = pd.read_csv('1.01.+Simple+linear+regression.csv')

print(data.head())

x = data['SAT']
y = data['GPA']
x_matrix = x.values.reshape(-1,1)

linReg = LinearRegression()
linReg.fit(x_matrix,y)

print(linReg.score(x_matrix,y))
print(linReg.coef_)
print(linReg.intercept_)

new_data = pd.DataFrame(data=[1740,1760], columns=['SAT'])

new_data['predicted GPA'] = linReg.predict(new_data)

print(new_data)

 


