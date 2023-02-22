import pandas as pd
import numpy as mp
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pd.read_csv("C:\\Users\\Illia\\Desktop\\real_estate_price_size.csv")

print(data.describe())
y = data['price']
x1 = data['size']
x = sm.add_constant(x1)
result = sm.OLS(y,x).fit()
print(result.summary())


plt.scatter(x1,y)
yhat = 223.1787*x1 + 1.019e+05
fig = plt.plot(x1,yhat, lw=4, c='green', label='regression line')
plt.xlabel('size', fontsize=20)
plt.ylabel('price', fontsize=20)


plt.show()