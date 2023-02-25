import pandas as pd
import numpy as mp
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

raw_data = pd.read_csv("C:\\Users\\Illia\\Desktop\\real_estate_price_size_year_view.csv")
data = raw_data.copy()

data['view'] = data['view'].map({'No sea view': 0, 'Sea view': 1})

y = data['price']
x1 = data[['size', 'year', 'view']]
x = sm.add_constant(x1)
result = sm.OLS(y,x).fit()
print(result.summary())


# plt.scatter(data['size'], y)
# yhat_no_sea_view = -5398000 + 223.0316*data['size'] + 2718.9489*data['year']
# yhat_sea_view = -5341270 + 223.0316*data['size'] + 2718.9489*data['year']
# fig = plt.plot(data[['size', 'year']], yhat_no_sea_view, lw=2, c='green')
# fig = plt.plot(data[['size', 'year']], yhat_sea_view, lw=2, c='yellow')
# plt.xlabel('size, year', fontsize=20)
# plt.ylabel('price', fontsize=20)
# plt.show()
