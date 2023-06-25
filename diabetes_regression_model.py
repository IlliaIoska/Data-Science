import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pd.read_csv(r"D:\pima_indians_diabetes\diabetes.csv")

print(data.describe())

y = data["Outcome"]
x1 = data["Glucose"]

plt.scatter(x1,y)
plt.xlabel("Glucose", fontsize=20)
plt.ylabel("Diabetes", fontsize=20)
plt.show()

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print(results.summary())