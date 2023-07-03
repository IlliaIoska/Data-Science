import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sb
sb.set()

data = pd.read_csv('binary_predictors.csv')

data_copy = data.copy()
data_copy['Admitted'] = data_copy['Admitted'].map({'Yes':1,'No':0})
data_copy['Gender'] = data_copy['Gender'].map({'Female':1,'Male':0})

y = data_copy['Admitted']
x1 = data_copy[['SAT','Gender']]
x0 = sm.add_constant(x1)
log_reg = sm.Logit(y,x0)
log_reg_results = log_reg.fit()

print(log_reg_results.summary())