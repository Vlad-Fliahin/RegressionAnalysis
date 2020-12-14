import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Fish.csv')
data = data.drop(['Length2', 'Length3', 'Height', 'Width'], axis=1)

x = np.array(data['Weight'].to_list()).reshape((-1, 1))
y = np.array(data['Length1'].to_list())
# print(x)
# print(y)

model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

print('intercept:', model.intercept_)
print('slope:', model.coef_)
y_pred = model.predict(x)
# print('predicted response:', y_pred, sep='\n')

import statsmodels.api as sm

x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
# print(model.summary())

import matplotlib.pyplot as plt

x = np.array(data['Weight'].to_list()).reshape((-1, 1))
y = np.array(data['Length1'].to_list())
plt.scatter(x, y)
plt.xlabel('Weight', fontsize=16)
plt.ylabel('Length', fontsize=16)
plt.plot(x, y_pred, color='green')

plt.show()
print('SSR: ', model.ssr)
print('SE: ', model.bse)
