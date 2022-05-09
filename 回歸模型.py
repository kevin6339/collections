# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 11:05:32 2022

@author: user
"""
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
df1=pd.read_excel('資料集.xlsx')
num1=df1['買賣件數']
num2=df1['買賣筆數']
num3=df1['買賣棟數']
covid=df1['covid19單月確診總量']
price=df1['房價']
series_dict1={'X1':num1,'y':price}
df1=pd.DataFrame(series_dict1)
X1=df1[['X1']]
y1=df1[['y']]
regr1=linear_model.LinearRegression()
regr1.fit(X1, y1)
print(regr1.coef_)
print(regr1.intercept_)


plt.scatter(X1, y1, color='black')
plt.plot(X1, regr1.predict(X1),color='blue',linewidth=1)
plt.show()

series_dict2={'X2':num2,'y':price}
df2=pd.DataFrame(series_dict2)
X2=df2[['X2']]
y2=df2[['y']]
regr2=linear_model.LinearRegression()
regr2.fit(X2, y2)
print(regr2.coef_)
print(regr2.intercept_)

plt.scatter(X2, y2, color='black')
plt.plot(X2, regr2.predict(X2),color='blue',linewidth=1)
plt.show()


series_dict3={'X3':num3,'y':price}
df3=pd.DataFrame(series_dict3)
X3=df3[['X3']]
y3=df3[['y']]
regr3=linear_model.LinearRegression()
regr3.fit(X3, y3)
print(regr3.coef_)
print(regr3.intercept_)


plt.scatter(X3, y3, color='black')
plt.plot(X3, regr3.predict(X3),color='blue',linewidth=1)
plt.show()


series_dict4={'covid':covid,'y':price}
df4=pd.DataFrame(series_dict4)
x4=df4[['covid']]
y4=df4[['y']]
regr4=linear_model.LinearRegression()
regr4.fit(x4, y4)
print(regr4.coef_)
print(regr4.intercept_)
plt.scatter(x4, y4, color='black')
plt.plot(x4, regr4.predict(x4),color='blue',linewidth=1)
plt.show()

series_dict5={'covid':covid,'件數':num1,'y':price}
df5=pd.DataFrame(series_dict5)
x5=df5[['covid','件數']]
y5=df5[['y']]
regr5=linear_model.LinearRegression()
regr5.fit(x5, y5)
print(regr5.coef_)
print(regr5.intercept_)

