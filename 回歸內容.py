# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 21:59:26 2022

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
df1=pd.read_excel('資料集.xlsx')
num1=df1['買賣件數']
covid=df1['covid19單月確診總量']
price=df1['房價']
series_dict5={'covid':covid,'件數':num1,'y':price}
df=pd.DataFrame(series_dict5)
x=df[['covid','件數']]
y=df['y']

modelRegL = LinearRegression()  # 建立線性迴歸模型
modelRegL.fit(x, y)  # 模型訓練：資料擬合
yFit = modelRegL.predict(x)  # 用迴歸模型來預測輸出
# 輸出迴歸結果 # YouCans, XUPT
print("\nModel1: Y = b0 + b1*x1 + b2*x2")
print('迴歸截距: w0={}'.format(modelRegL.intercept_))  # w0: 截距
print('迴歸係數: w1={}'.format(modelRegL.coef_))  # w1,..wm: 迴歸係數
# 迴歸模型的評價指標
print('R2 確定係數：{:.4f}'.format(modelRegL.score(x, y)))  # R2 判定係數
print('均方誤差：{:.4f}'.format(mean_squared_error(y, yFit)))  # MSE 均方誤差
print('平均絕對值誤差：{:.4f}'.format(mean_absolute_error(y, yFit)))  # MAE 平均絕對誤差
print('中位絕對值誤差：{:.4f}'.format(median_absolute_error(y, yFit)))  # 中值絕對誤差


m = x.shape[1]
n = x.shape[0]
yMean = np.mean(y)
SST = sum((y-yMean)**2)  # SST: 總平方和
SSR = sum((yFit-yMean)**2)  # SSR: 迴歸平方和
SSE = sum((y-yFit)**2)  # SSE: 殘差平方和
Fstats = (SSR/m) / (SSE/(n-m-1))  # F 統計量
probFstats = stats.f.sf(Fstats, m, n-m-1)  # F檢驗的 P值
print('F統計量：{:.4f}'.format(Fstats))
print('FF檢驗的P值：{:.4e}'.format(probFstats))

# 繪圖：原始資料點，擬合曲線
fig, ax = plt.subplots(figsize=(8, 6))  # YouCans, XUPT
ax.plot(range(len(y)), y, 'b-.', label='Sample')  # 樣本資料
ax.plot(range(len(y)), yFit, 'r-', label='Fitting')  # 擬合資料
ax.legend(loc='best')  # 顯示圖例
plt.title('Regression analysis with sales of toothpaste by SKlearn')
plt.xlabel('period')
plt.ylabel('sales')
plt.show()