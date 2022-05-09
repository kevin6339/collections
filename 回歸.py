#!/usr/bin/env python
# coding: utf-8

# In[16]:


import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


# In[20]:


df1=pd.read_excel("C:/Users/user/Desktop/新增資料夾 (2)/資料集.xlsx")
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


# In[6]:


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


# In[8]:


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


# In[9]:


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


# In[10]:


series_dict5={'covid':covid,'件數':num1,'y':price}
df5=pd.DataFrame(series_dict5)
x5=df5[['covid','件數']]
y5=df5['y']
regr5=linear_model.LinearRegression()
regr5.fit(x5, y5)
print(regr5.coef_)
print(regr5.intercept_)


# In[25]:


df1=pd.read_excel("C:/Users/user/Desktop/新增資料夾 (2)/資料集.xlsx")
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


# In[26]:


df1=pd.read_excel("C:/Users/user/Desktop/新增資料夾 (2)/資料集.xlsx")
num1=df1['買賣件數']
covid=df1['covid19單月確診總量']
price=df1['房價']
series_dict5={'covid':covid,'件數':num1,'y':price}
df=pd.DataFrame(series_dict5)
x=df[['covid','件數']]
y=df['y']
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


# In[ ]:





# In[ ]:





# In[ ]:




