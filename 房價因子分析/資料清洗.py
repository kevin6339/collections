# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:05:37 2022

@author: user
"""

import pandas as pd
import numpy as np
from pandas.io import sql
import pymysql
import date


number=[]
df=pd.read_excel('covid19統計.xlsx')
All_date=df[date.month('通報日')]
All_price=df['Total']



for i in range(0,839):
    number.append(str(i))
dict1={'順序':number,'時間':All_date , '房價': All_price }
df1 = pd.DataFrame(dict1)
df1.to_excel( '宜蘭市房價.xlsx' , index=False )
connection = pymysql.connect(host='127.0.0.1',
                              user='root',
                              password='@Ad25817593',
                              db='travels')
sql = "INSERT INTO `covid19` (`順序`,`日期`, `總量`) VALUES ( %s,%s,%s)"
cursor = connection.cursor()
for b,c,d in zip(number,All_date,All_price ):
    cursor.execute(sql,(str(b),str(c),int(d)))
    connection.commit()

sql = "SELECT * FROM `taichung`"
cursor.execute(sql)




    

   
    