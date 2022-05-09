# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:08:03 2022

@author: user
"""

import pandas as pd

df=pd.read_csv('covid.csv')
df.to_excel( 'covid.xlsx' , index=False )