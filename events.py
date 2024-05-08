# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 19:06:26 2020

@author: Nautilus
"""

import pandas as pd 
import numpy as np

df_main = pd.read_csv('securityevents1.csv', skiprows=0, index_col=False)

df_main_short = df_main.copy()
#print(df_main.columns)
#print(df_main_short['Task Category'])
series = df_main_short['Desc']

series3 = df_main_short['Event ID']
#print(series)
series2 = df_main_short['Date and Time']
data = {'Desc': series,
        'EvenID': series3,
        'DateTime': series2}
#print(df_main_short.set_index(['Task Category']).count(axis='columns'))
df = pd.DataFrame(data)

df.Desc = df.Desc.str[0:100]
df.Desc = df.Desc.str.replace('\n', '-' )
df.Desc = df.Desc.str.replace('\t', '' )
df.Desc = df.Desc.str.replace('\r', '' )

new_df = df.groupby('Desc')


print(df.DateTime.value_counts(normalize=True))

#print(new_df[])
#print(df.set_index(['Desc'],['Date and Time']).count(axis='columns'))
#series3 = {'Unique': df['Desc'].unique()}
#print(df.desc.value_counts())
#df_main = pd.read_csv("C:\Users\Nautilus\Documents\eventlog.csv")
    
    