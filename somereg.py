# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 11:29:29 2021

@author: Nautilus
"""

from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt 
import seaborn as sns 
import yfinance as yf 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error


data = yf.download(tickers='GME',period='2d', interval='15m')
data = data.reset_index()
data.head()
data.columns
#sns.regplot(x=data.index,  y=data['Adj Close'])
#sns.regplot(x=data.index,  y=data['High'])
#sns.regplot(x=data.index,  y=data['Low'])
data['temp'] = data.index

x=pd.DataFrame({'temp':data.index, 'low':data['Low'],'high':data['High']})
y=pd.DataFrame({'adj':data['Adj Close']})

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2,shuffle=False)


model = LinearRegression(normalize=True) 

model.fit(x_train,y_train)

model.score(x_train,y_train)

y_pred = model.predict(x_test)

print('Mean squared error', mean_squared_error(y_test,y_pred))
print('r2_score', r2_score(y_test,y_pred))
print('Mean absolute error', mean_absolute_error(y_test,y_pred))


sns.regplot(x=x_test.index,y=y_test)
sns.regplot(x=x_test.index, y=y_pred)
