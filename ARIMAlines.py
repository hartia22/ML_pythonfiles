# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 10:16:53 2022

@author: adria
"""

import pandas as pd
import yfinance as yf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
import math
from sklearn.preprocessing import MinMaxScaler 

'''
cryptocompare.get_historical_price_minute('BTC', currency='EUR')
cryptocompare.get_historical_price_minute('BTC', currency='EUR', limit=1440)
cryptocompare.get_historical_price_minute('BTC', 'EUR', limit=24, exchange='CCCAGG', toTs=datetime.datetime.now())
'''

#mylist = ['NEE', 'REGI', 'FSLR', 'PLUG', 'HCA', 'AMC', 'FCEL' ]
mylist = ['HCA']
main = []

for i in mylist:
    df = yf.download(i,period="4h",interval="1m",progress=False)
    df = df.reset_index()


'''
df['intdex'] = df.index/12
df['intdex'] = df['intdex'].astype('int')
'''
df.shape
df.head
df.columns
df['Open'].plot()
plt.title('ScatterPlot')
plt.show()

sns.distplot(df['Close'])


def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(25).mean()
    rolstd = timeseries.rolling(25).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)
    
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)
    
test_stationarity(df['Close'])

df_diff = np.diff(df['Close'])
#df['Close'] = np.append([0],df_diff)

df['Close'].plot()

test_stationarity(df['Close'])
'''
mm = MinMaxScaler() 
newdf = df
newdf = newdf.drop(labels=['Datetime', 'Open', 'High', 'Low',  'Adj Close', 'Volume'], axis=1)
newdf['Close'] = mm.fit_transform(newdf)

result = seasonal_decompose(newdf['Close'], model='multiplicative', freq = 30)
fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(16, 9)
'''
df_log = df
df_log = np.log(df['Close'])

df_log.plot()


df_std = df_log.rolling(9).std()
df_mean = df_log.rolling(9).mean()

plt.plot(df_std,color="blue",label = "STD")
plt.plot(df_mean,color="red", label = "Mean")
plt.show()

train, test = train_test_split(df_log,test_size=.1, shuffle = False)

plt.plot(train,color="Red",label="Train")
plt.plot(test,color="Blue",label="Test")
plt.legend()
plt.show

model_autoARIMA = auto_arima(train, start_p=0, start_q=0,
                      test='adf',       # use adftest to find             optimal 'd'
                      max_p=4, max_q=4, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
print(model_autoARIMA.summary())


model_autoARIMA.plot_diagnostics(figsize=(15,8))
plt.show()


model = ARIMA(train, order=(2,1,0))  
fitted = model.fit(disp=-1)  
print(fitted.summary())

train_count = test.size

fc, se, conf = fitted.forecast(train_count, alpha=0.01)  # 95% confidence
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, color = 'blue', label='Actual Stock Price')
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title('Altaba Inc. Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# report performance
mse = mean_squared_error(test, fc)
print('MSE: '+str(mse))
mae = mean_absolute_error(test, fc)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test, fc))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(fc - test)/np.abs(test))
print('MAPE: '+str(mape))

