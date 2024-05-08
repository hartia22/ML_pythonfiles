# -*- coding: utf-8 -*-
"""
Created on Sat May 22 22:10:00 2021

@author: Nautilus

"""
import pandas as pd
import yfinance as yf
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential 
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
#import cryptocompare
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

'''
cryptocompare.get_historical_price_minute('BTC', currency='EUR')
cryptocompare.get_historical_price_minute('BTC', currency='EUR', limit=1440)
cryptocompare.get_historical_price_minute('BTC', 'EUR', limit=24, exchange='CCCAGG', toTs=datetime.datetime.now())
'''

#mylist = ['NEE', 'REGI', 'FSLR', 'PLUG', 'HCA', 'AMC', 'FCEL' ]
mylist = ['HCA']
main = []

for i in mylist:
    df = yf.download(i,period="2d",interval="1m",progress=False)
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
    rolmean = timeseries.rolling(10).mean()
    rolstd = timeseries.rolling(10).std()
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

rh = 10
rk = 9
buff_index = max(df.index)
index_list = df[df.index<=buff_index-9].index


for i in range(rh):
    t = df[(df.index<=buff_index - (rk - i)) & (df.index >= i) ]['Close']
    t = t.reset_index()
    t = t.drop(columns=['index'])
    main.append(t)


df_T = pd.DataFrame({'t1':main[0]['Close'],
                     't2':main[1]['Close'],
                     't3':main[2]['Close'],
                     't4':main[3]['Close'],
                     't5':main[4]['Close'],
                     't6':main[5]['Close'],
                     't7':main[6]['Close'],
                     't8':main[7]['Close'],
                     't9':main[8]['Close'],
                     'tr':main[9]['Close']},index =  index_list)



x = df_T.drop(columns=['tr'])
y = df_T['tr']

ss = StandardScaler()

x_val = x.iloc[max(x.index)-11:]
x = x.iloc[:max(x.index)-11]
y_val = y.iloc[max(y.index)-11:]
y = y.iloc[:max(y.index)-11]

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=22, test_size = .2, shuffle=False)

x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)

y_train =  np.array(y_train)
y_test =  np.array(y_test)

model = Sequential() 

opt = Adam(learning_rate=.001)

init = initializers.TruncatedNormal(mean=0.0, stddev=0.001)
model.add(Dense(input_dim=9,units = 6, kernel_initializer = init))
model.add(Dense(45,activation='ReLU',kernel_initializer = init, use_bias=False))
model.add(Dense(90,activation='ReLU',kernel_initializer = init, use_bias=False))
model.add(Dense(1,activation='linear',kernel_initializer = 'normal', activity_regularizer=regularizers.l2(0.001)))

model.compile(loss='Huber',optimizer=opt,metrics= ['MeanSquaredError'])

epoch=100

print(model.summary())

model_f = model.fit(x_train,y_train,epochs=epoch,batch_size=32,verbose=1)


y_predict = model.predict(x_test)


mean_squared_error(y_predict,y_test)

y_predict_graph = pd.DataFrame(y_predict, columns=['main'])
y_test_graph = pd.DataFrame(y_test, columns=['main'])
y_graph = pd.DataFrame({'original':y_test_graph['main'],
                        'prediction':y_predict_graph['main']})


sns.lineplot(data = y_graph)

