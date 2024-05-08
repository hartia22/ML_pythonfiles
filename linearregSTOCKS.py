# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 07:26:55 2021

@author: Majic
"""

import pandas as pd
import yfinance as yf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
import datetime as dt
from scipy import stats 
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential 
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, Input




#mylist = ['NEE', 'REGI', 'FSLR', 'PLUG', 'HCA', 'AMC', 'FCEL' ]
mylist = ['PLUG']
main = []

for i in mylist:
    df = yf.download(i,period="1h",interval="5m",progress=False)
    df = df.reset_index()
    main.append(df)


buffer = main[0]
y = buffer['Adj Close']
x = buffer.drop(columns=['Datetime','Volume','Adj Close','Low','High','Close'])
buffer = buffer.drop(columns=['Volume','Low','High','Close'])
mms = StandardScaler()
#buffer['Volume'] = mms.fit_transform(buffer)

plt.figure(figsize=(10,10))
sns.lineplot(data = buffer)



model = LinearRegression()


#x['time'] = x['time'].map(dt.datetime.toordinal)     
#buffer = pd.DataFrame({'time':x['time'],'Adj':buff['Adj']}) 
           

y = np.array(y)
#x = np.array(x)
x['Open'] = x.index
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.05,shuffle=False)

model.fit(x_train,y_train)

print(model.score(x_train,y_train))

x_test = pd.DataFrame({'main':x_test.index}) 
y_predict = model.predict(x_test)
y_pred = model.intercept_ + model.coef_ * 78

sns.regplot(x=x_test.index,y=y_test)
sns.regplot(x=x_test.index,y=y_predict)

print('R2 score=', r2_score(y_test,y_predict))
print('Error margin = ', mean_squared_error(y_test,y_predict))



tfmodel = Sequential()

tfmodel.add(Dense(input_dim=1,units = 1activation='sigmoid'))
tfmodel.add(Dense(10,activation='relu'))
tfmodel.add(Dense(100,activation='relu'))
tfmodel.add(Dense(1,activation='relu'))

opt = Adam(learning_rate = .001)

tfmodel.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

history = tfmodel.fit(x_train,
                      y_train,
                      batch_size=32,
                      epochs=3,
                      validation_data=(x_test, y_test),
                      shuffle=True,
                      verbose=1)

tfmodel.summary()

'''
import tensorflow_probability as tfp

trend = tfp.sts.LocalLinearTrend(observed_time_series=x_test)
seasonal = tfp.sts.Seasonal(num_seasons=12, observed_time_series=x_test)
model = tfp.sts.Sum([trend, seasonal], observed_time_series=x_test)
'''
