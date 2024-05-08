# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 20:12:10 2020

@author: Majic
"""



import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split 
import tensorflow as tf
from sklearn import metrics



pd.set_option('display.max_columns',None)
df = pd.read_csv('C:\\Users\\harti\\Downloads\\Churn_Modelling.csv')

print(df.info())
print(df.describe())



df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
#df['NumOfProduct'] = df['NumOfProducts'].astype('category')
#hunting for outliers

e_df = pd.get_dummies(df,drop_first=True)
e_df.columns
e_df.info()

s = e_df.columns 
for i in s :
    sns.boxplot(e_df[i])
    plt.show()
    
    
plt.figure(figsize = (12,12))
sns.heatmap(e_df.corr(),annot=True)


# exited seems to be the target          
x = e_df.drop(columns=['Exited'])

y = e_df.drop(columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Geography_Germany',
       'Geography_Spain', 'Gender_Male'])

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.30, random_state=7)
              
ss = StandardScaler()
#these values are really large when compared to the rest of the values in the dataset 
# as well as having a very uneven distribution within themselves 

x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)

y_train =  np.array(y_train)
y_test =  np.array(y_test)

print(f'training shapes: {x_train.shape}, {y_train.shape}')
print(f'testing shapes: {x_test.shape}, {y_test.shape}')



from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
 
model = Sequential()


model.add(Dense(activation='relu', input_dim = 11, units = 6,kernel_initializer='normal'))
model.add(Dense(22,activation='sigmoid',kernel_initializer='normal'))
model.add(Dense(33,activation='sigmoid',kernel_initializer='normal'))
model.add(Dense(44,activation='sigmoid',kernel_initializer='normal'))
model.add(Dense(55,activation='sigmoid',kernel_initializer='normal'))
model.add(Dense(66,activation='sigmoid',kernel_initializer='normal'))

model.add(Dense(1, activation='relu',kernel_initializer='normal'))

opt = tf.keras.optimizers.Adam(.001)

model.compile(loss='binary_crossentropy',optimizer=opt,metrics= ['accuracy'])

epoch=50

model_f = model.fit(x_train,y_train,epochs=epoch,batch_size=128,verbose=1)


y_predict = model.predict(x_test)

y_predict = (y_predict > .5)

cm = metrics.confusion_matrix(y_test,y_predict)
print(cm)

cr=metrics.classification_report(y_test,y_predict)
print(cr)

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True)


s