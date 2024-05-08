# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 11:52:37 2021

@author: Majic
"""

import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential          
from tensorflow.keras.layers import Conv2D             
from tensorflow.keras.layers import MaxPooling2D  
from tensorflow.keras.layers import GlobalMaxPooling2D     
from tensorflow.keras.layers import Flatten             
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.layers import BatchNormalization 
from sklearn.model_selection import train_test_split
from skimage.transform import resize as imresize




x = np.load('images.npy')
x.shape
y = pd.read_csv('Labels.csv')

#y = pd.get_dummies(y)


#creating my validation set
#x_train, x_val,y_train,y_val = train_test_split(x,y,test_size=0.1, random_state = 22)
#creating my test and train sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 22)

x_train.shape
y_train.shape



for i in range(4):
    plt.imshow(x[i + (i*100)])
    plt.show()

# Convert labels to one hot vectors.

from sklearn.preprocessing import LabelBinarizer
enc = LabelBinarizer()
y_train = enc.fit_transform(y_train)
y_test = enc.fit_transform(y_test)

print(y_train.shape)
print(y_test.shape)

'''
c = y_train.columns
for i in c:
    print(y[i], ' : ',y[i].value_counts())
    '''

#Creating a model

model = Sequential()

model.add(Dense(32,activation='relu',input_shape = (128,128,3)))
model.add(BatchNormalization())
model.add(Dropout(rate = 0.2))

model.add(Conv2D(32, kernel_size = (2,2),activation='relu'))

model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(BatchNormalization())
model.add(Dropout(rate = 0.3))

model.add(Flatten())
model.add(Dropout(rate = 0.2))
model.add(Dense(12, activation = 'relu'))

#compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()



x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')
x_train /= 255.0 
x_test /= 255.0

x_train = x_train.reshape(x_train.shape[0], 128, 128, 3)
x_test = x_test.reshape(x_test.shape[0], 128, 128, 3)

opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
#model.fit(x,epochs=10)

print(x_train.shape)
print(x_test.shape)



history = model.fit(x_train,
                    y_train,
                    batch_size=32,
                    epochs=3,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    verbose=1)
#model.fit( x=x, batch_size=32, epochs=10, validation_split = 0.3)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

