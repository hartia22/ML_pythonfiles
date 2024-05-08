# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 12:44:14 2021

@author: Majic
"""

import re
from bs4 import BeautifulSoup                           
import contractions 
import numpy as np                                      
import pandas as pd                                     
import nltk                                            
import seaborn as sns
#nltk.download('stopwords')                             
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.corpus import stopwords                       
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer   
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE

pd.set_option('display.max_columns',None)

data = pd.read_csv("C:\\Users\\harti\\Downloads\\Tweets.csv")

data.head()

data = data.drop('tweet_id', axis=1)
c = data.columns

'''
for i in c:
    print(data[i].value_counts())
  '''  
#grab the tweets with their sentiments 
tweets = data.drop(['airline_sentiment_confidence', 'negativereason',
       'negativereason_confidence', 'airline', 'airline_sentiment_gold',
       'name', 'negativereason_gold', 'retweet_count', 'tweet_coord',
       'tweet_created', 'tweet_location', 'user_timezone'], axis=1)
data = data.drop(['airline_sentiment', 'text', 'tweet_coord',
       'tweet_created', 'tweet_location','negativereason_confidence', 
       'airline_sentiment_gold', 'negativereason_gold','user_timezone','name'],axis=1)

data = pd.get_dummies(data)
ss = MinMaxScaler()
data['retweet_count'] = ss.fit_transform(data[['retweet_count']])

'''
c = data.columns
for i in c:
    print(data[i].value_counts())
'''

tweets.head()
tweets.shape

def rem_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

tweets['text'] = tweets['text'].apply(lambda x: rem_html(x))

def rem_contra(text):
    return contractions.fix(text)

def rem_ats_specs(text):
  text = re.sub('[^a-zA-Z]+', ' ', text)
  return text

def to_lower(text):
    new_words = []
    for word in text:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

stopwords = stopwords.words('english')
def rem_stopwords(text):
    new_words = []
    for word in text:
        if word not in stopwords:
            new_words.append(word)
    return new_words

lemm = WordNetLemmatizer()
def lemmatize_list(text):
    new_words = []
    for word in text:
      new_words.append(lemm.lemmatize(word, pos='v'))
    return new_words

#Remove contractions
tweets['text'] = tweets['text'].apply(lambda x: rem_contra(x))
#remove special characters
tweets['text'] = tweets['text'].apply(lambda x: rem_ats_specs(x))
#tokenize text
tweets['text'] = tweets.apply(lambda x:nltk.word_tokenize(x['text']),axis=1)
#lower case transform
tweets['text'] = tweets['text'].apply(lambda x: to_lower(x))
#remove stop words 
tweets['text'] = tweets['text'].apply(lambda x: rem_stopwords(x))
#lemmatize 
tweets['text'] = tweets['text'].apply(lambda x: lemmatize_list(x))

def normalize(words):
    return ' '.join(words)

tweets['text'] = tweets['text'].apply(lambda x: normalize(x))

tweets.head()

tweets['airline_sentiment'].value_counts()
labels = tweets.drop(['text'], axis=1)
labels.loc[labels['airline_sentiment'] == 'negative'] = 0
labels.loc[labels['airline_sentiment'] == 'neutral'] = 1
labels.loc[labels['airline_sentiment'] == 'positive'] = 2


labels['airline_sentiment'] = labels['airline_sentiment'].astype(float)
#labels = np.dtype(float)
labels.value_counts()

vectorizer = TfidfVectorizer(max_features=1000)
data_features = vectorizer.fit_transform(tweets['text'])
#data_features = pd.merge(data_features,data)
data_features = data_features.toarray()

mms = MinMaxScaler()
data_features = mms.fit_transform(data_features)

x_train, x_test,y_train,y_test = train_test_split(data_features,labels,test_size = 0.20,random_state=22)

print(f'training shapes: {x_train.shape}, {y_train.shape}')
print(f'testing shapes: {x_test.shape}, {y_test.shape}')


sm = SMOTE(sampling_strategy="auto",random_state=1,k_neighbors=5)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

#x_super, y_super = sm.fit_sample(x_train,y_train)

x_super, y_super = sm.fit_sample(x_train,y_train)
print(f'testing shapes: {x_super.shape}, {y_super.shape}')

'''
forest = RandomForestClassifier(criterion = "gini", n_estimators=10, n_jobs=4, warm_start="true")

forest = forest.fit(x_train,y_train.ravel())

print(forest)

print(np.mean(cross_val_score(forest, x_train, y_train.ravel(), cv=10)))

results = forest.predict(x_test)

cr = classification_report(y_test, results)
print(cr)
cm = confusion_matrix(y_test,results)

sns.heatmap(cm,annot=True)
#adding the rest of the data to the classifier 

print(f'training shapes: {x_train.shape}, {y_train.shape}')
print(f'testing shapes: {x_test.shape}, {y_test.shape}')
'''
dim = x_train.shape[1]


#Tensor Flow Model 
tfmodel = Sequential()

tfmodel.add(Dense(5,activation='relu',input_dim=dim,kernel_initializer='normal'))
tfmodel.add(Dense(10,activation='sigmoid',input_dim=dim,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))
tfmodel.add(Dense(1,activation='sigmoid'))



opt = SGD(lr=0.001)
opt2 = Adam(learning_rate= 0.002)

tfmodel.compile(optimizer=opt2,loss = 'binary_crossentropy', metrics=['accuracy'])

tfmodel.summary()

tfmodel.fit(x_super,y_super,epochs=50,batch_size=64, validation_data=(x_test,y_test))

y_predict = tfmodel.predict(x_test)
y_predict[y_predict > .5] = 1.0
y_predict[y_predict < .2] = 0.0
y_predict[(y_predict >= .2) & (y_predict  <= .5)] = 2.0

loss, accuracy = tfmodel.evaluate(x_super, y_super, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = tfmodel.evaluate(x_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

from sklearn.metrics import classification_report
cr=classification_report(y_test,y_predict)
print(cr)

cm2 = confusion_matrix(y_test,y_predict)

sns.heatmap(cm2,annot=True)
