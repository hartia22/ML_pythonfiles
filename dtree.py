# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 19:47:21 2020

@author: Nautilus
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from IPython.display import Image  
#import pydotplus as pydot
from sklearn import tree
from os import system

df = pd.read_csv('C:\\Users\\Nautilus\\Downloads\\pima-indians-diabetes (1).csv')

s = df.columns
'''
for i in s:
    if(df[i].value_counts().count() < 100):
        print(i,' ',df[i].value_counts())
        '''

#print('\n',df.describe())


#df['Preg'] = df['Preg'].astype('category')
#df['age'] = df['age'].astype('category')

#print(df.info())

#df_dummies = pd.get_dummies(df)

#print(df_dummies.columns)

x = df_dummies.drop(['class'], axis = 1)
y = df_dummies['class']

print(x.columns)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30, random_state=7)

dtree = DecisionTreeClassifier(criterion='gini', random_state = 7, max_depth=3)

dtree.fit(x_train,y_train)

print(dtree.score(x_train,y_train))
print(dtree.score(x_test,y_test))

train_char_label = ['No', 'Yes']
Credit_Tree_File = open('credit_tree.dot','w')
dot_data = tree.export_graphviz(dtree, out_file=Credit_Tree_File, feature_names = list(x_train), class_names = list(train_char_label))
Credit_Tree_File.close()

print(dtree.score(x_test , y_test))
y_predict = dtree.predict(x_test)

cm=metrics.confusion_matrix(y_test, y_predict, labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')

