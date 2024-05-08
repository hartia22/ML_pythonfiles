# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:26:48 2020

@author: Nautilus
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix

pd.set_option('display.max_columns', None)

df = pd.read_csv('bank-full.csv')
# as per the data-set attribute information recomendation 
# the Duration heavily affects the outcome of our model and as such
# it will be dropped to have a more realistic model
# duration of 0 means the client has not been approached for a subscription
# there fore the target will always be 0)

#df = df.drop('duration', axis = 1)


df.head()
df.shape
df.dtypes
df.isna().sum()


s = df.columns 
for i in s:
    print(i,'\n',df[i].value_counts())

# default, target, Loan, housing housing can be changed to a binary
df.loc[df['Target']=='yes','Target'] = 1  
df.loc[df['Target']=='no','Target'] = 0   
df.loc[df['default']=='yes','default'] = 1  
df.loc[df['default']=='no','default'] = 0   
df.loc[df['housing']=='yes','housing'] = 1  
df.loc[df['housing']=='no','housing'] = 0    
df.loc[df['loan']=='yes','loan'] = 1  
df.loc[df['loan']=='no','loan'] = 0   
 



#there is an individual with 275 previous calls that might impact our model negatively 
print(df[df['previous']==275])
df.loc[29182,'previous'] = df['previous'].median()
df.describe()
#
print(df.skew())

print(df[df['pdays']==-1])

#UNIvariate and Bivariate analysis 
#df['job'] = df['job'].astype('category')
for i in s:
    if df[i].dtype != 'object':
        sns.distplot(df[i],kde=False) 
        plt.show()

#the distribution for column 'day' seems like it will have relatively low impact
# will probably drop after heatmap  
# balance also seems heavily skewed as most balance values are 0 but the max and 
# min are very far appart from each other. 
        

for i in s:
    if df[i].dtype == 'object':
        sns.countplot(df[i])
        plt.show()

#turn remaining objects into categories for encoding purposes
# there is an interesting distribution when observing the month
# I will turn it into a continuous variable to run further analysis 


df.loc[df['month']=='jan','month'] = 1
df.loc[df['month']=='feb','month'] = 1
df.loc[df['month']=='mar','month'] = 1
df.loc[df['month']=='apr','month'] = 2
df.loc[df['month']=='may','month'] = 2
df.loc[df['month']=='jun','month'] = 2
df.loc[df['month']=='jul','month'] = 3
df.loc[df['month']=='aug','month'] = 3
df.loc[df['month']=='sep','month'] = 3
df.loc[df['month']=='oct','month'] = 4
df.loc[df['month']=='nov','month'] = 4
df.loc[df['month']=='dec','month'] = 4


for i in s:
   if df[i].dtype == 'object':
       df[i] = df[i].astype('category')     







df['Target'] = df['Target'].astype('float')
#df['Target'].value_counts()
sns.heatmap(df.corr(), annot=True)
#sns.pairplot(df, hue='Target',diag_kind="hist")

#thanks to the pair plot we can see that the day of the month 
#seems to be spread evenly amongs our target variables 
#we should be able to drop it along with the month
df = df.drop('day', axis= 1)
df = df.drop('month', axis = 1)
scaler = PowerTransformer()

df['balance'] = scaler.fit_transform(df[['balance']])
df.loc[df['pdays']==-1,'pdays'] = 0
df['pdays'] = scaler.fit_transform(df[['pdays']])


df_dummies = pd.get_dummies(df, drop_first= True)
#plt.figure(figsize=(35,35))
#sns.heatmap(df_dummies.corr(), annot=True)

#sns.pairplot(df_dummies)
df_dummies['Target'] = df_dummies['Target'].astype('category')
x = df_dummies.drop('Target', axis = 1)
y = df_dummies['Target']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=7)

LRmodel = LogisticRegression(max_iter=1000,random_state = 13, penalty='l2', solver='liblinear',
                           multi_class='ovr',C=1.0,class_weight=None,
                           fit_intercept=True, dual=False, warm_start=False)

LRmodel.fit(x_train,y_train)

y_predict = LRmodel.predict(x_test)


cm=metrics.confusion_matrix(y_test, y_predict, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True)



results = pd.DataFrame({'model' : ['Logistic Regression'],
                       'score' : LRmodel.score(x_test, y_test),
                       'recall' : metrics.recall_score(y_test,y_predict),
                       'Precision' : metrics.precision_score(y_test,y_predict),
                       'F1 Score' : metrics.f1_score(y_test,y_predict),
                       'Roc Auc Score': metrics.roc_auc_score(y_test,y_predict)})
print(results)


DTCmodel = DecisionTreeClassifier(criterion='gini', max_depth=3)

DTCmodel.fit(x_train,y_train)
DTCpredict = DTCmodel.predict(x_test)
# Desicion
cm=metrics.confusion_matrix(y_test, DTCpredict, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True)

buffer = pd.DataFrame({'model' : ['Decision Tree'],
                       'score' : DTCmodel.score(x_test, y_test),
                       'recall' : metrics.recall_score(y_test,DTCpredict),
                       'Precision' : metrics.precision_score(y_test,DTCpredict),
                       'F1 Score' : metrics.f1_score(y_test,DTCpredict),
                       'Roc Auc Score': metrics.roc_auc_score(y_test,DTCpredict)})

results = pd.concat([results,buffer])

Bmodel = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),n_estimators=50, max_samples= .7, bootstrap=True, oob_score=True, random_state=7)
Bmodel.fit(x_train,y_train)

BmodelPredict = Bmodel.predict(x_test)

## Bagging model matrix 
cm=metrics.confusion_matrix(y_test, BmodelPredict, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True)

buffer = pd.DataFrame({'model' : ['Bagging Classifier'],
                       'score' : Bmodel.score(x_test, y_test),
                       'recall' : metrics.recall_score(y_test,BmodelPredict),
                       'Precision' : metrics.precision_score(y_test,BmodelPredict),
                       'F1 Score' : metrics.f1_score(y_test,BmodelPredict),
                       'Roc Auc Score': metrics.roc_auc_score(y_test,BmodelPredict)})

results = pd.concat([results,buffer])

RFmodel = RandomForestClassifier(n_estimators = 50,max_depth=4)
RFmodel.fit(x_train,y_train)
RFmodelPredict = RFmodel.predict(x_test)
## random forest class matrix

cm=metrics.confusion_matrix(y_test, RFmodelPredict, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True)

buffer = pd.DataFrame({'model' : ['Random Forest'],
                       'score' : RFmodel.score(x_test, y_test),
                       'recall' : metrics.recall_score(y_test,RFmodelPredict),
                       'Precision' : metrics.precision_score(y_test,RFmodelPredict),
                       'F1 Score' : metrics.f1_score(y_test,RFmodelPredict),
                       'Roc Auc Score': metrics.roc_auc_score(y_test,RFmodelPredict)})
results = pd.concat([results,buffer])


ADAmodel = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),
                          n_estimators = 100, learning_rate=0.1, random_state=7)
ADAmodel.fit(x_train, y_train)

ADAmodelPredict = ADAmodel.predict(x_test)
#ADA matrix
cm=metrics.confusion_matrix(y_test, ADAmodelPredict, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True)

buffer = pd.DataFrame({'model' : ['ADA Boost'],
                       'score' : ADAmodel.score(x_test, y_test),
                       'recall' : metrics.recall_score(y_test,ADAmodelPredict),
                       'Precision' : metrics.precision_score(y_test,ADAmodelPredict),
                       'F1 Score' : metrics.f1_score(y_test,ADAmodelPredict),
                       'Roc Auc Score': metrics.roc_auc_score(y_test,ADAmodelPredict)})
results = pd.concat([results,buffer])

ADAmodelLR = AdaBoostClassifier(base_estimator=LogisticRegression(max_iter = 1000))
ADAmodelLR.fit(x_train, y_train)

ADAmodelLRPredict = ADAmodel.predict(x_test)
## ADA with logistic regression matrix
cm=metrics.confusion_matrix(y_test, ADAmodelLRPredict, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True)

buffer = pd.DataFrame({'model' : ['ADA Boost LR'],
                       'score' : ADAmodelLR.score(x_test, y_test),
                       'recall' : metrics.recall_score(y_test,ADAmodelLRPredict),
                       'Precision' : metrics.precision_score(y_test,ADAmodelLRPredict),
                       'F1 Score' : metrics.f1_score(y_test,ADAmodelLRPredict),
                       'Roc Auc Score': metrics.roc_auc_score(y_test,ADAmodelLRPredict)})
results = pd.concat([results,buffer])

Gmodel = GradientBoostingClassifier()
Gmodel.fit(x_train,y_train)
GmodelPredict = Gmodel.predict(x_test)
## Matrix from Ggradient model
cm=metrics.confusion_matrix(y_test, GmodelPredict, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True)

buffer = pd.DataFrame({'model' : ['Gradient Boost'],
                       'score' : Gmodel.score(x_test, y_test),
                       'recall' : metrics.recall_score(y_test,GmodelPredict),
                       'Precision' : metrics.precision_score(y_test,GmodelPredict),
                       'F1 Score' : metrics.f1_score(y_test,GmodelPredict),
                       'Roc Auc Score': metrics.roc_auc_score(y_test,GmodelPredict)})
results = pd.concat([results,buffer])

print(results)