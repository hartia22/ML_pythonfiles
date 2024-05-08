# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:48:50 2020

@author: Nautilus
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random 
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
from sklearn.metrics import r2_score

pd.set_option('display.max_columns',None)

maindf = pd.read_csv('concrete.csv')

"""
Univariate analysis – data types and description of the independent attributes
which should include (name, range of values observed, central values (mean and
median), standard deviation and quartiles, analysis of the body of distributions /
tails, missing values, outliers, duplicates
"""

maindf.info();
#we appear to have no missing values 

print(maindf.describe())

print(maindf.skew())
maindf = maindf.drop_duplicates()

maindf.info();

c = maindf.columns 
for i in c:
    sns.boxplot(maindf[i],hue=maindf['strength'])
    plt.show()

#thanks to these box plots we can see the outliers hurting out Q scores on the 
# age, fineagg, superplastic, water and slag

#slag: we can see that slag has a huge skew towards the lower values, however there 
#are values that are much higher than q3. I will not make any changes as of this point as 
#the correlation graph shows it might have significant impact  

print(maindf[maindf['slag']>300]['slag'].value_counts())
maindf = maindf[maindf['slag']<340]
sns.boxplot(maindf['slag'])

ss = StandardScaler()
mnmx = MinMaxScaler()
'''
slagdata = pd.DataFrame(maindf['slag'])

ss.fit(slagdata)
slagdata = ss.fit_transform(slagdata)
maindf['slag'] = slagdata
'''
sns.boxplot(maindf['water'])
print(maindf[maindf['water']>230]['water'].value_counts())
maindf = maindf[maindf['water']<230]

maindf['water'] = ss.fit_transform(maindf[['water']])


sns.boxplot(maindf['superplastic'])
print(maindf[maindf['superplastic']>25]['superplastic'].value_counts())
maindf = maindf[maindf['superplastic']<30]
maindf['superplastic'] = ss.fit_transform(maindf[['superplastic']])
sns.boxplot(maindf['fineagg'])
print(maindf[maindf['fineagg']>990]['fineagg'].value_counts())
print(maindf[maindf['fineagg']<600]['fineagg'].value_counts())
maindf = maindf[maindf['fineagg']<990]
maindf['fineagg'] = ss.fit_transform(maindf[['fineagg']])


sns.boxplot(maindf['age'])
print(maindf['age'].value_counts())
print(maindf[maindf['age']>300])
maindf['age'] = np.log(maindf['age'])

print(maindf['strength'].value_counts())
sns.distplot(maindf['strength'])
# ill be creating a category for judging strength str<20 = 0(weak), str>40 = 2 (strong), anything else is 1 (average)

maindf['cat_str'] = maindf['strength']
maindf.loc[maindf['cat_str']<20,['cat_str']] = 0
maindf.loc[maindf['cat_str']>40,['cat_str']] = 2
maindf.loc[maindf['cat_str']>20,['cat_str']] = 1
maindf['cat_str'] = maindf['cat_str'].astype('int')
'''
Bi-variate analysis between the predictor variables and between the predictor
variables and target column. Comment on your findings in terms of their
relationship and degree of relation if any. Visualize the analysis using boxplots and
pair plots, histograms or density curves
'''

#run correlations between existing features 

maindf_corr = maindf.corr()

sns.heatmap(maindf_corr,annot=True)
#there seems to be a small positive correlation between the amount of cement while things like water 
# and fine aggregate seems to have a small negative correlation 
#all others seem to have almost no correlation to the target
maindf.columns
maindf = maindf.drop(columns=['ash'])


#######sns.pairplot(maindf, hue='cat_str')
#from this analysis we see that the amount of cement does have an impact on the strenght
# we can observe our "weak" category has a high frequency of low cement content
#aditionally more should be done to the age feature as we can group days into weeks 
'''
Feature Engineering techniques 
'''

maindf['weeks']=maindf['age']/7
maindf['weeks']=maindf['weeks'].astype('int')
newdf = maindf

newdf = newdf.drop(columns=['cement',  'water', 'superplastic',  'fineagg', 'age', 'strength'])
sns.pairplot(newdf,hue='cat_str')

#we can see that there is also see with this plot that low age is comon among our strong category 
#the other features might be irrlevant but i will put them on my first model for comparisons
y = maindf['strength']
#categorical strenght target variable
z = maindf['cat_str']
x = maindf.drop(columns=['strength','cat_str'])
x_train, x_test, z_train, z_test=train_test_split(x,z,test_size = .30,random_state = 7)
lre = LogisticRegression(C=1.0, class_weight=None,
                                                dual=False, fit_intercept=True,
                                                intercept_scaling=1,
                                                l1_ratio=None, max_iter=10000,
                                                multi_class='auto', n_jobs=None,
                                                penalty='l2', random_state=None,
                                                solver='lbfgs', tol=0.0001,
                                                verbose=0, warm_start=False)
lre.fit(x_train,z_train)

lre_predict = lre.predict(x_test)


'''
Creating the model and tuning it
'''
#cross validation 
folding = KFold(n_splits=5,random_state=23, shuffle=True)
lre_fold_results=cross_val_score(lre,x,z,cv=folding,scoring='r2')

results = pd.DataFrame({'model' : ['Log Regression'],
                        'r2 score' : lre_fold_results.mean()*100.0})

adae = AdaBoostClassifier()
adae_fold_results = cross_val_score(adae,x,z,cv=folding,scoring='r2')

buffer = pd.DataFrame({'model' : ['ADABoost'],'r2 score' : adae_fold_results.mean()*100.0})
results = pd.concat([results,buffer])

rtce = RandomForestClassifier()
rtce_fold_results = cross_val_score(rtce,x,z,cv=folding, scoring='r2')


buffer = pd.DataFrame({'model' : ['Random forest'],'r2 score' : rtce_fold_results.mean()*100.0})
results = pd.concat([results,buffer])

bce = BaggingClassifier(base_estimator=rtce,n_estimators=10)
bce_fold_results = cross_val_score(bce,x,z,cv=folding,scoring='r2')

buffer = pd.DataFrame({'model' : ['Bag of Trees'],'r2 score' : bce_fold_results.mean()*100.0})
results = pd.concat([results,buffer])

parameters = {"solver": ['newton-cg','lbfgs','liblinear','sag']}
rscv = RandomizedSearchCV(lre,parameters,10)
rscv.fit(x,z)


buffer = pd.DataFrame({'model' : ['LogReg(RandomSearchCV)'],'r2 score' : rscv.cv_results_['mean_test_score'].mean()*100.0})
results = pd.concat([results,buffer])

param_ = {"max_depth": [4, None],
              "max_features": sp_randint(1, 5),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

rscv2 = RandomizedSearchCV(rtce,param_,10)
rscv2.fit(x,z)

buffer = pd.DataFrame({'model' : ['Random forest(RandomSearchCV)'],'r2 score' : rscv2.cv_results_['mean_test_score'].mean()*100.0})
results = pd.concat([results,buffer])

linearreg = LinearRegression()
parameters_ = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
rscv3 = RandomizedSearchCV(linearreg,parameters_,10)
rscv3.fit(x,y)
buffer = pd.DataFrame({'model' : ['linear reg(RandomSearchCV)'],'r2 score' : rscv3.cv_results_['mean_test_score'].mean()*100.0})
results = pd.concat([results,buffer])

#After runing the randomized search
print(results)
