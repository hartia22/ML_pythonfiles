# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 06:08:40 2020

@author: Majic
"""

import pandas as pd
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import zscore
from sklearn.impute import SimpleImputer


pd.set_option('display.max_columns', None)

df = pd.read_excel('e:\\Credit Card Customer Data.xlsx')

df.head()

df.info()

df.describe()
#spending looks really different from the rest of the variables in regards to magnitude
# i might have to make some changes later
df['Customer Key'].value_counts().count()
#there are soome customer duplicates but upon examination the records are not duplicates 
df['Sl_No'].value_counts().count()
#this field also seems to be just an identifier of which our pandas alrady has an index 
# dropping both of these before EDA will help our visualisation 

df = df.drop(columns=['Customer Key', 'Sl_No']) 

s = df.columns 
for i in s:
    print(i, '- ' ,df[i].isna().sum())

#there are many NaN values in our dastaframe    
#imputer1 = SimpleImputer(missing_values=np.nan, strategy = 'mean')
#creditcolumn = imputer1.fit_transform(df['Avg_Credit_Limit']).T
#df['Avg_Credit_Limit'] = creditcolumn 

df.skew()   

# there is pretty hard skewing on some of the columns, particularly Total online visits and credit card limit
df = df.apply(zscore)


#applying Zscore function to get rid of the units which hurts our distance calculations 
sns.pairplot(df) 
#the pair plot further shows our skew data on Avg_Credit_Limit and Total_visits_online
# I will use a log to remedy 

#le logged pair plot shows more 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
#i will used the original dataframe before testing with the logged data
# i will use the elbow method to determine the optimal number of clusters

c = range(1,10)
means = []

for k in c:
    model=KMeans(n_clusters=k)
    model.fit(df)
    prediction = model.predict(df)
    means.append(sum(np.min(cdist(df,model.cluster_centers_,'euclidean'),axis=1))/df.shape[0])

# i can now visualize the average distances between clusteres of different sizes 

plt.plot(c,means,'bx-')
plt.show()

#3 to 4 seem to be the best amount of clusters 

fmodel = KMeans(3)
fmodel.fit(df)
fprediction = fmodel.predict(df)

df['label'] = fprediction

analysis = df.groupby(['label'])

df.boxplot(by='label', layout = (2,4),figsize=(15,10))

#in most axis the groups have very distict grouping values. 

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage

cleandf = df.drop(columns=['label'])

cleandf.head()
lmethods = ['average','single','weighted','centroid','median', 'ward']
for i in lmethods:
    l1 = linkage(cleandf,metric='euclidean',method=i)
    c, cd = cophenet(l1,pdist(cleandf))
    print(i,' -- ', c)
#i am choosing average based on the cophenet score     
l1 = linkage(cleandf,metric='euclidean',method='average')

dendrogram(l1,truncate_mode='lastp',p=20)
plt.show()

hmodel = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='single')
hmodel.fit(cleandf)
cleandf['label']= hmodel.labels_
'''
#THIS WAS IMPOSSIBLE TO READ
plt.figure(figsize=(10, 5))
plt.title('Agglomerative Hierarchical Clustering Dendogram')
plt.xlabel('sample index')
plt.ylabel('Distance')
dendrogram(l1, leaf_rotation=90.,color_threshold = 40, leaf_font_size=8. )
plt.tight_layout()
plt.show()
'''



from sklearn.metrics import silhouette_score

kscore = silhouette_score(df,df['label'],metric='euclidean')

hscore = silhouette_score(cleandf,cleandf['label'],metric='euclidean')

print('hierarchical clustering silhouette score: ',hscore )
print('K-meansclustering silhouette score: ', kscore)
#we can see that there is a highed coefficient on the K means cluster 

cleandf.boxplot(by='label', layout = (2,4),figsize=(15,10))
df.boxplot(by='label', layout = (2,4),figsize=(15,10))
#judging by the box plots we can observe that label 2 remaind mostly unchanged for both methods 
#which means that those points were easly clustered toghether using both algorithms 
#cluster 2 contains customers with high average credit card limits, multiple cards and high online engagement 
# cluster 0 seems to have high call engagement but not much else. the calls might be a good way to reach these customers 
