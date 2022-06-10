# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 23:21:07 2022

@author: dhirp

"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv("Mall_Customers_BP.csv")
df.head()

mms = MinMaxScaler()
df[['Annual Income (k$)', 'Spending Score (1-100)']] = mms.fit_transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])
print(df)


distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 15)

# Making a seperate array file having data for Annual income and spending score

array1 = df['Annual Income (k$)'].to_numpy()
array2 = df['Spending Score (1-100)'].to_numpy()
array = np.array(list(zip(array1, array2))).reshape(len(array1), 2)

for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(array)
    kmeanModel.fit(array)

    distortions.append(sum(np.min(cdist(array, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / array.shape[0])
    inertias.append(kmeanModel.inertia_)

    mapping1[k] = sum(np.min(cdist(array, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / array.shape[0]
    mapping2[k] = kmeanModel.inertia_
    
for key,val in mapping2.items():
    print(str(key)+' : '+str(val))

# Finding Centroids -
data = pd.DataFrame(array, columns=('Annual Income (k$)', 'Spending Score (1-100)'))
data.head()

kmeans = KMeans(n_clusters=5).fit(data)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c= kmeans.labels_.astype(float))
plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
plt.show()