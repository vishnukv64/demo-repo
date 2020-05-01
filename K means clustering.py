# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:01:23 2018

@author: welcome
"""
import pandas as pd
import numpy as np
import pylab as pl
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df = pd.read_csv("C:/Users/welcome/Downloads/kmeans dataset.csv")

df.head()

df.shape

X = df['Distance_Feature'].values
Y = df['Speeding_Feature'].values

X = df[['Distance_Feature']]
Y = df[['Speeding_Feature']]

X.head()
Y.head()

NC = range(1,20)

kmeans = [KMeans(n_clusters=i) for i in NC]

kmeans

score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]

score 

pl.plot(NC,score)

pl.xlabel('Number of Clusters')

pl.ylabel('Score')

pl.title('Elbow Curve')

pl.show()

pca = PCA(n_components=1).fit(Y)

pca_d = pca.transform(Y)

pca_c = pca.transform(X)

pca_d
pca_c

kmeans = KMeans(n_clusters=3)

kmeansoutput= kmeans.fit(X) 

kmeansoutput

pl.figure('3 Cluster K-Means')

pl.scatter(pca_c[:, 0], pca_d[:, 0], c=kmeansoutput.labels_)

pl.xlabel('Distance_Feature')

pl.ylabel('Speeding_Feature')

pl.title('3 Cluster K-Means')

pl.show()

kmeans.labels_


Spliting the data using cluster 

cluster_map = pd.DataFrame(df)

cluster_map['data_index'] = df.index.values

cluster_map.head()

cluster_map['cluster'] = kmeansoutput.labels_

cluster_map.to_csv('kmeans update.csv')

df.shape

cluster_map[cluster_map.cluster == 0].shape

cluster_map[cluster_map.cluster == 1].shape

cluster_map[cluster_map.cluster == 2].shape

cluster_map.head()

------------------------------------------------ 

url = "https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv"

data = pd.read_csv(url)

km = KMeans(n_clusters=5).fit(data)

cluster_map = pd.DataFrame()
cluster_map['data_index'] = data.index.values
cluster_map['cluster'] = km.labels_

