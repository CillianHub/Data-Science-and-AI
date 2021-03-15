# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:55:47 2021

@author: Cillian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs   #create random data groups

#creating random data
data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8,random_state=101)

#plot all the rows in the first column of the first element of the tuple vs all the rows in the seond column
plt.scatter(data[0][:,0],data[0][:,1])

#data[1] is the cluster that the corresponding point in data[0] belongs to
plt.scatter(data[0][:,0],data[0][:,1], c = data[1], cmap = 'rainbow')

from sklearn.cluster import KMeans #import the kmeans clustering method
 #for kmeans, we must know the number of clusters to run
kmeans = KMeans(n_clusters = 4)     #as we created the data we know that n_clusters = n_centers = 4

kmeans.fit(data[0])     #fit kmeans to the features

kmeans.cluster_centers_  #this gives an array of the co-ordinates of the centres of the clusters

kmeans.labels_          #gives a list of the labels it believes to be true for the clusters

#note, with real data, at this point, you would be done as you would have nothing to compare to.
#however, since we have the actual correct labels, we can see how accurate we are

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')