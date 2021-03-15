# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:17:06 2021

@author: Cillian
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()   #import cancer dataset
cancer.keys()       #get features of the dataset

print(cancer['DESCR']) #we can see there is a DESCR so we want to lok at it

#now turn the dataset into a dataframe
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

#we examine the targets from the dataset
cancer['target']
cancer['target_names']

#since we have a large amount of variables (features), we want to figure out which ones are
#important. We do PCA so we dont have to waste resources when applting a Machine learning alg

#we want to find the 2 most important features, ie the 2 Principal components

#first set evrything to standard scale
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)   #now set number of components to keep
pca.fit(scaled_data)        #now fit the scaled data

x_pca = pca.transform(scaled_data) #transform the scaled data
scaled_data.shape           #check the size of the new and old data
x_pca.shape                 #we went from 30 to 2 variables so it worked

#now we want to plot this data with matplotlib 
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma') #all the rows from col 0 
plt.xlabel('First principal component')                             #vs all the rows frm col 1
plt.ylabel('Second Principal Component')

#from this we can see clear separation in the malignant tumors vs the benign from just 2 components
#components do not relate directly to a feature

#examining what these compnents look like with a heatmap
df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma_r',)

#this shows the correlation between each principal compnent with each feature
    













