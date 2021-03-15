# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:44:41 2021

@author: Cillian
"""

#code tries to figure out if a school is private or not
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv('College_Data',index_col=0)
df.info()
df.describe()
sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',palette='coolwarm',fit_reg=False)
sns.set_style('whitegrid')
sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',palette='coolwarm',fit_reg=False)

sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm')
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)

g = sns.FacetGrid(df,hue="Private",palette='coolwarm',aspect=2) #can change the aspect ratio
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

#there is a school with grad rate > 100, whats the name of this school?
df[df['Grad.Rate']>100]     #we can see it is called casenovia college

#now we want to change this rate to be 100 so it makes sense
df['Grad.Rate']['Cazenovia College'] = 100  #throws warning that you have overwritten data
df[df['Grad.Rate']>100] #since this doesnt return any values, its all good

g = sns.FacetGrid(df,hue="Private",palette='coolwarm',aspect=2) 
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7) #we can now see grad rate stops at 100

kmeans = KMeans(n_clusters=2)   #since we know a college is either private or not (2 options)
                                #we use 2 clusters
                                
                                
#we do notdo train test split on unsupervised learning algorithms
#and we want to train it on everythin expect the private vector
kmeans.fit(df.drop('Private',axis=1))                                

#now get the cluster centre vectors
kmeans.cluster_centers_

#there is no way to evaluate clustering if yu dont have labels. however since
#this is just an exercise, we have them


#create a function to create a new element of the data frame (1 for private school, 0 for public
def converter(cluster):
    if cluster=='Yes':
        return 0
    else:
        return 1
    
df['Cluster'] = df['Private'].apply(converter) #apply it to the private column


#now we check the results
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))






                                
