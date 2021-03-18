# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 19:17:50 2021
K Nearest Neighbour Example 
@author: Cillian
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('KNN_Project_Data') #read in csv file
sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm') #plot the data to get some info

scaler = StandardScaler()    
scaler.fit(df.drop('TARGET CLASS', axis = 1))       
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis = 1))       #scaling the features
#as before, data from the test class impacts the scaling for this set but we will just scale everything for simplicity

df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1]) #create dataframe of scaled features

#note we dont HAVE to creat X and y separately, we can just pass in the args, but it makes it easier 
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],test_size=0.30, random_state = 101)

from sklearn.neighbors import KNeighborsClassifier #KNN classifier imported
knn = KNeighborsClassifier(n_neighbors = 1) #k value = 1 for KNN
knn.fit(X_train, y_train)  #training the set, with k = 1 originally 

predictions = knn.predict(X_test)

from sklearn.metrics import classification_report #import classification metrics
from sklearn.metrics import confusion_matrix
print(classification_report(y_test, predictions)) #easy way of interpreting data
print(confusion_matrix(y_test,predictions)) #or confusion matrix if you want

#below is the 'elbow' method for getting the best k value
error_rate = []     #empy list to hold values for each k
for i in range(1,60):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)                    #scrolling through k values getting error rates for each
    prediction_i = knn.predict(X_test) 
    error_rate.append(np.mean(prediction_i!=y_test)) 

#plotting the k value vs error rate, raise dthe k to 60 because at 40 we saw it was still dropping
plt.figure(figsize = (10,6))        
plt.plot(range(1,60),error_rate, color = 'blue', ls = 'dashed',marker = 'o', markerfacecolor = 'red', markersize = 10)
plt.title('Error rate vs K value')
plt.xlabel('K')
plt.ylabel('Error Rate')  


#now we retrain the model with k = 30 as we saw from the graph
knn = KNeighborsClassifier(n_neighbors = 30) #k value = 1 for KNN
knn.fit(X_train, y_train)  #training the set, with k = 1 originally 

predictions = knn.predict(X_test)
print(classification_report(y_test, predictions)) #re-do classification and confusion matrix
print(confusion_matrix(y_test,predictions))     #can see the values are much better




























