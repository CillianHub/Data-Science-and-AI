# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 19:29:34 2021

@author: Cillian
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('Classified Data')

scaler = StandardScaler()                       #we want all variable in the same scale 

features = df.drop('TARGET CLASS', axis = 1) #want everything except the Target class for the features

df_feat = pd.DataFrame(features, columns = df.columns[:-1]) #create feature dataframe, not yet scaled

X = df_feat
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)

from sklearn.neighbors import KNeighborsClassifier #KNN classifier imported
knn = KNeighborsClassifier(n_neighbors = 1) #k value = 1 for KNN

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)  #scale the data and fit
knn.fit(X_train_scaled, y_train)        #we scale after the data has been split so we dont impact the test data with training data

X_test_scaled = scaler.transform(X_test)#now we scale the test data so that it is in the same form
predictions = knn.predict(X_test_scaled)

from sklearn.metrics import classification_report #import classification metrics
from sklearn.metrics import confusion_matrix
print(classification_report(y_test, predictions)) #easy way of interpreting data
print(confusion_matrix(y_test,predictions)) #or confusion matrix if you want


#now use the elbow method to choose better/best k value
error_rate = []     #empy list to hold values for each k
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train_scaled, y_train)                    #scrolling through k values getting error rates for each
    prediction_i = knn.predict(X_test_scaled) 
    error_rate.append(np.mean(prediction_i!=y_test))  #average of where predictions were not correct ie ACCURACY value

#now have a look at all the k values
plt.figure(figsize = (10,6))
plt.plot(range(1,40),error_rate, color = 'blue', ls = 'dashed',marker = 'o', markerfacecolor = 'red', markersize = 10)
plt.title('Error rate vs K value')
plt.xlabel('K')
plt.ylabel('Error Rate')        #from this we can see k being bigger gives better error rates

#there is a tradeoff between compute time/data and larger k so we just pick a high 
#enough k that gives good error rate, in this case, 16, note that error is already very low


#now we check classification of this new k value

knn = KNeighborsClassifier(n_neighbors = 4) #k value = 1 for KNN

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)  #scale the data and fit
knn.fit(X_train_scaled, y_train)        #we scale after the data has been split so we dont impact the test data with training data
X_test_scaled = scaler.transform(X_test)#now we scale the test data so that it is in the same form

predictions = knn.predict(X_test_scaled)

print(classification_report(y_test, predictions)) #easy way of interpreting data
print(confusion_matrix(y_test,predictions)) #or confusion matrix if you want







