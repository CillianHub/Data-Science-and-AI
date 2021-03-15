# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 15:08:52 2021

@author: Cillian
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv('kyphosis.csv')  #import relevant data set/use var explorer to investigate
#kyphosis column is whether or not the patient had kyphosis after the operation, all subjects in df have had the operation
#age = age in months(children), number = number of vertebrae operated on, start = first vertebrae operated on

sns.pairplot(data = df, hue = 'Kyphosis', palette = 'Set1') #explore the data with pairplot

from sklearn.model_selection import train_test_split #import data splitting
X = df.drop('Kyphosis', axis = 1)   #Creating features, we dont want kyphosis column as this is the target
y = df['Kyphosis']      #creating target columns, target = kyphosis

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30) #splitting data into training and testing data

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()        #import the decision tree and create an instance of it
dtree.fit(X_train, y_train)             #fitting the data to the model

predictions = dtree.predict(X_test) #create predictions based off test features

from sklearn.metrics import classification_report #import classification metrics
from sklearn.metrics import confusion_matrix

print(classification_report(y_test, predictions)) #easy way of interpreting data
print(confusion_matrix(y_test,predictions)) #or confusion matrix if you want

#Now we want to compare this to the random forest method    
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 200) #200 is probably overkill for a dataset this size
rfc.fit(X_train, y_train)                   #train the model 
rfc_predictions = rfc.predict(X_test)       #get predictions based of testing features

print(classification_report(y_test, rfc_predictions)) # we can see that it performed similarly
print(confusion_matrix(y_test,rfc_predictions))       #however for larger data sets, random forests are almost always better

df['Kyphosis'].value_counts()  #we can see that the data is quite unbalanced, this also has an effect

#see lectures if you want to view these decision trees (requires pydot/graphviz), im not going to bother here




















