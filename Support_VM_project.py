# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 18:26:43 2021

@author: Cillian
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
%matplotlib inline

#use seaborn to load the iris dataset
iris = sns.load_dataset('iris')

#create a pairplot of the dataset, which species seems to be the most separable?
sns.pairplot(data = iris, hue = 'species')  #can see that setosa is the most easily defined

#Create a kde plot of sepal_length versus sepal width for setosa species of flower.
setosa = iris[iris['species']=='setosa']#create df of only setosa

#create KDE plot of sepal width vs length for setosas
sns.kdeplot(x ='sepal_width',y ='sepal_length', data = setosa ,cmap="plasma", shade=True, thresh=False)

#now we train test split
from sklearn.model_selection import train_test_split 
X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 101)

#now we create our SV, fit it and get predictions
from sklearn.svm import SVC #import Support vector machines
model = SVC()               #create the instance of SVC
model.fit(X_train,y_train)  #train the model

predictions = model.predict(X_test)

#evaluate our predictions
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions)) #easy way of interpreting data
print(confusion_matrix(y_test,predictions)) #or confusion matrix if you want

#we can see the reults are already good but can we improve them with gridsearch?
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 

grid = GridSearchCV(SVC(), param_grid,verbose = 2) #verbose just gives you info as its running
grid.fit(X_train,y_train)  #retrain/search for best parameters
grid.best_params_           #using grid to get the best parameters

grid_predictions = grid.predict(X_test)     #get new predictions
print(classification_report(y_test, grid_predictions)) #easy way of interpreting data
print(confusion_matrix(y_test,grid_predictions)) #or confusion matrix if you want

#in this case gridsearch did not help












