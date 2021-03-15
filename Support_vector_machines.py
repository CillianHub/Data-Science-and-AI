# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 17:50:37 2021

@author: Cillian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
%matplotlib inline

cancer = load_breast_cancer()   #create cancer variable (is dictionary)
cancer.keys()  #invesrigate the keys in the dictionary

print(cancer['DESCR']) #we can grab info from the dictionary like this
cancer['feature_names'] #gives us the features, ie the columns

df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names']) #create dataframes for the features
df_target = pd.DataFrame(cancer['target'],columns=['Cancer']) #create df for the target and name the column of data 'cancer'


from sklearn.model_selection import train_test_split #now we split the data

X = df_feat                 #we can use the featires df as the features
#y = df_target['Cancer']     #we can use the target df as the target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state = 101)

from sklearn.svm import SVC #import Support vector machines
model = SVC()               #create the instance of SVC
model.fit(X_train,y_train)  #train the model

predictions = model.predict(X_test)

from sklearn.metrics import classification_report #import classification metrics
from sklearn.metrics import confusion_matrix
print(classification_report(y_test, predictions)) #easy way of interpreting data
print(confusion_matrix(y_test,predictions)) #or confusion matrix if you want

#these are good results, but can we do better?
#we use gridsearch to find out the parameter values that we need to set to get the best scores
from sklearn.model_selection import GridSearchCV
#create a dictionary where the keys are the parameters you want to test and the values are the values you want to test
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001]} 

grid = GridSearchCV(SVC(), param_grid,verbose = 3) #verbose just gives you info as its running
grid.fit(X_train,y_train)
grid.best_params_           #using grid to get the best parameters

#now we re-run with best cases to get a new, better model
grid_predictions = grid.predict(X_test)     #get new predictions

print(classification_report(y_test, grid_predictions))  #now compare to previous cases
print(confusion_matrix(y_test,grid_predictions))        #note better on average but worse in some cases ie recall for case 0






















