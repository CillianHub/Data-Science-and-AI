# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 17:53:52 2021

@author: Cillian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# %matplotlib inline

ad_data = pd.read_csv('advertising.csv') #examine data
ad_data.info()
ad_data.describe()

#create a histogram of the age
sns.histplot(x = 'Age', data = ad_data, bins = 30)

#create a jointplot of Are income vs age
sns.jointplot(y = 'Area Income', x = 'Age', data = ad_data)

#create jointplot of kde distributions of time spent on site vs age
sns.jointplot(x = 'Age', y = 'Daily Time Spent on Site', data = ad_data, kind = 'kde', color = 'red')

#jointplot of daily time on site vs internet usage
sns.jointplot(x = 'Daily Time Spent on Site', y = 'Daily Internet Usage', data = ad_data)

#create pairplot to see more info with hue = clicked on ad
sns.pairplot(data = ad_data, hue = 'Clicked on Ad' )

#now split data ito training and testing data
ad_data.columns #note we wont be using city, country, ad topic
#we only want numerical values
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)



#now we want to train the model and get predictions as before
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report #import classification metrics 
#easy way of interpreting data
from sklearn.metrics import confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test,predictions))























