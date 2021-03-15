# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 13:09:49 2021

@author: Cillian
"""
%matplotlib inline 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('USA_housing.csv')
df.info()
described = df.describe()


sns.pairplot(df)
sns.distplot(df['Price'])
sns.heatmap(df.corr(),annot = True)

df.columns  #just gives a list of the column names
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]  #take out price (the target) and address as its not useful
#or could just use X = df[df.columns[0:5]]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

lm = LinearRegression()  #create a linear regression object
lm.fit(X_train, y_train) #fit this to the training data

print(lm.intercept_)
print(lm.coef_)  #corresponds to each column in X or X_train

cdf=pd.DataFrame(lm.coef_, X.columns, columns = ['Coeffs']) #creates table for coefficient per column

predictions = lm.predict(X_test) #we want to compare this to y_test

plt.scatter(y_test, predictions)
sns.distplot(y_test - predictions)

#now we want to test the error/accuracy of the model
from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rms = np.sqrt(mse)

#check residuals
sns.distplot(y_test-predictions)


#%% this is for the real boston dataset
from sklearn.datasets import load_boston
boston = load_boston()
print(boston['feature_names'])


















