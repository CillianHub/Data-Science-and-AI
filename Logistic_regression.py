# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 19:00:47 2021

@author: Cillian
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
train = pd.read_csv('titanic_train.csv') #sibsp = num of siblings/spouses on board
                                         #Parch = number of parents/children passenger has on board
                                         
#seaborn heatmap to tell where data is missing
sns.heatmap(train.isnull(), yticklabels= False, cbar = False, cmap = 'viridis')
#can see  that a bit of age data is missing but a lot of cabin knowledge is missing

sns.set_style('whitegrid')  #check out some more data pieces, ie how many survuved, sex of these 
sns.countplot(x = 'Survived', data = train, hue = 'Sex', palette = 'RdBu_r')
#can see females have higher survival rate

#check out whether other things had an impact such as class
sns.countplot(x = 'Survived', data = train, hue = 'Pclass')

sns.distplot(train['Age'].dropna(), kde = False, bins = 20) #drop NaN values with dropna()

sns.countplot(x = 'SibSp', data = train) #most dont have sibling or spouses
train['Fare'].hist(bins = 40, figsize =(10,4)) #see fares paid

import cufflinks as cf
cf.go_offline()
import plotly.io as pio #all this if you wanna do interactive graphs
pio.renderers.default='browser'
train['Fare'].iplot(kind = 'hist',bins = 40)

#now clean the data, want to fill in age data, edit or remove the cabin knowledge
sns.boxplot(x = 'Pclass', y = 'Age', data = train) #can see 3rd class passengers are younger

def impute_age(cols):
    Age = cols[0]  #we will put Age as the first col given to the function
    Pclass = cols[1]        #we will put pclass in here when caling function
    if pd.isnull(Age):
        if Pclass == 1:
            return 37   #check for class and fill in mean age from that class
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
    
train['Age'] = train[['Age','Pclass']].apply(impute_age, axis = 1)
sns.heatmap(train.isnull(), yticklabels= False, cbar = False, cmap = 'viridis')
#now we have no missing info for age

#now drop the cabin column as there is too much missing data
train.drop('Cabin', axis = 1, inplace = True) #want to drop it permentantly

train.dropna(inplace = True) #use this to drop the embarked data point as theres only one
sns.heatmap(train.isnull(), yticklabels= False, cbar = False, cmap = 'viridis')

sex = pd.get_dummies(train['Sex'], drop_first=True) #but we only need one of these columns
embark = pd.get_dummies(train['Embarked'],drop_first = True)

train = pd.concat([train,sex,embark] ,axis = 1) #attaches new cols male, q, s
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace = True)
train.head()
train.drop(['PassengerId'], axis = 1, inplace = True)  #removing all un-needed variables

#exercise:Pclass is a catgeory style of data, 1,2 or 3. Should use dummy args
#run with and then without changing the dummy variables to see the difference
#p_class = pd.get_dummies(train['Pclass'],drop_first = True)
#train = pd.concat([train,p_class] ,axis = 1)
#train.drop(['Pclass'], axis = 1, inplace = True)

#data now clean, going to use 'train' as full data set as it is already clean
#would usually clean test data and just use that

x = train.drop('Survived', axis = 1) #get the features (everything except survived)
y = train['Survived'] #survived is the target

#split up the data
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'], test_size=0.3,random_state=101)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)       #training the model on the data

predictions = logmodel.predict(X_test) #makes predictions based oon new test data

from sklearn.metrics import classification_report #import classification metrics
print(classification_report(y_test, predictions)) #easy way of interpreting data

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions)) #or confusion matrix if you want







