# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 15:51:05 2021

@author: Cillian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
loans = pd.read_csv('loan_data.csv')        #read in data for loans
loans.info()                #check out info and look at variables to get more info about the loans datafram

#now we do some exploratory data analysis
#create a histogram of two FICO dists, one for each credit policy outcome (1 or 0)
#easy way to do it seen below

sns.set(rc={'figure.figsize':(10,6)})       #can use this to set the size of a fig with sns
sns.histplot(data = loans, x = 'fico', hue = 'credit.policy', bins= 35)

#hard way/long way, done in lectures
plt.figure(figsize = (10,6))
loans[loans['credit.policy']==1]['fico'].hist(bins = 35, color = 'blue', label = 'Credit Policy = 1', alpha = 0.6)
loans[loans['credit.policy']==0]['fico'].hist(bins = 35, color = 'red', label = 'Credit Policy = 0', alpha = 0.6)
plt.legend()

#either way, we can see more people with credit policy = 1
#we can also see that in general, lower fico score = more likely to be credit policy 0
#can see form the cutoff, anyone with fico score under 660/670 wont have the appropriate credit score for lendingclub.com

#now do the same for not fully paid column
sns.histplot(data = loans, x = 'fico', hue = 'not.fully.paid', bins= 35)

#Create a countplot using seaborn showing the counts of loans by purpose,hue=not.fully.paid
plt.figure(figsize = (10,6))    #can also just use figize to set size of sns graphs
sns.countplot(data = loans, x = 'purpose', hue = 'not.fully.paid')

#see the trend between FICO score and interest rate
sns.jointplot(x = 'fico', y ='int.rate', data = loans, color = 'purple')

# lmplots to see if the trend differed between not.fully.paid and credit.policy, col value gives multiple graphs
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',col='not.fully.paid', palette = 'Set1')

#now we need to set up the data
#we can see that there is categorical data in the 'purpose' column, use get dummies to convert this into a variable
final_data = pd.get_dummies(loans,columns=['purpose'],drop_first=True)  #create final dataframe

#now we train test split
from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid', axis = 1)       #split up data into train and test
y = final_data['not.fully.paid']            #X = features, y = target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state = 101)

#now train a single decision tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()        #import the decision tree and create an instance of it
dtree.fit(X_train, y_train)             #fitting the data to the model
predictions = dtree.predict(X_test)     #create predictions


from sklearn.metrics import classification_report #import classification metrics
from sklearn.metrics import confusion_matrix
print(classification_report(y_test, predictions)) #easy way of interpreting data
print(confusion_matrix(y_test,predictions)) #or confusion matrix if you want

#now we want to train a random forest as oppposed to a single tree
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 300)
rfc.fit(X_train, y_train)                   #train the model 
rfc_predictions = rfc.predict(X_test)       #get predictions based of testing features

#now test again and compare to decision tree method
print(classification_report(y_test, rfc_predictions)) #easy way of interpreting data
print(confusion_matrix(y_test,rfc_predictions)) #or confusion matrix if you want

#note single decision tree did better for class 1 in recall, same for f1 score
#what is better depends on what metric you are looking at




















