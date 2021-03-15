# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 18:00:14 2021

@author: Cillian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
yelp = pd.read_csv('yelp.csv')  #import an investigate data
yelp.info()
yelp.describe()


yelp['text length'] = yelp['text'].apply(len) #like in the lecture, get the length of every row in
#the text column and put it into a new column

#Exploratory data analysis
sns.set_style('white')
#use facet grid to plot histograms of the text length for each star rating
g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length', bins = 50)

sns.countplot(x='stars',data=yelp,palette='rainbow')  #countplot of the number of occurrences for each star rating

stars = yelp.groupby('stars').mean()    #groupby to get the mean values of the numerical columns

stars.corr() #see how correlated each column is
sns.heatmap(stars.corr(),cmap='coolwarm',annot=True) #seaborn to create heatmap

#NLP classification, for ease , we will only use one or five star reviews
yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)] #only grab items with one or five stars


X = yelp_class['text']      #features
y = yelp_class['stars']     #labels 

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
#se the fit_transform method on the CountVectorizer object and pass in X (the 'text' column)
X = cv.fit_transform(X)

#do train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

#now we train the model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)     #fit to training data

predictions = nb.predict(X_test)    #get predictions
#check classification report and confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))    #this is all based just off the text data

#now we will try with TF-IDF transformer
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.pipeline import Pipeline
#vreate a pipeline to do the processes
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

#now we redo the train test split, training and predicitons
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

pipeline.fit(X_train,y_train)
new_predictions = pipeline.predict(X_test)
print(confusion_matrix(y_test,new_predictions))
print(classification_report(y_test,new_predictions)) 
