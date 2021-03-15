# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:46:11 2021

@author: Cillian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('TensorFlow_FILES/DATA/cancer_classification.csv')
df.info()                           #we can see theres no null values
df.describe().transpose()           #look at the count/mean of each column

#do some Exploratory Data Analysis - For classification, usually good to do a ocunt of the label to see how 
sns.countplot(x='benign_0__mal_1',data=df)         #well balanced the data is, in this case, well balanced

#also usually good to get an idea of inital correlations
sns.heatmap(df.corr())
df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar') #we dont want the last value, as it is the label itself

#now we will do train test split and scale after
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X = df.drop('benign_0__mal_1', axis = 1).values         #getting the values so its an array not a df
y = df['benign_0__mal_1'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#now create the model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
X_train.shape  #we can see we have 426 rows and 30 features in the training data

#create model
model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=15,activation='relu'))

model.add(Dense(units=1,activation='sigmoid'))  #last output can be one neuron with sigmoid function
                                                #as this is a binary classification problem
model.compile(loss='binary_crossentropy', optimizer='adam') 
#fit the model to the training data and allow for checks with validation data
model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test, y_test), verbose=1)

model_loss = pd.DataFrame(model.history.history) #now check the losses
model_loss.plot()           #we can see that we are overfitting

#now we will use Early Stopping to stop overfitting, first, retrain a model
model1 = Sequential()
model1.add(Dense(units=30,activation='relu'))
model1.add(Dense(units=15,activation='relu'))

model1.add(Dense(units=1,activation='sigmoid'))  #last output can be one neuron with sigmoid function
                                                #as this is a binary classification problem
model1.compile(loss='binary_crossentropy', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
#the patience var is how many epochs after no improvement seen will the model stop
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25) #and we want to min the losses so we choose to minimise it
#note: if we were monitoring accuracy we would probably want this to be 'max'

model1.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test, y_test),callbacks =[early_stop],verbose=1)
#note it stops early, so saves runtime and tells you you have too many epochs

model_loss1 = pd.DataFrame(model1.history.history) #now check the losses
model_loss1.plot() 


from tensorflow.keras.layers import Dropout
#you can also use dropout layers to prevent overfitting, create new model not to continue training old model
model2 = Sequential()
model2.add(Dense(units=30,activation='relu'))
model2.add(Dropout(0.5))  #this is the probability of turning off a particular neuron
model2.add(Dense(units=15,activation='relu'))
model2.add(Dropout(0.5))

model2.add(Dense(units=1,activation='sigmoid'))  #last output can be one neuron with sigmoid function
                                                #as this is a binary classification problem
model2.compile(loss='binary_crossentropy', optimizer='adam')

#note it goes on for a bit longer because it is still learning
model2.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test, y_test),callbacks =[early_stop],verbose=1)
model_loss2 = pd.DataFrame(model2.history.history) #now check the losses
model_loss2.plot() 
#note the loss is going down, much better performance


#now we want to do predictions and evaluate them
predictions = model2.predict_classes(X_test)  #get predictions from test data

#now check performance
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))





