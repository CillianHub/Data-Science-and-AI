# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 18:21:29 2021

@author: Cillian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("TensorFlow_FILES/DATA/fake_reg.csv") #import data and look at it

sns.pairplot(df)    #Some exploratory Data Analysis

from sklearn.model_selection import train_test_split
X = df[['feature1','feature2']].values  #.values just gives you th evalues as a numpy array
y = df['price'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42) #train test split

from sklearn.preprocessing import MinMaxScaler #now we wantt to scale/normalise the data
help(MinMaxScaler) #get info
scaler = MinMaxScaler()     #create an instance

scaler.fit(X_train) #calculates std, min and max, only run on training set to avoid data leakage
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#now to create neural network
import tensorflow as tf
from tensorflow.keras.models import Sequential #start with simple sequential model and add layers
from tensorflow.keras.layers import Dense, Activation #Dense = simple layer to be added
#dense = densely connected neurons, ie, every neuron connected toevery neuron in the next layer
#now to construct the model, 2 ways:                      

#create empty list and add layers or if you we already know the layers, use method 2 in #
model = Sequential()                                    #model = Sequential([
model.add(Dense(4,activation='relu'))                   #Dense(units=2),
model.add(Dense(4,activation='relu'))                   #Dense(units=2),
model.add(Dense(4,activation='relu'))                   #Dense(units=2),

# Final output node for prediction
model.add(Dense(1))

model.compile(optimizer='rmsprop',loss='mse')   #mse = mean squared error, another optimiser would be adam
#now we train the model, X = features, y = labels, epochs = number of times it will go through the dataset
model.fit(X_train,y_train,epochs=250)
model.history.history #will give loss per epoch

#loss = model.history.history['loss']  #create a list from the history of the loss values
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()  #get a plot for the loss


#Evaluation: In this case, the chosen type of error, as we defined earlier is MSE
#method 1: model.evaluate   #returns the models loss on the test set (in the metric you chose)
training_score = model.evaluate(X_train,y_train,verbose=0)
test_score = model.evaluate(X_test,y_test,verbose=0)

#now get actual predictions
test_predictions = model.predict(X_test)
test_predictions = pd.Series(test_predictions.reshape(300,)) #series instead of np  so we can concatenate

pred_df = pd.DataFrame(y_test,columns=['Test Y'])           #create DF with actual Y values
pred_df = pd.concat([pred_df,test_predictions],axis=1)      #add predicted y values
pred_df.columns = ['Test Y','Model Predictions']            #label these columns

#now do some data analysis, look at relationship between predicted and true values
sns.scatterplot(x='Test Y',y='Model Predictions',data=pred_df)

#create error column
pred_df['Error'] = pred_df['Test Y'] - pred_df['Model Predictions']

#now we check the error
from sklearn.metrics import mean_absolute_error,mean_squared_error
mean_absolute_error(pred_df['Test Y'],pred_df['Model Predictions']) #on average about 4 dollars off
df.describe()   #since we have an average price of about 500 dollars, 4 dollars off is good!

mse = mean_squared_error(pred_df['Test Y'],pred_df['Model Predictions'])
rms = np.sqrt(mse)  #get mse and rmse

#how to test new cases
#take a new gemstone with the following features [[Feature1, Feature2]]
new_gem = [[998,1000]]

#first, scale the features, as you did with the training data
new_gem = scaler.transform(new_gem)    

#now predict as you always would
model.predict(new_gem)


#if you want to save /load a model
from tensorflow.keras.models import load_model
model.save('my_model.h5') #creates a HDF5 file


#to load it again later
later_model = load_model('my_model.h5')
#now we can use it as we would the previous model
later_model.predict(new_gem)









