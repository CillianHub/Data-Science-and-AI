# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:35:04 2021

@author: Cillian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#price of houses given features, regression problem
#import real world data, price of houses in Seattle6
df = pd.read_csv('TensorFlow_FILES/DATA/kc_house_data.csv')

df.isnull().sum()       #get a count per column of null values, in this case we have no null values
df.describe().transpose() #get info on the mean/count/std

#Exploratory Data Analysis
plt.figure(figsize=(12,8))
sns.distplot(df['price'])       #we can see there are some outliers on the upper price scale

#now check out some different features
sns.countplot(df['bedrooms'])

df.corr()['price']  #check the correlation values for the price column
df.corr()['price'].sort_values() #now check them in order, we cann see sqft_living has the highest

plt.figure(figsize=(12,8))  #now check out this highly correlated feature with a scatter plot
sns.scatterplot(x='price',y='sqft_living',data=df)

sns.boxplot(x='bedrooms',y='price',data=df) #can check for other features

#checking out other features such as long and lat, look for trends
sns.scatterplot(x='price',y='long',data=df) #we can see that long = -122.2 seems to have higher prices
sns.scatterplot(x='price',y='lat',data=df)#can see that 47.6/47.7 are more pricey#

#now we compare the two, plotting lat and long of each house will show the residential areas of the area
sns.scatterplot(x='long',y='lat',data=df,hue='price') 

#hard to make out the color grade on the map because of outliers, so drop a few outliers
df.sort_values('price',ascending=False).head(20) 
#looking at this and our histogram of houses, so we should cut it off a percentage at the top of the scale
len(df)*(0.01)                  #this gives the top one percent of houses (1% chosen arbitrarily)
one_percent_removed  = df.sort_values('price',ascending=False).iloc[216:]

#now we check again, the scatterplot
sns.scatterplot(x='long',y='lat',data=one_percent_removed,hue='price') 
#to improve this graph, add alpha, remove edge colouring and chnage the palette

sns.scatterplot(x='long',y='lat', data=one_percent_removed,hue='price',palette='RdYlGn',edgecolor=None,alpha=0.2) 

#we can see waterfront properties seem to be more expensive so check that out
sns.boxplot(x='waterfront',y='price',data=df)

#NOW WE CHECK THE FEATURE DATA
df = df.drop('id',axis=1)       #we dont need the id column as it contains no info
df['date'] = pd.to_datetime(df['date']) #convert date to a datetime object

df['month'] = df['date'].apply(lambda date:date.month)  #extract the month and create a new column
df['year'] = df['date'].apply(lambda date:date.year) #do the same for the year

sns.boxplot(x='month',y='price',data=df) #now we can look at the price by month, could also do by year

df.groupby('month').mean()['price'].plot() #now we can plot the average price per month

df = df.drop('date',axis=1)     #since we have the dates now in month and year, we dont need date col

#note, for zipcode, model would assume that these are linear values as opposed to categorical data
#since too many to dummy variable, and no clear correlation, we can drop it

df = df.drop('zipcode',axis=1)

#for yr_renevated, we can leave as is as if it is a higher number, its more likely to be worth more
#same for sqr_ft_basement, no basement= 0 = no basement, but larger basement worth more



#NOW WE SET THE TRAIN SETS SPLIT
X = df.drop('price',axis=1).values      #calling values to get Array as opposed to dataframe
y = df['price'].values

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#create split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


#now we scale post split
scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)      #transform and fit data
X_test = scaler.transform(X_test)           #we dont want to fit test data, only tranform it

#CREATE A MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

model = Sequential()

model.add(Dense(19,activation='relu'))          #this may be overkill
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(1))                             #final layer only has one neuron for the price

model.compile(optimizer='adam',loss='mse')

#tensorflow needs .values data,so if we hadnt converted from DF yet, we could do it now with .values
#batch size often done in powers of 2, smaller the batch size = longer training will take but less
#likely to overfit data, only testing on small batches per epoch
model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=128,epochs=400)    

#now we evaluate
losses = pd.DataFrame(model.history.history)
losses.plot()       #plot the loss history, note training and validation loss are similar = good

#NOTE : if loss plot starts to go up again, it means youre overfitting to your data

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
predictions = model.predict(X_test) #get real predicted results

mean_absolute_error(y_test,predictions)
np.sqrt(mean_squared_error(y_test,predictions)) #rms error, note its not great, off by 100k roughly

explained_variance_score(y_test,predictions) #tells how much variance is explained by our model

plt.scatter(y_test,predictions) #scatterplot of predicitons vs real values
plt.plot(y_test,y_test,'r')     #add a line of the real values and colour it red

#we can see from this that our predicitons are good for lower priced houses but are getting skewed
#by the higher priced houses

errors = y_test.reshape(6480, 1) - predictions
sns.distplot(errors)

#to test for a single new set of values
single_house = df.drop('price',axis=1).iloc[0]  #grab the first house on the list (minues the price)
single_house = scaler.transform(single_house.values.reshape(-1, 19)) #transform and scale these values

model.predict(single_house)         #make a prediction
df.iloc[0]['price']         #check vs actual value





#Try again with top 1% removed to see if its any better

one_percent_removed = one_percent_removed.drop('id',axis=1)       #we dont need the id column as it contains no info
one_percent_removed['date'] = pd.to_datetime(one_percent_removed['date']) #convert date to a datetime object

one_percent_removed['month'] = one_percent_removed['date'].apply(lambda date:date.month)  #extract the month and create a new column
one_percent_removed['year'] = one_percent_removed['date'].apply(lambda date:date.year) #do the same for the year

one_percent_removed = one_percent_removed.drop('date',axis=1)     #since we have the dates now in month and year, we dont need date col
one_percent_removed = one_percent_removed.drop('zipcode',axis=1)

#NOW WE SET THE TRAIN SETS SPLIT
X = one_percent_removed.drop('price',axis=1).values      #calling values to get Array as opposed to dataframe
y = one_percent_removed['price'].values


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#create split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


#now we scale post split
scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)      #transform and fit data
X_test = scaler.transform(X_test)           #we dont want to fit test data, only tranform it

#CREATE A MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

model = Sequential()

model.add(Dense(25,activation='relu'))          #this may be overkill
model.add(Dense(25,activation='relu'))
model.add(Dense(25,activation='relu'))          #i changed the vals to try to improve
model.add(Dense(25,activation='relu'))          #because it was actually worse
model.add(Dense(25,activation='relu')) 
model.add(Dense(1))                             #final layer only has one neuron for the price

model.compile(optimizer='adam',loss='mse')

#tensorflow needs .values data,so if we hadnt converted from DF yet, we could do it now with .values
#batch size often done in powers of 2, smaller the batch size = longer training will take but less
#likely to overfit data, only testing on small batches per epoch
model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=128,epochs=400)    

#now we evaluate
losses = pd.DataFrame(model.history.history)
losses.plot()       #plot the loss history, note training and validation loss are similar = good

#NOTE : if loss plot starts to go up again, it means youre overfitting to your data

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
predictions = model.predict(X_test) #get real predicted results


mean_absolute_error(y_test,predictions)
np.sqrt(mean_squared_error(y_test,predictions)) #rms error, note its not great, off by 100k roughly

explained_variance_score(y_test,predictions) #tells how much variance is explained by our model

plt.scatter(y_test,predictions)














