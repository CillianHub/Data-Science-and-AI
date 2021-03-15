# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:15:35 2021

@author: Cillian
"""

import pandas as pd
import numpy as np

df = pd.read_csv('TensorFlow_FILES/DATA/cancer_classification.csv')

X = df.drop('benign_0__mal_1',axis=1).values
y = df['benign_0__mal_1'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard #also importing tensorboard
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

pwd #check filepath

#Tensorboard
from datetime import datetime
datetime.now().strftime("%Y-%m-%d--%H%M")   #gives the current datetimestamp, will name our folders

log_directory = 'logs\\fit'         #will creat a logs file 

board = TensorBoard(log_dir=log_directory,histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch',
    profile_batch=2,
    embeddings_freq=1)

model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=15,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')


model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop,board]
          )


print(log_directory)

pwd









