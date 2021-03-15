# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:53:52 2021

@author: Cillian
"""
#Will someone pay off a loan, based on historical data

#Do the imports needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout



#Exploratory Data Analysis
df = pd.read_csv('TensorFlow_FILES/DATA/lending_club_loan_two.csv')
df['loan_status'].count()
sns.countplot(x='loan_status',data=df) #do some checks on the label, we can see they are not well balanced

#create a histogram of the loan amount
sns.histplot(x='loan_amnt',data=df, bins = 40)  #spikes indicate loans frequently at even numbers

plt.figure(figsize=(12,7))      #have a look at the correlations
sns.heatmap(df.corr(),annot=True,cmap='viridis')    #loan amt very highly correlated with installment rate

#they are probably using a formula to determine installment based off loan amnt, we may need to ignore installments
sns.scatterplot(x='installment',y='loan_amnt',data=df)

sns.boxplot(x='loan_status',y='loan_amnt',data=df) #check to see if theres a relationship between loan 
#amnt and if it was paid, we can see theres a very slight indication that larger loans are hard to pay off
print(df.groupby('loan_status')['loan_amnt'].describe()) #check the loan amnt relationship

#now explore the grades and subgrades systems in the dataset
grade_sorted = sorted(df['grade'].unique())  #we can see each unique value for grade and subgrade, sorted in this case for graphing
subg_sorted = sorted(df['sub_grade'].unique())

#we want a countplot per grade, with the hue = loan status
sns.countplot(x='grade',data=df,order = grade_sorted,hue='loan_status')  #we can see that for higher grades (a,b) are less likely to be charged off

plt.figure(figsize=(12,4))
sns.countplot(x = 'sub_grade', data = df, order = subg_sorted)
plt.figure(figsize=(12,4))          #get count plot for the subgrades
sns.countplot(x = 'sub_grade', data = df, order = subg_sorted, hue = 'loan_status') #get it with hue = loan_status


#say for example we just want to look at f and g and get their countplots
f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]       #df of f or g will take both
subgrade_order = sorted(f_and_g['sub_grade'].unique())  #get the sorted list that we want to get coutplots for
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status') #plot this ith regard to the subgrade


#now we want whether the loan_status is paid to be diplayed by either a 1 or a 0
df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})  #can do this with mapping
#now we want a bar chart showing the correlations of the different features with the label
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind = 'bar')






#now we want to deal with Missing Data
len(df)  #check th number of entries

df.isnull().sum()       #now we get the number of missing values per column
100*df.isnull().sum()/len(df)   #get this as a percentage of the number of data entries

#start with emp_title and emp_length (employment title and length(0-10))
#start with how many unique job titles are there
df['emp_title'].nunique()           #we can see there are a lot
df['emp_title'].value_counts()      #too many to convert to a dummy variable
df=df.drop('emp_title',axis=1)    #so we drop the title
sorted(df['emp_length'].dropna().unique())  #we want all the unique lengths of employment
emp_length_order = ['< 1 year',
 '1 year',
 '2 years',
 '3 years',                 #manually reordering as it didnt work properly
 '4 years',
 '5 years',
 '6 years',
 '7 years',
 '8 years',
 '9 years',
 '10+ years']

#now do a countplot for each year
plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order)

#see if it has any bearing on loan status by using hue
sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')

#we want to see is the percentage of charge offs per employment length category
emp_co=df[df['loan_status']=='Charged Off'].groupby('emp_length').count()['loan_status']
emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']

percent_charged_off = emp_co/(emp_co+emp_fp)
#we can see that for each group,they all have roughly the same ratio of people who charged off their loan
percent_charged_off.plot(kind = 'bar')

df = df.drop('emp_length',axis=1)   #since they dont have real input to the function, drop them

df.isnull().sum()   #check which features still have missing data

df['purpose'].head(10) #check out the purpose column
df['title'].head(10)  #and check the title (reason given by bank), we can see its the same info, so drop one

df = df.drop('title',axis=1) 

#what does the mort_acc do? its the number of mortgage accounts 
df['mort_acc'].value_counts() #we can see most poeple have 0 additional mortgage accounts, one person has 34
#we saw that there were almost 10% of mort_accs missing and we cant afford to lose 10% of our data
#so how do fill it in reasonably?

#is there any other feature that highly correlated with mort_acc?
df.corr()['mort_acc'].sort_values()   #looks like total accounts has decent correlation

#get the mean number of mort_accounts per total_acc:
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']  #we want to use thisas a lookup to fill values

def fill_mort_acc(total_acc,mort_acc):
    if np.isnan(mort_acc):                #check if value is missing from morrt_acc
        return total_acc_avg[total_acc]     #return the mean number of mort accounts for that total_acc num
    else:
        return mort_acc         #return the mort_acc

#now apply it with a lmbda statement
df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

df.isnull().sum() #make sure you have no nana values now

#since revol_util and pub_rec_bankruptcies are very low, we can just drop the rows that are missing the data
df.dropna(inplace = True)
df.isnull().sum()  #and check to make sure it worked





#categorical data and dummy variables
#first we want to look at all the data types that are objects (non numeric)
df.select_dtypes(['object']).columns
#check out the first one: term
df['term'].value_counts()   #we can see there are only two options, we can leave this as is as its numeric


df['term'] = df['term'].apply(lambda term: int(term[:3]))   #grab first two characters = the integer part

#we know that grade is part of subgrade, so we can drop the grade feature
df = df.drop('grade',axis = 1)

#convert subgrade into dummy variables
subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)
#now drop  the subgrade and concat the dummies
df = df.drop('sub_grade',axis=1)
df = pd.concat([df,subgrade_dummies],axis=1)

df.select_dtypes(['object']).columns #check object type columns again
#Convert these columns: ['verification_status', 'application_type','initial_list_status','purpose'] 
#into dummy variables and concatenate them as they have few categories
dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)

df.select_dtypes(['object']).columns
     
#now check out home ownership
df['home_ownership'].value_counts()
#replace NONE and ANY with OTHER since there are so few
df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

#now get dummies and concat
dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)

#looking at address, we will extract the zipocode by taking the last 5 characters
df['zip_code'] = df['address'].apply(lambda address:address[-5:])
df['zip_code'].value_counts()
#now get the zipcode dummies, drop the zipcode and addresses
dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)

df.select_dtypes(['object']).columns #check remaining data types

#issue_d is the month which the loan was fully piad, so this is data leakage as it relates to the label
df = df.drop('issue_d',axis=1)

#earliest_cr_line = month the borrowerest earliest line was opened
#we want to extract the year from this, this can be treated as continuous data
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:])) #we want last 4 chars
df = df.drop('earliest_cr_line',axis=1)


df.select_dtypes(['object']).columns

df = df.drop('loan_status',axis=1)  #we have this already changed to loan_repaid





#train_test_split
from sklearn.model_selection import train_test_split
X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values
print(len(df))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


#scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#create the model
model = Sequential()
# input layer
model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))
# hidden layer
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))
# hidden layer
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))
# output layer
model.add(Dense(units=1,activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam')


model.fit(x=X_train, y=y_train, epochs=25,batch_size=256,validation_data=(X_test, y_test))


#save the model
model.save('full_data_project_model.h5') 



#evaluate the model
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()


#get predictions
from sklearn.metrics import classification_report,confusion_matrix
predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions))
confusion_matrix(y_test,predictions)


#would you give this guy a loan?
import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer

model.predict_classes(scaler.transform(new_customer.values.reshape(1,78)))    #gives a one which means they did
#check whether they actually did or not
df.iloc[random_ind]['loan_repaid'] #they did!!










