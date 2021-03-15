# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:44:16 2021

@author: Cillian
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

customers = pd.read_csv('Ecommerce customers')
describe = customers.describe()
customers.info()
sns.jointplot(data = customers, x = 'Time on Website', y = 'Yearly Amount Spent')
sns.jointplot(data = customers, x = 'Time on App', y = 'Yearly Amount Spent')

#seems to be more of a correlation between time on app
sns.jointplot(data = customers, x = 'Time on App', y = 'Length of Membership', kind = 'hex')

#get more info by looking at all the data with pairlpot
sns.pairplot(customers)
#can see that 'Yearly Amount Spent' seems to have the highest correlation with Length of membership

sns.lmplot(x='Length of Membership',y ='Yearly Amount Spent', data = customers)
#can see a good linear fit as error is low 

customers.columns
y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

#splitting up test/train data with y = target value = Yearly amount spent and X = numerical cols
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#now train the data and check the coefficients
lm = LinearRegression() 
lm.fit(X_train,y_train)
lm.coef_

#create predictions based off this model for the test data
predictions = lm.predict(X_test)

#now compare real values vs predictions
fig = sns.jointplot(y_test, predictions)        #or use plt.scatter() wih plt.xlabel()
fig.set_axis_labels('y_test', 'Predicted values')

#now Evaluate the model by checking the error
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rms = np.sqrt(mse)

metrics.explained_variance_score(y_test,predictions)  #get variance score

#now check residuals to make sure its normally distributed
sns.distplot(y_test-predictions) #can alter bins if needs be

#now check coeffs vs columns
cdf=pd.DataFrame(lm.coef_, X.columns, columns = ['Coeffs'])
# we can see theres a far higher correlation between app time and money spent vs time on
#website and money spent. They could focus on the app as thats the money maker or the website
#to make it catch up to the app


