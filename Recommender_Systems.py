# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:14:33 2021

@author: Cillian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
 #create a list of column names

df = pd.read_csv('u.data', sep='\t', names=column_names) #create dataframe for ratings
movie_titles = pd.read_csv("Movie_Id_Titles") #this is a dataframe of the movie titles
#we want to merge these such that instead of the movie _id, we have the name of the movie

df = pd.merge(df,movie_titles,on='item_id') #merging them on item ID
#this doesnt replace the names but it adds an extra column with a name to represent the ID


#Exploratory Data Analysis (EDA)
sns.set_style('white')

#create a dataframe with the rating of each movie and then the number or ratings each movie has
#step one, group by title, take rtating column and sort them and check the df head
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()

#now check the movies with the most ratings
df.groupby('title')['rating'].count().sort_values(ascending=False).head()

#now create a df based off these
#create ratings df with the mean rating for each title
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())

#now add a number of ratings column to this dataframe
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())


#now we want to explore the data a bit
plt.figure(figsize=(10,4))#gets the number of ratings for each film in a hist
ratings['num of ratings'].hist(bins=70)
#could also use seaborn
plt.figure(figsize=(10,4))  #can adjust size as you would matplotlib
sns.distplot(ratings['num of ratings'], kde = False, bins = 70)

#do the same for the ratings themselves
plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)

#can use a jointplot to see the relationship between ratings and number of ratings
sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)

#now we build the recommender system (Recommends similar movies)
#matrix that has the user ids on one access and the movie title on another axis, wil be a lot of nan values
#as most people wont have seen most movies
moviemat = df.pivot_table(index='user_id',columns='title',values='rating')

ratings.sort_values('num of ratings',ascending=False).head(10) #check movies with most ratings

#checking two moives, get a series for each movie for the rating each user gave it
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']

#We can then use corrwith() method to get correlations between two pandas series
#get the correlation between starwars ratings and other movies ratings, then do same for liar liar
#ie how correlated are users ratings of other movies with that persons rating of starwars
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])   #turn this into a dataframe
corr_starwars.dropna(inplace=True)                      #drop NaN values

corr_starwars.sort_values('Correlation', ascending = False).head(5)
#we can see that weirdness happens (selecting movies that have only been watched by someone who rated starwars high)
#so we want to filter out movies with a low number of views
corr_starwars = corr_starwars.join(ratings['num of ratings']) #adding the number of ratings to the df
#as we have the title as the index, the will join appropriately. 

#now do the filtering by setting number of ratings to 100
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head(10)

#do the same for liarliar
corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()












