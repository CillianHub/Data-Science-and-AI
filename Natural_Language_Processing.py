# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:32:30 2021

@author: Cillian
"""

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


nltk.download_shell() #hit l in command line to see available packages
#then press d to dowload and then type: 'stopwords', and press enter, then q to quit shell
#stopwords is used to get a list of commonly used words to be ignored 'and' 'as' 'the'...
message_test = [line.rstrip() for line in open('SMSSpamCollection')]    #read in data and split it up into reviews
print(len(message_test))        #rstrip removes unwanted spaces at the end of each line

for message_no, message in enumerate(message_test[:10]):        #to print the first ten messages
    print(message_no, message)
    print('\n')

#to read CSV to dataframes, we need to know how to break up the dataset, ie separate data for each line by tab
messages= pd.read_csv('SMSSpamCollection', sep='\t',names=["label", "message"])

#exploratory data analysis
messages.describe()

messages.groupby('label').describe() #check the number of each type of message

#use feature engineering to create some  new data variables from the data for example,message length
messages['Length'] = messages['message'].apply(len)

#plot out the lenght of the messages in a histogram
messages['Length'].plot(bins=50, kind='hist')
#we can see that there is one or two very large messages
messages.Length.describe()
messages[messages['Length'] == 910]['message'].iloc[0] #location of the loingest message

#make a histogram (distplot in sns) and use 'by' to separate data by label
messages.hist(column='Length', by='label', bins=50,figsize=(12,4))

#we can see that spam messages tend to have a longer message

#Text pre-processing
import string

mess = 'Sample message! Notice: it has punctuation.'

# Check characters to see if they are in punctuation
nopunc = [char for char in mess if char not in string.punctuation]
nopunc = ''.join(nopunc)    #punctuation now removed and rejoined
nopunc.split()      #now it is in list so we can access items individually

from nltk.corpus import stopwords #now we want to get rid of stopwords
stopwords.words('english')[0:10] #check the first ten stopwords

#now take out the stopwords
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#now we want to put all this in a function to apply to the entire dataframe

def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]  #remove punctuation
    nopunc = ''.join(nopunc) #rejoin into list
    #return the list of words without punctutation
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#now test on 'messages'
messages.head(5)
#apply it to the first five elements of the DF
messages['message'].head(5).apply(text_process)


#Vectorisation- Bag of words
from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
print(len(bow_transformer.vocabulary_))         #prints total number of vocab words

#check out the 4th message in the df
message4 = messages['message'][3]
print(message4)

#now we want to see its vectorisation
bow4 = bow_transformer.transform([message4])
print(bow4)     #we can see there are seven vectors so 7 unique words in that message, 2 of which appear twice
print(bow4.shape)

#see which ones appeared twice
print(bow_transformer.get_feature_names()[4068])
print(bow_transformer.get_feature_names()[9554])

messages_bow = bow_transformer.transform(messages['message']) #create transformer
print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)

#formula for sparsity
sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(sparsity))


from sklearn.feature_extraction.text import TfidfTransformer
#import tfidef transformer and create an instance of it, fitted to the bank of words
tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4) #check out the fourth message from earlier
print(tfidf4)               #term frequency inverse document frequency of the 4th argument = weighted value

#to check tfidf of a particular word
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

#now do this for the entire bag of words corpus into TF-IDF corpus 
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)


#train the model, can use most classification algorithms, here we will use naive bayes
from sklearn.naive_bayes import MultinomialNB
#create an insatnce of this and fit it with the TF-IDF  and the actual data labels
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

#now we can try it out on our single random message
spam_detect_model.predict(tfidf4)[0]
messages.label[3]


#predict for all cases
all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)


#should be using train test split, so lets do that
from sklearn.model_selection import train_test_split
msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3)
#note grabbing raw data which hasnt had stopwords and punctuation removed and vectorised,
#can either do it all again or do a pipeline 
#scikitlearns data pipeline feature below:
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

#now we can just pass in raw text objects into the pipeline and it will do the preprocessing for us

pipeline.fit(msg_train,label_train) #fit it like a normal estimator
predictions = pipeline.predict(msg_test)

#now check the accuracy of the predictions
from sklearn.metrics import classification_report
print(classification_report(predictions,label_test))


#easy to change the classification algorithm, eg
from sklearn.ensemble import RandomForestClassifier
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', RandomForestClassifier()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
pipeline.fit(msg_train,label_train) #fit it like a normal estimator
predictions = pipeline.predict(msg_test)

print(classification_report(predictions,label_test))

#very nice





