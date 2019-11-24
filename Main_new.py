#!/usr/bin/env python
# coding: utf-8

import os
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import json
import random
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical, plot_model
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


cmdpos= "head -n 1 data/filtered.csv > data/sampled.csv && tail -n +2 data/filtered.csv | shuf -n 10000 >> data/sampled.csv"
cmdneg= "head -n 1 data/filtered.txt > data/sampled.txt && tail -n +2 data/filtered.txt | shuf -n 10000 >> data/sampled.txt"
os.system(cmdpos)
os.system(cmdneg)

# loading negative samples saved in sampled.txt

f = open('data/sampled.txt', 'r')
lines = f.readlines()

tweets = []
labels = []
model=["LSTM","Naive_Bayesian","Logistic_Regression"]

# varibles to split data for training and testing

len_train =8000
len_total=10000

# loading positive samples saved in sampled.csv

with open('data/sampled.csv', newline='') as csvfile:
    categories = csvfile.readline().split(",")
    tweetreader = csv.reader(csvfile, delimiter=',')
    counter = 0
    for row in tweetreader:
        tweet = dict(zip(categories, row))
        if tweet['language'] == 'English':
            tweets.append(tweet['content'])   # collecting only tweets with text content in english using the dictionary 
            labels.append(1)
            counter += 1
        if counter > len_total:		# capping total rows to len_total
            break
csvfile.close()

# appending postive labels to negative labels.

for line in lines:
    tweets.append(line)
    labels.append(0)

f.close()

            
tweets_to_labels = dict(zip(tweets, labels))

# shuffling rows of entire postive and negative labels.
random.shuffle(tweets)

#  seperating target variable i.e, Y from the shuffled data.

actual = []

for tweet in tweets:
    actual.append(tweets_to_labels[tweet])

# creating word embedings 

vectorizer = CountVectorizer(binary=True, lowercase=True)
total = vectorizer.fit_transform(np.array(tweets))

# splitting total data to train and test sets.

Xtrain = total[:len_train]
Ytrain=actual[:len_train]
Xtest = total[len_train:len_total]

#          LSTM

def lstm():
	#load positive tweets (random tweets)
	pos = open('data/sampled.csv').read()
	npos = 0
	label, texts = [], []
	for i, line in enumerate(pos.split("\n")):
	    content = line.split(',')
	    if len(content) < 4:
	    	continue;
	    if content[4] != "English":
	    	continue;
	    label.append(1)
	    texts.append(content[2])
	    npos += 1

	# load negative labels (random tweets)
	neg = open('data/sampled.txt').read()
	nneg = 0
	for i, line in enumerate(neg.split("\n")):
	    label.append(0)
	    texts.append(line)
	    nneg += 1

	texts, label = shuffle(texts, label)

	df = pd.DataFrame()
	df['text'] = texts
	df['label'] = label

	df.head()

	# encoding the tweets data with the labels 
	enc = LabelEncoder()
	y = enc.fit_transform(label)
	train_x, test_x, train_y, test_y = train_test_split(df['text'], y, test_size=0.20)

	maxlen = 280

	#  tokenizing the data for training 
	token = Tokenizer()
	token.fit_on_texts(df['text'])

	sequences = token.texts_to_sequences(train_x)
	padded = sequence.pad_sequences(sequences, maxlen=maxlen)

	# initializing the model 
	def make_model():
		inputs = Input(name='inputs',shape=[maxlen])
		layer = Embedding(len(token.word_index)+1,50,input_length=maxlen)(inputs)
		layer = LSTM(64, dropout=0.2, return_sequences=True)(layer)
		layer = LSTM(64, dropout=0.2)(layer)
		layer = Dense(256, name='FC1')(layer)
		layer = Activation('relu')(layer)
		layer = Dense(1, name='out_layer')(layer)
		layer = Activation('sigmoid')(layer)
		model = Model(inputs=inputs,outputs=layer)
		return model

	model = make_model()
	model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.00011), metrics=['accuracy'])

	history = model.fit(padded,train_y,batch_size=128,epochs=10,
	          validation_split=0.20)

	test_sequences = token.texts_to_sequences(test_x)
	test_padded = sequence.pad_sequences(test_sequences,maxlen=maxlen)
	results = model.predict(test_padded)

	accuracy = model.evaluate(test_padded, test_y)
	print('LSTM accuracy: \n', accuracy[1])

	# Plot training & validation accuracy values
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.ylim(0,1)
	plt.xlim(0,11)
	plt.legend(['Train_accuracy', 'Test_accuracy'], loc='upper left')
	plt.show()

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.ylim(0,1)
	plt.xlim(0,11)
	plt.legend(['Train_loss', 'Test_loss'], loc='upper right')
	plt.show()


#	  naive_bayes classifier

def Naive_Bayes(X_train,Y_train,X_test):

	Xtrain=X_train.toarray()
	Ytrain=Y_train
	Xtest=X_test.toarray()
	model = GaussianNB()
	model.fit(Xtrain, Ytrain)
	return model.predict(Xtest)


#     logistic_regression classifier

def Logistic_Regression(X_train,Y_train,X_test):

	Xtrain=X_train.toarray()
	Ytrain=np.array(Y_train)
	Xtest=X_test.toarray()
	lr=LogisticRegression(solver='liblinear')
	lr.fit(Xtrain,Ytrain)
	return lr.predict(Xtest)

# 	testing	accuracy

def print_accuracy(Ytest):
	pred=Ytest
	correct = 0
	true_positive = 0
	total_positive = 0
	true_negative = 0
	total_negative = 0

	for i in range(len(pred)):
	    if actual[i+len_train]:
	        total_positive += 1
	    else:
	        total_negative += 1
	    if pred[i] == actual[i+len_train]:
	        correct += 1
	        if actual[i+len_train]:
	            true_positive += 1
	        else:
	            true_negative += 1
	print(correct / len(pred))

#  implementing earlier mentioned techniques.

for i in model:
	if i=="LSTM":
		lstm()
	if i=="Naive_Bayesian":
		print("Naive Bayes accuracy:")
		print_accuracy(Naive_Bayes(Xtrain,Ytrain,Xtest))
	if i=="Logistic_Regression":
		print("Logistic Regression accuracy:")
		print_accuracy(Logistic_Regression(Xtrain,Ytrain,Xtest))
	
