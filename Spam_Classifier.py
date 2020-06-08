# Building a model for Spam Classifier using Naive Bayes Algorithm for Test Data

import nltk
import re
import pandas as pd
import numpy as np
from nltk import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

dataset=pd.read_csv("E:\\ExcelR\\Learning Codes\\NLP & TM\\Implementation of Spam Classifier in Python\\SMSSpamCollection",sep="\t",names=["labels","messages"])
corpus=[]

for i in range(len(dataset)):
    cs=re.sub("[^a-zA-Z]"," ",dataset["messages"][i])
    cs=cs.lower()
    cs=cs.split()
    cs=[PorterStemmer().stem(word) for word in cs if word not in set(stopwords.words("english"))]
    cs=" ".join(cs)
    corpus.append(cs)
               
# Converting the corpus into bag of Words(to get our independent variable)
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()

# To get our dependent variable

Y=pd.get_dummies(dataset["labels"])
Y=Y.drop("ham",axis=1)

# Splitting the data into train and test

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)

from sklearn.naive_bayes import MultinomialNB

model=MultinomialNB().fit(X_train,Y_train)

Y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm=confusion_matrix(Y_test, Y_pred)
acc_score=accuracy_score(Y_test, Y_pred) #97.94 % accuracy
