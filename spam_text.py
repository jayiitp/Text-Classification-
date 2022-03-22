#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 22:36:08 2022

@author: life
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection._split import train_test_split


##fetching the data
data=pd.read_csv('smsspam.tsv',sep='\t')
print(data.head())

#checking whether it has any null value
print(data.isnull().sum())

#unique label
print(data['label'].unique())

#creating data
X=data['message']
y=data['label']


#split the data

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=(42))


#get the model
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

lr=Pipeline([('tfid',CountVectorizer()),('lr',LinearSVC())])
lr.fit(X_train, y_train)
lr.score(X_train, y_train)
predict_value=lr.predict(X_test)


#measure the accuracy
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

print(accuracy_score(y_test, predict_value))
print(confusion_matrix(y_test, predict_value))
print(classification_report(y_test, predict_value))
