# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 19:04:01 2018

@author: admin
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#reading the json file and data visualisation
train = pd.read_json("E:/datasets/train/train.json")

train.head()
print (train.isnull().sum())
train.shape

ed = train['cuisine'].value_counts().plot(kind='bar', title ="Cuisine types", figsize=(8, 5), legend=True, fontsize=10)
ed.set_ylabel("recipes", fontsize=10)
ed.set_xlabel("Cuisines",fontsize=10)
plt.show()

#preprocessing and data validation
y = train[['cuisine']]
print(y.head())
X  = train[['ingredients']]
print(X.head())

docs = []
for ingredient in train.ingredients:
    temp = ""
    for item in ingredient:
        temp = temp + item + " "
    docs.append(temp)
print(len(docs))

#need to convert or transform all list into vectors 
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(docs)
print(X_transformed)
X_transformed.shape
target_enc = LabelEncoder()
y = target_enc.fit_transform(train.cuisine)
y = y.reshape(-1)

#splitting the datasets to fit the model
data_train, data_test, label_train, label_test = train_test_split(X_transformed, y, test_size=0.33, random_state=7)
data_train
label_train.shape
label_test = label_test.reshape(-1,1)
label_test.shape

#applying logistic regression algorithm
logreg = LogisticRegression()
logreg.fit(data_train, label_train)

#Predicting the test set results and calculating the accuracy
print('Accuracy test set: {:.2f}'.format(logreg.score(data_test, label_test)))
Y_pred = logreg.predict(data_test)

print(Y_pred)

#we have 77% accuracy with out split train and test dataset



