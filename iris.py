# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 19:16:45 2018

@author: welcome
"""
Probelm statement : 


import pandas as pd 
import numpy as np
from sklearn.datasets import load_iris
data = load_iris()
type(data)
load_iris = pd.DataFrame(data.data, columns = data.feature_names)
load_iris['target'] = data['target']
load_iris.head()
load_iris.to_csv("load iris dataset.csv")




import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(5,1)
plt.figure(figsize=(6,5*4))   

for i, col in enumerate(load_iris[load_iris.iloc[:,0:5].columns]):
    ax5 = plt.subplot(gs[i])
    sns.distplot(load_iris[col][load_iris.target==1],bins=10, color='r')
    sns.distplot(load_iris[col][load_iris.target==0],bins=10, color='g')
    ax5.set_xlabel('')
    ax5.set_title('feature:' + str(col))
plt.show()

corr = breast_cancer.corr()
sns.heatmap(corr)

from sklearn.model_selection import train_test_split

X = load_iris.loc[:,load_iris.columns!='target']
Y = load_iris.loc[:,load_iris.columns=='target']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.33,random_state=5)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

model = gnb.fit(X_train,Y_train)

Y_pred = gnb.predict(X_test)

print(Y_pred)

print(Y_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test,Y_pred))

This means that 94 percent of the time the classifier is able to make the correct prediction 


------------------------- 

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

model = LR.fit(X_train,Y_train)

Y_pred = LR.predict(X_test)

print(accuracy_score(Y_test,Y_pred))

--------------------------------------------







