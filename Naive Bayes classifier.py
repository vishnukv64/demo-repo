# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 19:16:45 2018

@author: welcome
"""
Probelm statement : 

we will build a machine learning model to use tumor information to predict whether or not 
a tumor is malignant or benign.

import pandas as pd 
import numpy as np
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
type(data)
breast_cancer = pd.DataFrame(data.data, columns = data.feature_names)
breast_cancer['target'] = data['target']
breast_cancer.head()
breast_cancer.to_csv("breast Cancer dataset.csv")




import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(10,1)
plt.figure(figsize=(6,30*4))   

for i, col in enumerate(breast_cancer[breast_cancer.iloc[:,0:30].columns]):
    ax5 = plt.subplot(gs[i])
    sns.distplot(breast_cancer[col][breast_cancer.target==1],bins=50, color='r')
    sns.distplot(breast_cancer[col][breast_cancer.target==0],bins=50, color='g')
    ax5.set_xlabel('')
    ax5.set_title('feature:' + str(col))
plt.show()

corr = breast_cancer.corr()
sns.heatmap(corr)

from sklearn.model_selection import train_test_split

X = breast_cancer.loc[:,breast_cancer.columns!='target']
Y = breast_cancer.loc[:,breast_cancer.columns=='target']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.33,random_state=5)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

model = gnb.fit(X_train,Y_train)

Y_pred = gnb.predict(X_test)

print(Y_pred)

print(Y_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test,Y_pred))

This means that 94.15 percent of the time the classifier is able to make the correct prediction 
as to whether or not the tumor is malignant or benign.
These results suggest that our feature set of 30 attributes are good indicators of tumor class.

------------------------- 

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

model = LR.fit(X_train,Y_train)

Y_pred = LR.predict(X_test)

print(accuracy_score(Y_test,Y_pred))

--------------------------------------------

help(datasets)






