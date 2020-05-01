# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:27:51 2018

@author: welcome
"""

# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])

---------------------------------------------------- 



# Feature Extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features:", fit.n_features_)
print("Selected Features:",fit.support_)
print("Feature Ranking:",fit.ranking_)
----------------------------------------------------------------------------------- 


# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
dataframe.to_csv("Model selction.csv") 

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)

-------------------------------------------------------------------------------- 
X.shape
Y.shape

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

from sklearn import metrics, linear_model

lm = linear_model.LogisticRegression()
model = lm.fit(X_train, y_train)

print("Score", model.score(X_test, y_test))

from sklearn.model_selection import KFold # import KFold

kf = KFold(n_splits=10)

kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator

print(kf) 

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


from sklearn.cross_validation import cross_val_score, cross_val_predict

scores = cross_val_score(model, X_test, y_test, cv=2)

print("Cross-validated scores:", scores)

----------------------------------------------------------------------




