# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 17:15:46 2018

@author: welcome
"""



In simple linear regression, 
a single independent variable is used to predict the value of a dependent variable. 

Multiple Linear Regression 
more than one independent variable is used to predict the value of a dependent variable. 


Scikit-learn is a powerful Python module for machine learning. 
It contains function for regression, classification, clustering, 
model selection and dimensionality reduction. 
 
Problem Statement 

The goal of this exercise is to predict the housing prices in boston region using the features given.

Feature description 

Data description
crim
per capita crime rate by town.

zn
proportion of residential land zoned for lots over 25,000 sq.ft.
A proportion refers to the fraction of the total that possesses a certain attribute.
For example, suppose we have a sample of four pets - 
a bird, a fish, a dog, and a cat. ... 
Therefore, the proportion of pets with four legs is 2/4 or 0.50.

indus
proportion of non-retail business acres per town.

chas
Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

nox
nitrogen oxides concentration (parts per 10 million).

rm
average number of rooms per dwelling.

age
proportion of owner-occupied units built as prior to 1970 .

dis
weighted mean of distances to five Boston employment centres.

rad
index of accessibility to radial highways.

tax
full-value property-tax rate per $10,000.

ptratio
pupil-teacher ratio by town.

black
1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.

lstat
lower status of the population (percent).

medv
median value of owner-occupied homes in $1000s.

-------------------------------------- 

#Importing the Packages 

import pandas as pd
import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.cross_validation import train_test_split 

# Importing the dataset 

bostan = pd.read_csv("C:/Users/welcome/Downloads/boston.csv")

bostan.head()

bostan.shape

print(bostan.columns.values)

----------------------------------------------- 

# working on null values 

print(bostan.isnull().sum())
----------------------------------------------------
# Checking Outliers for bostan dataset 

z = np.abs(stats.zscore(bostan))

bostan_o = bostan[(z<3).all(axis=1)]

bostan_o.shape

bostan = bostan_o

bostan.shape

corr = bostan.corr()

corr

sn.heatmap(corr)
--------------------------------------------------------
# split the dataset training & test set 

Our ultimate target variable is MEDV.. that is considers as price. 

that we need to fix as Y. Dependtant variable (MV column)

x- indepentant variable 

X = bostan.drop('MV',axis=1) # x contains all the valeus except price column

X.head()

--------------------------------------------------------------------------------------------
# calculating the co-efficent values of all the vaiables


y = mx + b 

Y - Predicted value 
mx- The amount of impact our X has on our Y  
b - Y intercept
 - 

The constant term in linear regression analysis seems to be such a simple thing.
Also known as the y intercept,
it is simply the value at which the fitted line crosses the y-axis. 

The slope indicates the steepness of a line and the intercept indicates the location
where it intersects an axis.
The slope and the intercept define the linear relationship between two variables,
and can be used to estimate an average rate of change. 


lm = LinearRegression()

lm.fit(X,bostan.MV)

help(lm)
print('Estimated coefficient', lm.intercept_)

print('Number of Coefficients', len(lm.coef_))

pd.DataFrame(list(zip(X.columns, lm.coef_)), columns = ['Features' ,'Estimated Coeffiencies'])

Zip function is used to return the tuples of object 
--------------------------------------------------------------------------------------------------
# Visualization 

plt.scatter(bostan.RM,bostan.MV,color=['b','r'])
plt.xlabel("Average number of room")
plt.ylabel("Housing Price")
plt.title("Relationship between rm & Price")
plt.xlim(1,10)
plt.ylim(1,60)
plt.show()

there is a positive correlation between RM and housing prices.

---------------------------------------------------------------------------------------
# Split the dataset 

X_train, X_test, Y_train, Y_test = train_test_split(
        bostan.RM,bostan.MV,test_size=0.33, random_state=5)


print(type(X_train))
print(type(Y_train))
print(type(X_test))
print(type(Y_test))

print(X_train)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)



print(X_train.values.reshape(-1,1))
print(X_test.values.reshape(-1,1))
print(Y_train.values.reshape(-1,1))
print(Y_test.values.reshape(-1,1))

X_train = X_train.values.reshape(-1,1)
X_test = X_test.values.reshape(-1,1)
Y_train = Y_train.values.reshape(-1,1)
Y_test = Y_test.values.reshape(-1,1)

lm = LinearRegression() # create linear object

lm.fit(X_train,Y_train)

# pred_train = lm.predict(X_train)
Y_pred = lm.predict(X_test)

plt.scatter(Y_test,Y_pred, color=['b','r'])
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices and predicted Prices")

Ideally, the scatter plot should create a linear line. 
Since the model does not fit 100%, the scatter plot is not creating a linear line.

----------------------------------------------------------------------------------------
# Error Validating the model 

Mean Squared Error

The mean squared error tells you how close a regression line is to a set of points.
It does this by taking the distances from the points to the regression line (these distances are the “errors”) and squaring them.
The squaring is necessary to remove any negative signs.

To check the level of error of a model, 
we can calculate Mean Squared Error. 
It is one of the procedure to measures the average of the squares of error. 
Basically, it will check the difference between actual value and the predicted value. 

mse = metrics.mean_squared_error(Y_test,Y_pred)

print(mse)

The smaller the means squared error, the closer you are to finding the line of best fit.
Depending on your data, it may be impossible to get a very small value for the mean squared error. 


Root Mean Squared Error
Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors).
Residuals are a measure of how far from the regression line data points are;
RMSE is a measure of how spread out these residuals are. In other words,
it tells you how concentrated the data is around the line of best fit. 

It is calculated by taking the square root of Mean Squared Error. 
Conveniently, the RMSE as the same units as the quantity estimated (y). RMSE = np. sqrt(MSE)

print(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))

For the RMSE value, For good predictive model the chi and RMSE values should be low 
(<0.5 and <0.3, respectively).

Normally a RMSE > 0.5 is related to a bad predictive model.

a lower number for RMSE is better model
---------------------------------------------------------------------------- 

Residual Plots

Residual plots are a good way to visualize the errors in your data. 
If you have done a good job then your data should be randomly scattered around line zero.
If you see structure in your data, that means your model is not capturing some thing.
Maye be there is a interaction between 2 variables that you are not considering,
or may be you are measuring time dependent data.
If you get some structure in your data, you should go back to your model and
check whether you are doing a good job with your parameters.


plt.scatter(lm.predict(X_train), lm.predict(X_train)-Y_train,c=['b','r'],s=40,alpha=0.5)
#plt.scatter(lm.predict(X_test), lm.predict(X_test)-Y_test,c='g',s=40)
plt.hlines(y=0,xmin=0,xmax=50)
plt.title('Residual plot using training(blue) and testing(green) data')
plt.ylabel('Residual')

print("lr.coef_: {}".format(lm.coef_))
print("lr.intercept_: {}".format(lm.intercept_))
print("Training set score: {:.2f}".format(lm.score(X_train, Y_train)))
print("Test set score: {:.7f}".format(lm.score(X_test, Y_test)))

--------------------------------------------------------------------------------------------------

Creating model with another feature 

X_train, X_test, Y_train, Y_test = train_test_split(
        bostan.ZN,bostan.MV,test_size=0.33, random_state=10)

print(X_train.values.reshape(-1,1))
print(X_test.values.reshape(-1,1))
print(Y_train.values.reshape(-1,1))
print(Y_test.values.reshape(-1,1))

X_train = X_train.values.reshape(-1,1)
X_test = X_test.values.reshape(-1,1)
Y_train = Y_train.values.reshape(-1,1)
Y_test = Y_test.values.reshape(-1,1)

lm = LinearRegression() # create linear object


a = lm.fit(X_train,Y_train)

# pred_train = lm.predict(X_train)
Y_pred = a.predict(X_test)

plt.scatter(Y_test,Y_pred,c=['r','b'])
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices and predicted Prices")

mse = metrics.mean_squared_error(Y_test,Y_pred)

print(mse)

print(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))

plt.scatter(lm.predict(X_train), lm.predict(X_train)-Y_train,c=['b','r'],s=40,alpha=0.5)
#plt.scatter(lm.predict(X_test), lm.predict(X_test)-Y_test,c='g',s=40)
plt.hlines(y=0,xmin=0,xmax=50)
plt.title('Residual plot using training(blue) and testing(green) data')
plt.ylabel('Residual')

----------------------------------------------------------------------------------------- 

print("lr.coef_: {}".format(a.coef_))
print("lr.intercept_: {}".format(a.intercept_))
print("Training set score: {:.2f}".format(a.score(X_train, Y_train)))
print("Test set score: {:.7f}".format(a.score(X_test, Y_test)))

---------------------------------------------------------------------------------- 

Multiple Linear Regression 

X_train, X_test, Y_train, Y_test = train_test_split(
        bostan[['ZN','RM']],bostan.MV,test_size=0.33, random_state=5)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

lm = LinearRegression() # create linear object

a = lm.fit(X_train,Y_train)

# pred_train = lm.predict(X_train)
Y_pred = a.predict(X_test)

print("lr.coef_: {}".format(a.coef_))
print("lr.intercept_: {}".format(a.intercept_))
print("Training set score: {:.2f}".format(a.score(X_train, Y_train)))
print("Test set score: {:.7f}".format(a.score(X_test, Y_test)))

mse = metrics.mean_squared_error(Y_test,Y_pred)

print(mse)

print(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))

-----------------------------------------------------------------------------------
Model Tunnning using cross validation 

X_train, X_test, Y_train, Y_test = train_test_split(
        bostan[['ZN','RM']],bostan.MV,test_size=0.33, random_state=5)

lm = LinearRegression() # create linear object

a = lm.fit(X_train,Y_train)

# pred_train = lm.predict(X_train)
Y_pred = a.predict(X_test)

scores = cross_val_score(lm, Y_test, Y_pred, cv=5)
rmse_scores = np.sqrt(scores)

def display_scores(scores):
    print("Scores:",scores)
    print("Mean:", scores.mean())
    print("RMSE:", rmse_scores)

print(display_scores(scores))

----------------------------------------------------------------- 

