"""
Created on Wed Sep 19 11:36:32 2018

@author: admin
"""

#linear regression Dataset

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split 
from sklearn import metrics

-------------------------------------------------------------------------------
#importing the data
usahousing = pd.read_csv("E:/datasets/USA_Housing.csv")

usahousing.head()

usahousing.shape

print(usahousing.columns.values)

-------------------------------------------------------------------------------
#printing the null values

print(usahousing.isnull().sum())

-------------------------------------------------------------------------------
#checking outliers in the dataset

usahousing = usahousing.drop(columns=["Address"])

z= np.abs(stats.zscore(usahousing))

usah1 = usahousing[(z<3).all(axis=1)]

usah1.shape

usahousing= usah1

corr= usahousing.corr

corr

sn.heatmap(corr)

-------------------------------------------------------------------------------
#choosing tghe independent variable

X= usahousing.drop("Price",axis=1)

X.shape

-------------------------------------------------------------------------------
#Linear Regression

lm=LinearRegression()

lm.fit(X,usahousing.Price)

help(lm)

print('Estimated Coefficient',lm.intercept_)

print('Number of Coefficients',len(lm.coef_))

pd.DataFrame(list(zip(X.columns,lm.coef_)),columns=['features','Estimated coefficients'])

-------------------------------------------------------------------------------
#visualisation

plt.scatter(usahousing.Avg_ai,usahousing.Price,color=['b','r'])

plt.xlabel("average area income")

plt.ylabel("Prices")

plt.title("relationship btw area income and prices")

plt.xlim(100,100000)

plt.ylim(100000,1000000)

plt.show()



