# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 09:48:06 2018

@author: welcome
"""


A z-score indicates how many standard deviations an element is from the mean.
A z-score can be calculated from the following formula. 

how to calculate SD 

Work out the Mean (the simple average of the numbers)

Then for each number: subtract the Mean and square the result.

Then work out the mean of those squared differences.

Take the square root of that and we are done!


Outlier Detection

import pandas as pd 

import numpy as np 

from scipy import stats 

data = pd.read_csv("D:/Big Data/Hadoop Admin Tutorials/sample data/cars.csv")

data.head()

data.shape

data = data.drop(columns=['Model','Origin','Origin_values'])



z = np.abs(stats.zscore(data))

print (z)

#threshold = 3 

print(np.where(z>3))

print(z[4][3])

data_o = data[(z<3).all(axis=1)]

type(data_o)

data_o.shape

------------------------------------------------------------------------  



