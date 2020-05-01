# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 11:11:37 2018

@author: welcome
"""

Probelm Statment

The problem we will tackle is predicting the max temperature for tomorrow in our city using 
one year of past weather data. 


import pandas as pd
import numpy as np
df = pd.read_csv("c:/Users/welcome/Downloads/temps.csv")

Following are explanations of the columns:

year: 2016 for all data points

month: number for month of the year

day: number for day of the year

week: day of the week as a character string

temp_2: max temperature 2 days prior

temp_1: max temperature 1 day prior

average: historical average max temperature

actual: max temperature measurement

friend: your friend’s prediction, a random number between 20 below the average and 20 above the average

df.shape

df.describe()

# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(df)
# Display the first 5 rows of the last 12 columns
features.iloc[:,5:].head(5)

features.shape

features.head()



# Labels are the values we want to predict
labels = np.array(features['actual'])

#Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('actual', axis = 1)

#Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

