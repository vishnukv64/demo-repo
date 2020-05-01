# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:11:06 2018

@author: admin
"""
import pandas as pd
import json as js
import csv as csv
import scipy as scipy
import numpy as np
import pdb
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_json("E:/datasets/train/train.json")
train.head()
train.shape

#EDA
ed = train['cuisine'].value_counts().plot(kind='bar', title ="Cuisine types", figsize=(8, 5), legend=True, fontsize=10)
ed.set_ylabel("recipes", fontsize=10)
ed.set_xlabel("Cuisines",fontsize=10)
plt.show()



with open("E:/datasets/train/train.json") as json_data:
    data = js.load(json_data)
    json_data.close()

classes = [item['cuisine'] for item in data]
ingredients = [item['ingredients'] for item in data]
unique_ingredients = set(item for sublist in ingredients for item in sublist)
unique_cuisines = set(classes)

big_data_matrix = scipy.sparse.dok_matrix((len(ingredients), len(unique_ingredients)), dtype=np.dtype(bool))

for d,dish in enumerate(ingredients):
    for i,ingredient in enumerate(unique_ingredients):
        if ingredient in dish:
            big_data_matrix[d,i] = True

clf2 = BernoulliNB(alpha = 0, fit_prior = False)
f = clf2.fit(big_data_matrix, classes)
result = [(ref == res, ref, res) for (ref, res) in zip(classes, clf2.predict(big_data_matrix))]
accuracy_learn = sum (r[0] for r in result) / len(result)

print('Accuracy on the learning set: ', accuracy_learn)

with open("E:/datasets/test/test.json") as json_data_test:
    test_data = js.load(json_data_test)
    json_data_test.close()

ingredients_test = [item['ingredients'] for item in test_data]
big_test_matrix = scipy.sparse.dok_matrix((len(ingredients_test), len(unique_ingredients)), dtype=np.dtype(bool))
for d,dish in enumerate(ingredients_test):
    for i,ingredient in enumerate(unique_ingredients):
        if ingredient in dish:
            big_test_matrix[d,i] = True

result_test = clf2.predict(big_test_matrix)
ids = [item['id'] for item in test_data]
result_dict = dict(zip(ids, result_test))

writer = csv.writer(open('submission.csv', 'wt'))
writer.writerow(['id','cuisine'])
for key, value in result_dict.items():
   writer.writerow([key, value])

print('Result saved in file: submission.csv')