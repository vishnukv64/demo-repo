# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 18:13:54 2018

@author: admin
"""

#cuisine dataset
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import random
import gc
import seaborn as sns

train = pd.read_json("E:/datasets/train/train.json")
sub = pd.read_json("E:/datasets/test/test.json")

train.head()

train.shape


#EDA

ed = train['cuisine'].value_counts().plot(kind='bar', title ="Cuisine types", figsize=(8, 5), legend=True, fontsize=10)
ed.set_ylabel("recipes", fontsize=10)
ed.set_xlabel("Cuisines",fontsize=10)
plt.show()

ed = train['ingredients'].value_counts().plot(kind='jointplot', title ="Cuisine types", figsize=(8, 5), legend=True, fontsize=10)
ed.set_ylabel("cusine", fontsize=10)
ed.set_xlabel("ingredients",fontsize=10)
plt.show()

#preprocessing

new_test=pd.DataFrame()
new_train=pd.DataFrame()
cut_df=pd.DataFrame()
cut_percentage=0.01  
for cuisine in train['cuisine'].drop_duplicates().values :
    temp=pd.DataFrame()
    temp=train[train['cuisine']==cuisine]
    rows_test = random.sample(list(temp.index), round(0.3*(1-cut_percentage)*len(train[train['cuisine']==cuisine])))
    new_test=new_test.append(temp.ix[rows_test])
    rows_train= random.sample(list(temp.drop(rows_test).index), round(0.7*(1-cut_percentage)*len(train[train['cuisine']==cuisine])))
    new_train=new_train.append(temp.ix[rows_train])
    rows=rows_test+rows_train
    cut_df=cut_df.append(temp.drop(rows))
    del temp

ax=plt.subplot()
CuisineCall = list(range(0,len(cut_df['cuisine'].value_counts().index)))
LABELS=cut_df['cuisine'].value_counts().index
ax.bar(CuisineCall,cut_df['cuisine'].value_counts(),width=0.5,color='r',align='center',label='cut data')
ax.bar(CuisineCall,new_train['cuisine'].value_counts(),width=0.5,color='b',align='center', label='new train data')
ax.bar(CuisineCall,new_test['cuisine'].value_counts(),width=0.5,color='g',align='center',label='new test data')
plt.xticks(CuisineCall, LABELS,rotation=85)
ax.autoscale(tight=True)
plt.legend()

plt.show()



X = train.loc[:,train.columns!='ingredients']
Y = train.loc[:,train.columns=='cuisine']



