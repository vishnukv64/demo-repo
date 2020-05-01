# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 16:48:34 2018

@author: welcome
"""

Probelm Statement

We will use this dataset to try and predict gas consumptions (in millions of gallons) in 48 US states
based upon gas tax (in cents), per capita income (dollars), paved highways (in miles) and 
the proportion of population with a drivers license.



import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline

data = pd.read_csv("E:/datasets/petrol_consumption.csv")

data.head()

data.describe()

X = data.drop('Petrol_Consumption', axis=1)  

y = data['Petrol_Consumption']  

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeRegressor  
regressor = DecisionTreeRegressor(max_depth=3)  
regressor.fit(X_train, y_train) 

y_pred = regressor.predict(X_test)  

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  

df

from sklearn import metrics
from sklearn import tree  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  

Decision Tree Visualization

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import collections

features = X.columns

dot_data = tree.export_graphviz(regressor,feature_names = features, out_file=None,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')

edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree1.png')

