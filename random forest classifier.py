# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 18:53:24 2018

@author: welcome
"""



# Required Python Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


data = pd.read_csv("C:/Users/welcome/ran_new.csv")

data  = data.drop(['ID'],axis=1)

data.columns
data.shape

X = data.iloc[:,0:9]
Y = data.iloc[:,9]

X.head()
X.shape

Y.shape
Y.head()


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

def random_forest_classifier(features, target):
    clf = RandomForestClassifier()
    clf.fit(features, target)
    print(clf)
    return clf 

trained_model = random_forest_classifier(X_train,Y_train) 
predictions = trained_model.predict(X_test)
 
for i in range(0, 5):
    print("Actual outcome :: {} and Predicted outcome :: {}".format(list(Y_test)[i], predictions[i]))

model = RandomForestClassifier(n_estimators=15)

model.fit(X_train,Y_train)

estimater = model.estimators_[10]

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import collections
from sklearn import tree


features = X.columns
#class_names = Y.columns

dot_data = tree.export_graphviz(estimater,feature_names = features, out_file=None,rounded=True)

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

graph.write_png('random_classifier.png')

print("Train Accuracy :: ", accuracy_score(Y_train, trained_model.predict(X_train)))
print("Test Accuracy  :: ", accuracy_score(Y_test,predictions))
print("Confusion matrix ", confusion_matrix(Y_test, predictions))




