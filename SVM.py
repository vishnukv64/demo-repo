# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 15:24:04 2018

@author: welcome
"""

Probelm Statement 

Our task is to predict whether a bank currency note is authentic or not
based upon four attributes of the note 

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline

bankdata = pd.read_csv("C:/Users/welcome/Downloads/bill_authentication.csv")  

bankdata.shape  

bankdata.head()  

X = bankdata.drop('Class', axis=1)  
y = bankdata['Class']  

X.shape
y.shape

from sklearn.model_selection import train_test_split  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

from sklearn.svm import SVC  
from sklearn import svm
svclassifier = SVC(kernel='linear')  

svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix  

from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test,y_pred))  

print(classification_report(y_test,y_pred))

-------------------------------------------------------------- 
from sklearn.datasets.samples_generator import make_blobs 

# creating datasets X containing n_samples 
# Y containing two classes 
X, Y = make_blobs(n_samples=1000, centers=2, 
                  random_state=0, cluster_std=0.40) 
  
# plotting scatters  
plt.scatter(X[:,], X[:,], c=Y, s=50, cmap='spring'); 
plt.show()
---------------------------------------------------------------- 
# creating line space between -1 to 3.5  
xfit = np.linspace(-1, 3.5) 
  
# plotting scatter 
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='autumn')
plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)

  
# plot a line between the different sets of data 
for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]: 
    plt.plot(xfit,m*xfit+b,'-k') 
plt.xlim(-1, 3.5); 

----------------------------------------------- 
