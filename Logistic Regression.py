# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 18:35:07 2018

@author: welcome
"""

Logistic Regression is a Machine Learning classification algorithm
that is used to predict the probability of a categorical dependent variable.
In logistic regression, the dependent variable is a binary variable
that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.).


the logistic model (or logit model) is a statistical model
that is usually taken to apply to a binary dependent variable

Data
The dataset comes from the UCI Machine Learning repository,
and it is related to direct marketing campaigns (phone calls)
of a Portuguese banking institution.
The classification goal is to predict whether the client will subscribe (1/0)
to a term deposit (variable y). The dataset can be downloaded from here.

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns

url = "https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv"

data = pd.read_csv(url)

type(data)

data.head()

data.to_csv('banking.csv')

data.shape

list(data.columns)
--------------------------------------------------------------------- 
Input variables

age (numeric)

job : type of job (categorical: “admin”, “blue-collar”, “entrepreneur”, “housemaid”, “management”, “retired”, “self-employed”, “services”, “student”, “technician”, “unemployed”, “unknown”)
marital : marital status (categorical: “divorced”, “married”, “single”, “unknown”)
education (categorical: “basic.4y”, “basic.6y”, “basic.9y”, “high.school”, “illiterate”, “professional.course”, “university.degree”, “unknown”)
default: has credit in default? (categorical: “no”, “yes”, “unknown”)
housing: has housing loan? (categorical: “no”, “yes”, “unknown”)
loan: has personal loan? (categorical: “no”, “yes”, “unknown”)
contact: contact communication type (categorical: “cellular”, “telephone”)
month: last contact month of year (categorical: “jan”, “feb”, “mar”, …, “nov”, “dec”)
day_of_week: last contact day of the week (categorical: “mon”, “tue”, “wed”, “thu”, “fri”)
duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y=’no’). The duration is not known before a call is performed, also, after the end of the call, y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model
campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
previous: number of contacts performed before this campaign and for this client (numeric)
poutcome: outcome of the previous marketing campaign (categorical: “failure”, “nonexistent”, “success”)
emp.var.rate: employment variation rate — (numeric)
cons.price.idx: consumer price index — (numeric)
cons.conf.idx: consumer confidence index — (numeric)
euribor3m: euribor 3 month rate — (numeric)
nr.employed: number of employees — (numeric)
--------------------------------------------------------------------

Predict variable (desired target):

y — has the client subscribed a term deposit? (binary: “1”, means “Yes”, “0” means “No”)

The education column of the dataset has many categories and
we need to reduce the categories for a better modelling.


data.dtypes

----------------------------------------------------------------------

data['education'].unique()

data['marital'].unique()

data['contact'].unique()

data['education'] = np.where(data['education']=='basic.9y','Basic',data['education'])

numpy.where(condition[, x, y])
Return elements, either from x or y, depending on condition.

data['education'] = np.where(data['education']=='basic.4y','Basic',data['education'])

data['education'] = np.where(data['education']=='basic.6y','Basic',data['education'])

data['education'].unique()
----------------------------------------------------------------- 

Data Exploration 


data['y'].value_counts()

sns.countplot(x='y', data=data)

----------------------------------------------- 

Getting percentage of subcriber details 

count_no_sub = len(data[data['y']==0])

count_sub = len(data[data['y']==1])

pct_no_sub = count_no_sub/(count_no_sub+count_sub) 

print("Percentage of no subscription is",pct_no_sub*100)

pct_sub = count_sub/(count_no_sub+count_sub) 

print("Percentage of Yes subscription is",pct_sub*100)

Our classes are imbalanced,
the ratio of no-subscription to subscription instances is 89:11

--------------------------------------------------------------------------------------- 

data.groupby('y').mean()

data.groupby('job').mean()

data.groupby('marital').mean()

data.groupby('education').mean()
----------------------------------------------------- 

%matplotlib inline

pd.crosstab(data.job,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')

pd.crosstab(data.marital,data.y).plot(kind='bar')
plt.title('Purchase Frequency for martial status Title')
plt.xlabel('martial')
plt.ylabel('Frequency of Purchase')

pd.crosstab(data.education,data.y).plot(kind='bar')
plt.title('Purchase Frequency for martial status Title')
plt.xlabel('martial')
plt.ylabel('Frequency of Purchase')

data.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
----------------------------------------------------- 

Create dummy variables

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']

for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1

var
cat_list

data.columns.values

type(data)

cat_vars

data_vars = data.columns.values.tolist()

data_vars

type(data_vars)

to_keep = [i for i in data_vars if i not in cat_vars]

data_final=data[to_keep]

data_final.head(10)
data_final.columns.values

-------------------------------------------------------------------

Imbalanced data sets are a special case for classification problem
where the class distribution is not uniform among the classes.
Typically, they are composed by two classes: The majority (negative) class and
the minority (positive) class.

to increase the number of underepresented cases in a dataset
used for machine learning. SMOTE is a better way of increasing the number of rare cases
than simply duplicating existing cases.

The module returns a dataset that contains the original samples,
plus an additional number of synthetic minority samples,
depending on the percentage you specify.

SMOTE stands for Synthetic Minority Oversampling Technique.
This is a statistical technique for increasing the number of cases in your dataset in a balanced way.
The module works by generating new instances from existing minority cases that you supply as input.
This implementation of SMOTE does not change the number of majority cases.


SMOTE implementions

X = data_final.loc[:,data_final.columns!='y']
Y = data_final.loc[:,data_final.columns=='y']

from imblearn.over_sampling import SMOTE

os = SMOTE(random_state=0)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=0)

columns = X_train.columns

columns

os_data_X, os_data_Y = os.fit_sample(X_train,Y_train)

os_data_X = pd.DataFrame(data=os_data_X, columns=columns)

os_data_Y = pd.DataFrame(data=os_data_Y, columns=['y'])

print(" Length of oversampled data", len(os_data_X))

print("No of no subscription in oversampled",len(os_data_Y[os_data_Y['y']==0]))

print("Number of Yes subcription in oversampled",len(os_data_Y[os_data_Y['y']==1]))

print("Proportion of no subscription data in oversampled data is ",len(os_data_Y[os_data_Y['y']==0])/len(os_data_X))

print("Proportion of subscription data in oversampled data is ",len(os_data_Y[os_data_Y['y']==1])/len(os_data_X))


------------------------------------------------------------------- 

Recursive Feature Elimination

Recursive Feature Elimination (RFE) is based on the idea to repeatedly construct a model
and choose either the best or worst performing feature,
setting the feature aside and then repeating the process with the rest of the features.
This process is applied until all features in the dataset are exhausted.
The goal of RFE is to select features by recursively considering smaller and smaller sets of features


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

len(os_data_X.columns)
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_Y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

--------------------------------------------------------

checking with P-Value with all columns 

cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'default_no', 'default_unknown', 
      'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"]


X=os_data_X[cols]
Y=os_data_Y['y']


import statsmodels.api as sm
logit_model=sm.Logit(Y,X)
result=logit_model.fit()
print(result.summary2())

cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 
      'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 


X=os_data_X[cols]
Y=os_data_Y['y']


X.head()
Y.head()

ogit_model=sm.Logit(Y,X)
result=ogit_model.fit()
print(result.summary2())

-------------------------------------------------------- 

Logistic Regression Model Fitting

from sklearn.linear_model import LogisticRegression
from sklearn import metrics


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)


Predicting the test set results and calculating the accuracy

print('Accuracy test set: {:.2f}'.format(logreg.score(X_test, Y_test)))

Y_pred = logreg.predict(X_test)

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(Y_test, Y_pred)

print(confusion_matrix)

The result is telling us that we have 6124+5170 correct predictions
and 2505+1542 incorrect predictions.

from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_pred))

Interpretation: 
Of the entire test set, 74% of the promoted term deposit were the term deposit that the customers liked.
Of the entire test set, 74% of the customer’s preferred term deposits that were promoted.

ROC Curve

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(Y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(Y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

The receiver operating characteristic (ROC) curve is another common tool used with binary classifiers.
The dotted line represents the ROC curve of a purely random classifier;
a good classifier stays as far away from that line as possible (toward the top-left corner).



