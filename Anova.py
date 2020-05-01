# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 05:01:09 2018

@author: welcome
"""
ANOVA is used to compare the means of three or more samples.

1.	One way ANOVA (It contains one independent variable)
2.	Two way ANOVA (It contains two independent variable)



One way anova 

'''The Probelem statement is there is no difference between means of weights in below three difference groups''''

group1 = [45,67,34,78,90,100,56,78,88,55] --69.1

group2 = [78,45,77,24,78,90,100,25,50,70] - 63.7

group3 = [89,98,67,45,34,70,65,20,75,99] -- 66.2

from scipy import stats
stats.f_oneway(group1,group2,group3)

0.05 < 0.1 #reject null hypo


--------------------------------------------------------------------

Two way anova

''' The problem statement which variable is useful for giving good length of tooth. 
download the dataset Tooth growth.
Here we are using three variable like Len, supp, dose'''

import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy import stats

data = pd.read_csv("C:/Users/welcome/Downloads/ToothGrowth.csv")

formula = 'len~C(supp)+C(dose)+C(supp):C(dose)'

model = ols(formula,data).fit()

aov_table = anova_lm(model,typ=2)

print(aov_table)

----------------------------------------------------------

how to interpret with two way anova results 

0.05 < 0.0318


F = variation between sample means / variation within the samples

sum_sq = It is defined as being the sum, over all observations, of the squared differences of each observation from the overall mean.
------------------------------------------------------------




