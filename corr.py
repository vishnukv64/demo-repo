# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 19:45:24 2018

@author: welcome
"""



data

corr = data.corr()

corr

import seaborn as sn

sn.heatmap(corr)


The closer ρ is to 1,
the more an increase in one variable associates with an increase in the other. 

the closer ρ is to -1, 
the increase in one variable would result in decrease in the other. 

if X and Y are independent, then ρ is close to 0, 
but not vice versa! 


