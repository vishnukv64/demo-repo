# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 22:21:06 2018

@author: welcome
"""
It is a statement about one or more populations. 
It is usually concerned with the parameters of the population. 
e.g. the hospital administrator may want to test the hypothesis that the average length of stay of patients
admitted to the hospital is 5 days

Decision : Reject H0 if P-value <alpha 

Null hypothesis H0: It is the hypothesis to be tested
Alternative hypothesis HA : It is a statement of what we believe is true if our sample data cause us to reject the null hypothesis


a = [10,12,9,11,11,12,9,11,9,9]
b = [13,11,9,12,12,11,12,12,10,11]

from scipy import stats
c = stats.ttest_ind(a,b)
print(c)
c

H0 < alpha 

0.05 < 0.08 # reject Null Hpothethis
-------------------------------------------------------------------------

'''Let’s say we want to calculate the resting systolic blood pressure of 20 first-year resident female doctors
 and compare it to the general public population mean of 120 mmHg.
The null hypothesis is that there is no significant difference between the blood pressure of the resident female doctors
and the general population'''

from scipy import stats

female_doctors = [128, 127, 118, 115, 144, 142, 133, 140, 132, 131,111, 132, 149, 122, 139, 119, 136, 129, 126, 128]

stats.ttest_1samp(female_doctors,120)

HO < alpha 

0.05 < 0.002 # reject Null Hpothethis

--------------------------------------------------------------------

'''Let’s look at an example to compare the blood pressure of male consultant doctors with the junior resident female doctors.
we explored above Our null hypothesis in this case is that there is no statistically significant difference in the mean of male consulting doctors 
and junior resident female doctors'''

female_doctor_bps = [128, 127, 118, 115, 144, 142, 133, 140, 132, 131, 
                     111, 132, 149, 122, 139, 119, 136, 129, 126, 128]

male_consultant_bps = [118, 115, 112, 120, 124, 130, 123, 110, 120, 121,
                      123, 125, 129, 130, 112, 117, 119, 120, 123, 128]

stats.ttest_ind(female_doctor_bps, male_consultant_bps)

# reject Null Hpothethis


--------------------------------------------------------------

Paired T-Test

'''We will measure the amount of sleep got by patients before and after taking soporific drugs to help them sleep.
The null hypothesis is that the soporific drug has no effect on the sleep duration of the patients'''

control = [8.0, 7.1, 6.5, 6.7, 7.2, 5.4, 4.7, 8.1, 6.3, 4.8]
treatment = [9.9, 7.9, 7.6, 6.8, 7.1, 9.9, 10.5, 9.7, 10.9, 8.2]


stats.ttest_rel(control, treatment)

--------------------------------------------------------------











