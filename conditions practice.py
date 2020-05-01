# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 13:08:31 2018

@author: admin
"""

#Studying conditions
# If condition
a=100
b=200
if b>a:
    print("b is greater than a")
    
#elif and else condition
    
a=50
b=200
if a>b:
    print("a is greater than b")
elif a==b:
    print("a and b are equal")
else:
    print("b is greater than a")
    
#nested if 
    
Country =  str(input("Enter your country"))
luggage_weight = int(input("Enter the weight of luggage"))
if Country=="INDIA":
    if luggage_weight >=100:
        print("the luggage cost is 1000$")
    else:
        print("the luggage cost is 2500$")
else:
    print("free")
    