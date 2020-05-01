# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 10:32:48 2018

@author: welcome
"""

''' Syntax of If Statements 

if condition :
   indentedStatementBlock
   
1. If
2. Else 
3. Else If
4. Nested If '''

# If Statements 

a = int(input(" Enter the first your number "))
b = int(input("Enter the second number"))

if a > b:
   print("your first number is greater than your second number")
-------
# If Else Statements 

a = int(input(" Enter the first your number "))
b = int(input("Enter the second number"))

if a > b:
    print("your first number is greater than your second number")
else:
    print ("your second number is greater than your first number")    
---------------
#Else If statements 
a = int(input(" Enter the first your number "))
b = int(input("Enter the second number"))
c = int(input("Enter your Third Number"))
if (a>b) & (a>c):
    print("your first number is greater than your second & third number")
elif b>c:
    print ("your second number is greater than your first & third number")
else:
    print ("your third number is highest")

# Nested If Statements
country = str(input("Enter your country name"))
luggage_weight = int(input("Enter your Luggage_weight"))
if country == "INDIA":
    if luggage_weight >=100:
        print("your logistics cost is $1000")
    else:
        print("your logistics cost is $2500")
else:
    print("free")


