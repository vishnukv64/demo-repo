# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 19:44:40 8

@author: welcome
"""

print ("Today we going to cover these types in Functions")

print ("Functions \n •	Function definition and call \n •	Function Scope \n • Arguments \n •	Function Objects \n •	Anonymous Functions \n  •Packaging Importing") 


''' What is Function 
When we want do the repeated task again & again with different respected parameters. So we are created 
functions do this task & it can use anywhere in programming'''' 

'''Syntax

def function_name(arguments):
    Function block''' 
    

----------------------------------

#without arguments 
def sample():
    return "hi"
    
sample()    

print(sample())

----------------------------------------

#with arguments 
def yournames(a,b,c):
    return a

yournames("prakash","siva","Prakash")
yournames(1,2,3)

--------------------------------------------------------- 

def add_num(x,y):
    print('first number is %d and second number is %d'%(x,y))
    return x+y

add_num(1,2)
-------------------------------------------------------

def int_int_str(sno,pincode,name):
    print('your s.no is %d \n'
          'your pincode is %d \n'
          'your name is %s'%(sno,pincode,name))

int_int_str(100,631502,"prakash")

int_int_str(10,10,10)

-------------------------------------------------------------

# Default arguments value

def default_value(name,address = "Kanchipuram"):
    print('Your name is %s \n'
           'your address is %s' %(name,address))

default_value("prakash")

default_value("John",address = "Chennai")
--------------------------------------------------------------
#interchanging the arguments

def chan_arg(x,y,z):
    return x,y,z

chan_arg(x=10,z=20,y=19)

--------------------------------------------------------------------
# Mutaable Arguments

def mut_arg(name,add_name=[]):
    for item in name:
        add_name.append(item)
    return add_name

mut_arg((20,30))

----------------------------------------------------------------------------
#accepting the variable arguments

''''
1. In addition to named arguments, functions can accept two special collections of arguments.

2. The first is a variable-length, named tuple of any additional positional arguments received by the function.
This special argument is identified by prefixing it with a single asterisk (*).

3. The second is a variable-length dictionary containing all keyword arguments passed to the function
that were not explicitly defined as part of the function arguments. 
This argument is identified by prefixing it with two asterisks (**).

'''

def var_arg(*name,**sno):
    print(name)
    print(sno)

var_arg(('besent','Techno','kodambakkam'))

var_arg(1,2,3,d=10)

----------------------------------------------------------------------------

#Function Scope 

''''
Variables can only reach the area in which they are defined, which is called scope.
Think of it as the area of code where variables can be used.
Python supports global variables (usable in the entire program) and local variables.''''

Local Variables 

def func(c,d):
    a=100
    print(a)
    return c,d

func("hi",100)
a
-----------------------------------------------------------------------------

Global Variables 

a = 100

def func(c,d):
    print(a)
    return c,d

func("hi",100)

a
------------------------------------------------------------------------------

#Anonymous Functions
'''
anonymous function means that a function is without a name.
As we already know that def keyword is used to define the normal functions.
The lambda keyword is used to create anonymous functions. It has the following syntax:

    lambda arguments: expression''''


def sqrt(x):
    return x*x

g = lambda x :x*x

print(g(7))
print(sqrt(7))
---------------------------------------------------------

Lambda functions can be used along with built-in functions like filter(), map() and reduce().

#lambda functions with filter()

This functions will be return true values only. 

list1 = [100,200,300,400,500,600]

list_fin = list(filter(lambda x: (x<=300),list1))

print(list_fin)

for i in list1:
    if i >=300:
        print(i)

----------------------------------------------------

#lambda functions with map()
This function will be useful for return all modified values as list. 
when we want to do any modifications for large dataset, this function will be useful.

names=["Tamilnadu","Kerala","Karnataka","Andira","Orisha","Delhi","Tamilnadu"]

final_names=list(map(lambda x: x=="Tamilnadu",names))
print(final_names)

----------------------------------------------------------------------

#lambda functions with map()

The function is called with a lambda function and a list and a new reduced result is returned.

list1 = [1,2,3,4,5,5,6,6,7,8,8]
from functools import reduce
sum = reduce((lambda x, y:x-y),list1)
print(sum)
---------------------------------------------------------------------

# Import Modules 

The Python programming language comes with a variety of built-in functions. Among these are several common functions, including:
•	print() which prints expressions out
•	abs() which returns the absolute value of a number
•	int() which converts another data type to an integer
•	len() which returns the length of a sequence or collection

how to install package in python IDE : 
    
    syntax : pip install packagename

how to import module 
    syntax : import module name 

how to access the model from package 

    syntax : from pakckagename import modulename 

how to call a function from module 

    syntax : [module].[function]

------------------------------------------------------------------ 

#import random 
for i in range(100):
    print(random.randint(1,50))

-------------------------------------------------------------------


from random import randint
for i in range(100):
    print(randint(1,10))
-----------------------------------------------------------------------
















    
    



















