# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 09:32:52 2018

@author: welcome
"""
'''Operators
x+y - 
x-y
x*y
x/y = 
x//y
x%y
pow(x,y)
x**y'''


pow(2,3)
2**3

# What are data types in Python

'''1. Numbers 
       (i) Integer
       (ii) Float
       (iii) Complex
   2. Strings 
   3. List 
   4. Tuple 
   5. Set
   6. Dictionaries
   7. Conversion between data types '''
#you can use min / max / len all other function in list / tuples 

# 1. Numbers 
   
# Integers
i = 1
type(i)

#Float
J = 0.89
type(J)

#Complex 
c = complex(8.,-3.)
c
type(c)

#2. Strings 

s = "Hi"


s[0:3]
s[1]
s.upper()
S.lower()
len(s)
type(S)

#3. List
a = [10,100,"bye"]
a
len(a)
type(a)
a[2]

#4. Tuples 
b = (10,100,"bye")
b
type(b)

# Difference between Tuple & List 

a.insert(3,"hi")
a

b[1] = 1000

del b[-1]
del a[-1]

#5.Set 

d = set([1,2,4,5,"egg","non-veg"])
d
type(d)
len(d)

#6. Dictionaries 
Dict = {'S.no':'hi','name':'Prakash','address':'Kpm'}
Dict
Dict['S.no']

#Conversion between data types

a_int = 100
b_float = 32.5 
c = a_int + b_float
print(c)
type(c)
c = int(a_int + b_float)
c
type(c)


#Conversion between list / tuples 

list_1 = [1,2,3,4,5]
tuple_1 = (2,3,4,5)
list_1
type(list_1)
tuple_1
type(tuple_1)

print(type(tuple(list_1)))

print(type(list(tuple_1)))

# Converrting string into list 

string_list = list("Hello")
int_list    = list("12345")


string_list[0]
int_list[0]






