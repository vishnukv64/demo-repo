# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 16:06:01 2018

@author: admin
"""

==============================================================================
#Functions assignment


#1
 
def sqrt(x):
    return x*x

print(sqrt(7))

===========================================================================================
#2

def avg(b):
    return sum(b)/len(b) 

b= [50,60,70,90]
avg= avg(b)

print("the average of the given numbers is=", round(avg,2))

=====================================================================================
#3

list1 =[50,300,1500,9800,7844,56,66,887,2123,4785,652,4856,1000,6666,4512]
 
list1_new = list(filter(lambda x:(x>1000),list1))
print(list1_new)
list2= print(list1_new)

for i in list1:
    if i>1000:
        i*500
        print(i*500)

====================================================================================        
#4
'''
ReferenceError	Raised when a weak reference proxy is used to access a garbage collected referent.
RuntimeError	Raised when an error does not fall under any other category.
StopIteration	Raised by next() function to indicate that there is no further item to be returned by iterator.
SyntaxError	Raised by parser when syntax error is encountered.
IndentationError	Raised when there is incorrect indentation.
'''

-------------------------------------------------------------------------------
#class assignment

#1

class new:
    def add(a,b):
        return a+b
    def multiply(c,d):
        return c*d

print(new.add(10,7))
print(new.multiply(8,8))

===================================================================================
#2

class sch:
    
    sch_name= "Asan"

def __init__(self,name,rollno,standard):
    self.name=name
    self.rollno=rollno
    self.standard=standard
        
def add(self,address):
     return "{} is located in {}".format(self.name,address)
 
details = sch("vishnu",145263, "twelve")

print("You have now enquired about out student name is {} and rollno {} and standard {}".format(details.name,details.rollno,details.standard))

===================================================================================

#4

class dota:
    
    def __init__(self):
        self.__sales = 1000
    
    def free_to_play(self):
        print ("Free to play is {}".format(self.__sales))

    def setmoney(self,steam):
        self.__sales = steam   
        return self.__sales
        
dota_update = dota()

print(dota_update.free_to_play())

print(dota_update.setmoney(2000))


=============================================================================================
#5

class steam:
    def steam_sale(self):
        print("The list of the games are \
              1. PUBG \
              2. cs.go")
    def NA_games(self):
        print("overwatch")

class steam_new:
    def steam_sale(self):
        print("All games are available")
    def NA_games(self):
        print("NONE")
        
def details(info):
    info.steam_sale()
    
steam1 = steam()

steam_new1 = steam_new()

details(steam1)
details(steam_new1)

==============================================================================================

#6


Operator Overloading Special Functions in Python

Operator	         Expression	     Internally
Addition	         p1 + p2	         p1.__add__(p2)
Subtraction	     p1 - p2	         p1.__sub__(p2)
Multiplication	  p1 * p2	         p1.__mul__(p2)
Power	            p1 ** p2	         p1.__pow__(p2)
Division	        p1 / p2	         p1.__truediv__(p2)
Floor Division	  p1 // p2	         p1.__floordiv__(p2)
Remainder (modulo)	p1 % p2	     p1.__mod__(p2)
Bitwise Left Shift	p1 << p2	     p1.__lshift__(p2)
Bitwise Right Shift	p1 >> p2	     p1.__rshift__(p2)
Bitwise AND	       p1 & p2	     p1.__and__(p2)
Bitwise OR	          p1 | p2	         p1.__or__(p2)
Bitwise XOR	      p1 ^ p2	         p1.__xor__(p2)
Bitwise NOT	      ~p1             p1.__invert__()

=================================================================================================

#6

#class related errors

1. Syntax errors
2. exceptions
3. Handling exceptions
4. defining cleanup actions
5. Logical excecptions