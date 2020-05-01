# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 11:36:49 2018

@author: welcome
"""

# for statements 

for a in range(3):
    print(a)
------------------------------------

for a in range(3):
    print(a+1)
-----------------------------------
for a in [1,2,3]:
    print(a)    
---------------------------
for a in 'hello':
    print(a)
------------------------
list_1 = ["Hello","python"]

for i in list_1:
    print(len(i))
---------------------------------

for i in range(10):
    print(i)
    if i==9:
        print ("reached range")
--------------------------------------------------
colours = ["RED"]
for i in colours:
    if i == "RED":
        colours +=["BLACK"]
    if i == "BLACK":
        colours +=["WHITE"]
print(colours)
-----------------------------------------------
# finding all the numbers in list 

print("Enter your list of numbers by comma")
s = input()
s
type(s)
numbers = s.split(",")
numbers
type(numbers)
sum = 0
type(sum)
for j in numbers:
    sum+=int(j)
print("The sum is",sum)
--------------------------------------------------------


