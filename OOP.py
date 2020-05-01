# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 06:36:21 2018

@author: welcome
"""

#Creating Class 

class new:
    a=1000
    msg ="Welcome to OOPS Concept"
    

print(new.a)    
print(new.msg)

--------------------------------------------- 

#Class with function

class new1:
    def add(a,b):
        return a+b

print(new1.add(2,3))    

------------------------------------------------ 

# Creating class and object 

class movie:
    #Class attribute 
    movies = "watching"
    
    #Instance Attribute 
    def __init__(self,name,price):
        self.name = name
        self.price = price

#Instantiate the movie Class
Kab = movie("Kabali",1000)
En20 = movie("Enthiran",5000)

#access the class attributes 

print("your movie is {}".format(Kab.__class__.movies))
print("your movie is {}".format(Kab.__class__.movies))

#access the instance attribute 

print("your watching movie is {} price is {}".format(Kab.name,Kab.price))
print("your watching movie is {} price is {}".format(En20.name,En20.price))

-----------------------------------


# creating methods & functions inside the class

class emp:
    
    # Class attribute
    com_name ="TCS"
    
    # Instance attribute
    def __init__(self,name,empid,dept):
        self.name = name
        self.empid = empid
        self.dept = dept
    # Methods (functions)
    def add(self,address):
        return "{} is located in {}".format(self.name,address)
    
    def sal(self,salary):
        return "{} salary is {} and he is in {} dept".format(self.name,salary,self.dept)

#initalize the object
        
details = emp("John",12345,"Operations")

# access the class attributes 

print("You are enquired about the {} Company".format(details.__class__.com_name))

#access the instance attribute

print("The Employee name is {} Emp is {} and worked in dept {}".format(details.name,details.empid,details.dept))

#access the functions 

print(details.add("Chennai"))

print(details.sal("100000"))

-------------------------------------------------------------------------


# •	Inheritance

# Parent Class

class student:
 
    def __init__(self, name, course):
        self.name = name      # __name is private to Vehicle class
        self.course = course
 
    def getaddress(self,address):          # getName() is accessible outside the class
        return self.address
 
class studentnew(student):
 
    def __init__(self, name, course, ph):
        # call parent constructor to set name and color  
        super().__init__(name, course)       
        self.ph = ph
 
    def getDescription(self):
        return "student name"+ self.name + "course is " + self.course + "ph no is" + self.ph
 
# in method getDescrition we are able to call getName(), getColor() because they are 
# accessible to child class through inheritance
 
c = studentnew("Prakash", "data science", "1235")
print(c.getDescription())


-------------------------------------------------------

#Multiple Inheritance 

#Parent Class

class parentclass1():
 
    def method1(self):
        print("This is function of first parent class")
 
class parentclass2():
 
    def method2(self):
        print("This is the function of second parent class")
 
class child(parentclass1, parentclass2):
 
    def child_method(self):
        print("This is the child method")
out = child()
out.method1()
out.method2()

----------------------------------------------------------------------------


# Encapsulation 

class besent:
    
    def __init__(self):
        self.__fees = 1000
    
    def offer_course(self):
        print ("Course Fess are {}".format(self.__fees))

    def setfees(self,feesnew):
        self.__fees = feesnew   
        return self.__fees
        
besent_update = besent()

print(besent_update.offer_course())

print(besent_update.setfees(2000))

---------------------------------------------------------------------------- 

#Ploymorphism 

class besent:
    def av_course(self):
        print("The list of the course are \
              1. Data Science \
              2. Python")
    def NA_course(self):
        print("Net working")

class besent_new:
    def av_course(self):
        print("All course are available")
    def NA_course(self):
        print("NONE")
        
def details(info):
    info.av_course()
    
besent1 = besent()

besent_new1 = besent_new()

details(besent1)
details(besent_new1)

------------------------------------------------------------------------------

#Operation overloading 

class sample:
    
    def __init__(self,name):
        self.__name = name
        
    def names(self):
        return self.__name

    def __add__(self,new_name):
        return sample(self.__name + new_name.__name)

a = sample("Prakash")

b = sample("Python")

c = a +b 

print(c.names())

-------------------------------------------------------------------------



















