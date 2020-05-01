# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 09:38:28 2018

@author: welcome
"""

import matplotlib.pyplot as plt

# x axis values 
x = [1,2,3,4,5,6,7,8] 
# corresponding y axis values 
y = [2,4,1,5,8,9,1,5] 

z= [0,2,1,5,4,5,5,5,5,4,45,5,5]
# plotting the points  
plt.plot(x, y) 
  
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 
  
# giving a title to my graph 
plt.title('My first graph!') 
  
# function to show the plot 
plt.show() 

----------------------------------------------

Two line plots on same Graph 

import matplotlib.pyplot as plt 
  
# line 1 points 
x1 = [1,2,3] 
y1 = [2,4,1] 
# plotting the line 1 points  
a = plt.plot(x1, y1, label = "line 1") 
  
# line 2 points 
x2 = [1,2,3] 
y2 = [4,1,3] 
# plotting the line 2 points  
b = plt.plot(x2, y2, label = "line 2") 
  
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 
# giving a title to my graph 
plt.title('Two lines on same graph!') 
  
# show a legend on the plot 
plt.legend() 
  
# function to show the plot 
plt.show(a,b) 
--------------------------------------------------------------


character	color
‘b’	blue
‘g’	green
‘r’	red
‘c’	cyan
‘m’	magenta
‘y’	yellow
‘k’	black
‘w’	white


---------------------------------------------------- 


character	description
'-'	solid line style
'--'	dashed line style
'-.'	dash-dot line style
':'	dotted line style
'.'	point marker
','	pixel marker
'o'	circle marker
'v'	triangle_down marker
'^'	triangle_up marker
'<'	triangle_left marker
'>'	triangle_right marker
'1'	tri_down marker
'2'	tri_up marker
'3'	tri_left marker
'4'	tri_right marker
's'	square marker
'p'	pentagon marker
'*'	star marker
'h'	hexagon1 marker
'H'	hexagon2 marker
'+'	plus marker
'x'	x marker
'D'	diamond marker
'd'	thin_diamond marker
'|'	vline marker
'_'	hline marker
----------------------------------------------- 


x1 = [1,2,3] 
y1 = [2,4,1] 

#setting x and y axis range 
plt.ylim(0,8) 
plt.xlim(0,8) 

# plotting the line 1 points  
plt.plot(x1, y1, color='r', linestyle ='dashed', label = "line 1") 

---------------------------------------------------------------------------------------

import pandas as pd 

#how to import the dataset using pandas package

data = pd.read_csv("D:/Big Data/Hadoop Admin Tutorials/sample data/cars.csv") # read_csv functions will use the import data 

print(data) # print the data 
----------------------------------------------------------------------

y = data['MPG']

x = data['Cylinders']

type(y)

plt.ylim(1,50) 
plt.xlim(1,10)

x
# plotting the points  
plt.plot(x, y) 
  
# naming the x axis 
plt.xlabel('x - axis --> Cyclinders') 
# naming the y axis 
plt.ylabel('y - axis --> MPG') 
  
# giving a title to my graph 
plt.title('Line Graph for entire data set') 
  
# function to show the plot 
plt.show() 

-----------------------------------------------     

BAR Chart 

x = [1,2,3,4,5]
y = [100,200,300,400,500]

plt.xlabel('X- Axis')
plt.ylabel('Y-Axis')

plt.bar(x,y,color=['r','g','y'])


--------------------------------------------- 
Dataset using bar chart 

create bars 

bandwidth = 0.5 

b1 = data['MPG']
b2 = data['Accelerate']

#Creating the position of bars 

p1 = data['Cylinders']

# create barplot 

a = plt.bar(p1,b1,width = bandwidth, color='yellow',label = 'MPG')

b = plt.bar(p1,b2,width = bandwidth, color='green',label = 'Accelerate')

plt.title("Cars Dataset")

plt.xlabel("Cyclinders Value")

plt.ylabel("MPG & Accelerate")

plt.legend()

plt.show(a)

------------------------------------------------------------------------ 

Histogram

# frequencies 

ages = [10,20,20,40,50,60,60,70,70,70]

# setting the ranges 

range = (0,80)

bins = 10

plt.hist(ages,bins,range,color='blue',histtype='bar')

# x-axis label 
plt.xlabel('age') 
# frequency label 
plt.ylabel('No. of people') 
# plot title 
plt.title('My histogram') 
  
# function to show the plot 
plt.show()
------------------------------------------------- 

how to use histogram in dataset 

bandwidth = 0.5

year = data['Year']

range = (70,84)

bins = 10

plt.hist(year,bins, range,color='blue',histtype='bar', width = bandwidth)
# x-axis label 
plt.xlabel('Year"s') 
# frequency label 
plt.ylabel('No of times accurred ') 
# plot title 
plt.title('Cars dataset') 

plt.legend()
  
# function to show the plot 
plt.show()

------------------------------------------------------------------

how to use scatter plot in dataset 

mpg = data['MPG']
acc = data['Accelerate']
year = data['Year']

# x-axis label 
plt.xlabel('Years') 
# frequency label 
plt.ylabel('MPG & Accelerate') 
# plot title 
plt.title('scatter plot!') 

plt.scatter(year,mpg,color='r') 
plt.scatter(year,acc,color='g')

# showing legend 
plt.legend() 

plt.show()


mpg = data['MPG']
acc = data['Accelerate']
values = data['Origin_values']

# x-axis label 
plt.xlabel('Years') 
# frequency label 
plt.ylabel('MPG & Accelerate') 
# plot title 
plt.title('scatter plot!') 

plt.scatter(values,mpg,color='r') 
plt.scatter(values,acc,color='g')

# showing legend 
plt.legend() 

plt.show()

----------------------------------------------------------------------
mport matplotlib.pyplot as plt 
  
# defining labels 
activities = ['eat', 'sleep', 'work', 'play'] 
  
# portion covered by each label 
slices = [3, 7, 8, 6] 
  
# color for each label 
colors = ['r', 'y', 'g', 'b'] 
  
# plotting the pie chart 
plt.pie(slices, labels = activities, colors=colors,  
        startangle=90, shadow = True, explode = (0, 0, 0.1, 0), 
        radius = 1.2, autopct = '%1.1f%%') 
  
# plotting legend 
plt.legend() 
  
# showing the plot 
plt.show() 

-------------------------------------------------------- 

box plot 

mpg = data['MPG']

year = data['Year']

mpg = [1,2,3,4,5,6,90,500,50]

plt.boxplot(mpg)

plt.boxplot(mpg,0,'')

--------------------------------------------------- 
import seaborn as sb

Distribution plt

sb.distplot(mpg)

-------------------------------------------------------- 

























