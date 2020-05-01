# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 19:54:24 2018

@author: welcome
"""
File Operations 

•	Opening a file
•	Using Files – txt ,Csv, Xlsx
•	Appending to file
•	Exporting file


------------------------------------------------


'''
File handling in Python requires no importing of modules. ''''

1. Open()
2. Read()
3. Append()
4. Close()

Mode of the Files 

'r' - reading

'w' - writing

'a' - appending

The open () has two arguments

Syntax : open(filename,"attributes")

------------------------------------------------------

# Opening & Reading text file 
import os 
print(os.getcwd())

filename = "C:/Users/welcome/hello.txt"

file = open(filename,"r")

for line in file:
    print (line)

file.close()

---------------------------------------------------------------

# Opening & Reading CSV file 

import csv
with open('sample1.csv', newline ='') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

--------------------------------------------------------------- 
# Opening & Reading Xlsx File 

import xlrd

workbook = xlrd.open_workbook('sample.xlsx')

worksheet = workbook.sheet_by_index(0)

first_row = [] # The row where we stock the name of the column
for col in range(worksheet.ncols):
    first_row.append( worksheet.cell_value(0,col) )

#transform the workbook to a list of dictionary 
data = []
for row in range(1,worksheet.nrows):
    elm={}
    for col in range(worksheet.ncols):
        elm[first_row[col]]=worksheet.cell_value(row,col)
    data.append(elm)
print (data)
---------------------------------------------------------------------

# Writing a txt file 

file = open('1234.txt','w')
file.write('Hello World')
file.write("\n This is our new text file")
file.close()

--------------------------------------------------------------------
# writing a excel file 


data = (
        ['EB Bill', 2000],
        ['Gas', 500],
        ['Food', 5000],
        ['Water', 500],
        ['Cab',2000],
        )
type(data)

import xlsxwriter

#workbook = open('111.xlsx','w')

# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('111.xlsx')
worksheet = workbook.add_worksheet()

row =0
col =0

for item, cost in (data):
    worksheet.write(row, col , item)
    worksheet.write(row,col+1,cost)
    row+=1

workbook.close()

----------------------------------------------------------------------

data = (
        ['EB Bill', 2000],
        ['Gas', 500],
        )
type(data)

import xlsxwriter

#workbook = open('111.xlsx','w')

# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('222.csv')
worksheet = workbook.add_worksheet()

row =0
col =0

for item, cost in (data):
    worksheet.write(row, col , item)
    worksheet.write(row,col+1,cost)
    row+=1

workbook.close()
---------------------------------------------------------------------


