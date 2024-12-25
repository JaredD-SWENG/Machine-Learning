"""
Created on Thu Aug 24 20:50:30 2023

@author: Jared Daniel
@email: jjd6385@psu.edu
@assignment: M1 Assignment
"""

M = [[3 for _ in range(90)] for _ in range(100)] #M is a 100 X 90 matrix that is filled with 3's 
O = [[i for i in range(100)] for _ in range(90)] #O is a 90 X 100 matrix with each row going from 0 to 99

n = 5

#Mn is a matrix that is the same as M, except it's first row has been multiplied by n
Mn = []
for i in range(len(M)):
    if i == 0:
        Mn.append([i*n for i in M[i]])
    else:
        Mn.append(M[i])
        
#Mo is the resulting matrix of the product of M and O
Mo = []
for row in M:
    temp = []
    for col in zip(*O): #the zip(*O) allows me to get the columns of O directly
        for i in row:
            summed = 0
            for j in col:
                summed += (i*j)
        temp.append(summed)
    Mo.append(temp)

         