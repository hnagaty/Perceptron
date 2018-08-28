# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 21:51:46 2017

@author: Hany Nagaty
14-Nov-2017
"""

import numpy as np
import copy
from PIL import Image
import string


#mypath='C:\\Users\\Hany\\Dropbox\\Online Courses\\Nu, Machine Learning\\Assigments\\Assignment01\\Assignment 1 Dataset\\Train\\'
mypath='C:\\Users\\hnaga\\Dropbox\\Online Courses\\Nu, Machine Learning\\Assigments\\Assignment01\\Assignment 1 Dataset\\Train\\'


# Read all training letters into a 3d array, where
# d0 --> # of instances of each letter (7)
# d1 --> # of pixels in each letter instance (144)
# d2 --> # of distinct letters (26)

allletters=np.zeros([7,145,1],dtype=int) # empty array. This holds ALL instances of a ALL the lettera
for i in range(97,123,1): # loop through the asci charaters from "a" to "z"
  oneletter=np.zeros([1,145],dtype=int) # empty array. This holds ALL instances of a SINGLE letter
  for j in range(1,8,1): # loop for 7 variants in each letter
    loc=chr(i)+str(j)
    letterW = Image.open(mypath+"A1"+loc+".jpg")
    vec=np.array(letterW).flatten()
    vec=np.append(vec,1)
    oneletter=np.vstack((oneletter,vec))
  oneletter=np.delete(oneletter,0,axis=0) # remove the zero row that was used for initialisation
  allletters=np.dstack((allletters,oneletter))
  #print(loc)
allletters=np.delete(allletters,0 ,axis=2) # remove the zeros plane that was used in initalisation



def tclass(w,x):
    t=w.T@x
    if t==0:
        return (-1)
    else:
        return(abs(t)/t)
    

def testW(l): # Check for 1st misclassified point for "l letter" classifier
    allgood=True
    for i in range(26):
        for j in range (7):
            X=allletters[j,:,i]
            Y=tclass(W,X)
            
            #print ("Running testW. This is letter ",chr(i+97),str(j)," with result ",allgood,sep='')
            if (l==i)==(Y==-1): # not(xor)
                allgood=False
                #print ("Running testW. This is letter ",chr(i+97),str(j)," with result ",allgood,sep='')
                break
        if not(allgood):
            break
    return(allgood,i,j)


# The classifier
Eita=0.05
AllWs=np.zeros([26,145])
for k in range(26):
    W=np.zeros([145,])
    W[0]=1
    i=0
    MisClassState=False # False means there is a misclassified point. True means all points are correctly classified
    while not(MisClassState):
        MisPoint=testW(k)
        MisClassState=MisPoint[0]
        MisPointi=MisPoint[1]
        MisPointj=MisPoint[2]
        X=allletters[MisPointj,:,MisPointi]
        i=i+1
        t=-tclass(W,X)
        W=W+Eita*X*t
        print ("Iteration",i,"for letter",chr(k+97),"with state",MisClassState,W.T@W)
    AllWs[k]=W
#AllWs=np.delete(AllWs,0,axis=0)

#Read the test data
mypath='C:\\Users\\Hany\\Dropbox\\Online Courses\\Nu, Machine Learning\\Assigments\\Assignment01\\Assignment 1 Dataset\\Test\\'
#mypath='C:\\Users\\hnaga\\Dropbox\\Online Courses\\Nu, Machine Learning\\Assigments\\Assignment01\\Assignment 1 Dataset\\Test\\'

testdata=np.zeros([1,145],dtype=int) 
for j in range(8,10,1):
  for i in range(97,123,1):
    loc=chr(i)+str(j)
    letterW = Image.open(mypath+"A1"+loc+".jpg")
    vec=np.array(letterW).flatten()
    vec=np.append(vec,1)
    testdata=np.vstack((testdata,vec))
testdata=np.delete(testdata,0 ,axis=0)

# define classes
TestActualClass = list(string.ascii_lowercase*2)
WClasses = list(string.ascii_lowercase)

def Classify(Wt,S):
    return(Wt.T@S)

# Classify the test data
TestClass=[]
for i in range(52):
    res=np.apply_along_axis(Classify,1,AllWs,testdata[i,:])
    print(TestActualClass[i],"is classified as",WClasses[np.argmax(res)])
    TestClass=TestClass + list(WClasses[np.argmax(res)])

    
# Test the accuracy
Acc=np.array(TestClass)==np.array(TestActualClass)
AccSplit=np.array(np.split(Acc,2))
AccValue=sum(AccSplit)

# Plot the accuracy
import matplotlib.pyplot as plt
x=range(26)
y=AccValue
xlab=list(string.ascii_lowercase)
plt.bar(x, y,align="center")
plt.xticks(x, xlab)
plt.yticks([0,1,2])
plt.show()