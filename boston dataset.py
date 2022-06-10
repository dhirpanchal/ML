# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:38:33 2022

@author: dhirp
"""
import pandas as pd

from sklearn.datasets import load_boston
dataset=load_boston()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)

del df["TAX"]

x=dataset.data
y=dataset.target

x1=df.values
y1=dataset.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

corr=df.corr()

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xtrain=sc.fit_transform(x_train)
xtest=sc.transform(x_test)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain,y_train)

print(model.score(xtest,y_test))

#--------------------------------------------------------------
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.3,random_state=0)

sc=StandardScaler()
x1train=sc.fit_transform(x1_train)
x1test=sc.transform(x1_test)

model1=LinearRegression()
model1.fit(x1train,y1_train)

print(model1.score(x1test,y1_test))


