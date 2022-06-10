# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:26:13 2022

@author: dhirp
"""

import pandas as pd

df=pd.read_csv("Fish.csv")
fshcorr=df.corr()

x=df.iloc[:,2:].values
y=df.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)

print(model.score(x_test,y_test))