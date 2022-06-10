# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:53:16 2022

@author: dhirp
"""

import pandas as pd
dataset=pd.read_csv("https://raw.githubusercontent.com/edyoda/ML-with-Rishi/main/Social_Network_Ads.csv")
x=dataset.iloc[:,2:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)