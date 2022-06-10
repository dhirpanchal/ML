# -*- coding: utf-8 -*-
"""
Created on Tue May  3 12:21:28 2022

@author: dhirp
"""
import pandas as pd

url="https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt"
hr=pd.read_csv(url)
hr1=hr
cols=hr.columns
	
del hr1["Unnamed: 0"]

print(hr1.info() ,"\n\n", hr1.describe())

hr1.isnull().sum()


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

x=hr1.iloc[:,:-1].values 
y=hr1.iloc[:,-1].values

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.25,random_state=0)
knn=KNeighborsRegressor(n_neighbors=10)
knn.fit(x, y)

y_pred=knn.predict(x_test)
